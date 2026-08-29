from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image

DEFAULT_CLIP_MODEL = "openai/clip-vit-large-patch14"

FEMALE_PROMPTS = (
    "a photo of a woman's face",
    "a photo of a girl's face",
    "a close-up photo of a woman",
)

MALE_PROMPTS = (
    "a photo of a man's face",
    "a photo of a boy's face",
    "a close-up photo of a man",
)

ALL_PROMPTS = (*FEMALE_PROMPTS, *MALE_PROMPTS)

FEMALE = "female"
MALE = "male"

CROP_MARGIN = 0.25


@dataclass
class ClipBundle:
    """A loaded CLIP model together with its cached text embeddings."""

    model: Any
    processor: Any
    device: str
    dtype: Any
    text_embeds: Any


_model_cache: dict[tuple[str, str], ClipBundle] = {}
_cache_lock = threading.Lock()


def crop_with_margin(
    image: Image.Image, bbox: list[float], margin: float = CROP_MARGIN
) -> Image.Image:
    """
    Crop `bbox` out of `image`, expanded by `margin` (a fraction of the bbox
    width/height) on every side and clamped to the image bounds.

    Parameters
    ----------
        image: PIL.Image.Image
            the source image
        bbox: list[float]
            [x1, y1, x2, y2]
        margin: float
            extra margin on each side, as a fraction of the bbox size

    Returns
    -------
        PIL.Image.Image
    """
    if len(bbox) != 4:
        msg = f"bbox length must be 4, got {len(bbox)}"
        raise ValueError(msg)

    width, height = image.size
    x1, y1, x2, y2 = (float(v) for v in bbox)

    dx = (x2 - x1) * margin
    dy = (y2 - y1) * margin

    left = round(x1 - dx)
    top = round(y1 - dy)
    right = round(x2 + dx)
    bottom = round(y2 + dy)

    # clamp to the image, always keeping at least a 1px box
    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    right = max(left + 1, min(right, width))
    bottom = max(top + 1, min(bottom, height))

    return image.crop((left, top, right, bottom))


def extract_embeds(output: Any) -> Any:
    """
    `CLIPModel.get_text_features` / `get_image_features` return a plain tensor
    on transformers 4.x, but a `BaseModelOutputWithPooling` (whose
    `pooler_output` holds the projected embeddings) on transformers 5.x.
    """
    if hasattr(output, "norm") and hasattr(output, "shape"):
        # already a tensor
        return output

    pooler_output = getattr(output, "pooler_output", None)
    if pooler_output is not None:
        return pooler_output

    if isinstance(output, (tuple, list)) and output:
        return output[0]

    msg = f"unexpected CLIP feature output: {type(output).__name__}"
    raise TypeError(msg)


def _load_model(model_name: str, device: str) -> ClipBundle:
    """
    Load a CLIP model and pre-compute the (normalized) text embeddings of
    `ALL_PROMPTS`. Exceptions are intentionally not caught here.
    """
    import torch
    from transformers import CLIPModel, CLIPProcessor

    dev = device or "cpu"
    dtype = torch.float16 if str(dev).startswith("cuda") else torch.float32

    model = CLIPModel.from_pretrained(model_name, torch_dtype=dtype)
    model = model.to(dev)
    model.eval()

    processor = CLIPProcessor.from_pretrained(model_name)

    with torch.no_grad():
        text_inputs = processor(
            text=list(ALL_PROMPTS), return_tensors="pt", padding=True
        )
        text_inputs = {k: v.to(dev) for k, v in text_inputs.items() if hasattr(v, "to")}
        text_embeds = extract_embeds(model.get_text_features(**text_inputs))
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    return ClipBundle(
        model=model,
        processor=processor,
        device=dev,
        dtype=dtype,
        text_embeds=text_embeds,
    )


def get_model(model_name: str, device: str) -> ClipBundle:
    "Lazy singleton cache, keyed by (model_name, device)."
    key = (model_name, str(device or "cpu"))
    with _cache_lock:
        bundle = _model_cache.get(key)
        if bundle is None:
            bundle = _load_model(model_name, key[1])
            _model_cache[key] = bundle
        return bundle


def clear_cache() -> None:
    with _cache_lock:
        _model_cache.clear()


def _group_probabilities(probs: Any) -> tuple[str, float]:
    """
    `probs` is a sequence of per-prompt probabilities laid out as
    `ALL_PROMPTS`: female prompts first, male prompts second.
    """
    n_female = len(FEMALE_PROMPTS)
    # clamped: summing float32 softmax outputs can overshoot 1.0 slightly
    female = min(1.0, max(0.0, sum(float(p) for p in probs[:n_female])))
    male = min(1.0, max(0.0, sum(float(p) for p in probs[n_female:])))

    if female >= male:
        return (FEMALE, female)
    return (MALE, male)


def classify_faces(
    image: Image.Image,
    bboxes: list[list[float]],
    *,
    model_name: str,
    device: str,
) -> list[tuple[str, float]]:
    """
    Classify each bbox of `image` as "female" or "male" with CLIP zero-shot.

    Every bbox is cropped with a 25% margin, all crops are run through CLIP in
    a single batch, the image-text logits are softmaxed over all six prompts
    and the per-group probabilities are summed.

    Parameters
    ----------
        image: PIL.Image.Image
            the image the bboxes were detected on
        bboxes: list[list[float]]
            list of [x1, y1, x2, y2]
        model_name: str
            a Hugging Face CLIP model id
        device: str
            torch device to run on

    Returns
    -------
        list[tuple[str, float]]
            ("female" | "male", probability of the winning group) per bbox
    """
    if not bboxes:
        return []

    import torch

    bundle = get_model(model_name, device)

    if image.mode != "RGB":
        image = image.convert("RGB")

    crops = [crop_with_margin(image, bbox) for bbox in bboxes]

    with torch.no_grad():
        image_inputs = bundle.processor(images=crops, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].to(
            device=bundle.device, dtype=bundle.dtype
        )

        image_embeds = extract_embeds(
            bundle.model.get_image_features(pixel_values=pixel_values)
        )
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

        logit_scale = bundle.model.logit_scale.exp()
        text_embeds = bundle.text_embeds
        logits = logit_scale * image_embeds.to(text_embeds.dtype) @ text_embeds.t()
        probs = logits.float().softmax(dim=-1).cpu().tolist()

    return [_group_probabilities(row) for row in probs]
