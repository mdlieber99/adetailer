from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

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
ANY = "any"

CROP_MARGIN = 0.25


def parse_face_filter(value: str) -> tuple[str, int | None]:
    """
    Split a face filter choice into its gender and its optional ordinal.

    "Female 2" -> ("female", 2), "Male" -> ("male", None), "Any" -> ("any", None).
    Casing and extra whitespace are ignored; anything unrecognized is treated
    as "any", i.e. no filtering.

    Parameters
    ----------
        value: str
            one of the `Face filter` dropdown choices

    Returns
    -------
        tuple[str, int | None]
            (gender, 1-based ordinal or None)
    """
    parts = str(value or "").strip().lower().split()
    if not parts:
        return (ANY, None)

    gender, rest = parts[0], parts[1:]
    if gender not in (ANY, FEMALE, MALE):
        return (ANY, None)

    if gender == ANY or len(rest) != 1:
        return (gender, None)

    ordinal = rest[0]
    if ordinal.isdecimal() and int(ordinal) > 0:
        return (gender, int(ordinal))

    return (gender, None)


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


def face_filter_decisions(  # noqa: PLR0913
    results: list[tuple[str, float]],
    bboxes: list[list[float]],
    *,
    wanted: str,
    ordinal: int | None,
    min_confidence: float,
    keep_unknown: bool,
    sort_key: Callable[[list[float]], Any] | None,
) -> tuple[list[int], list[str]]:
    """
    Decide which classified faces the tab keeps, and describe every decision.

    Faces whose winning probability is below `min_confidence` count as
    *unknown* and are kept only when `keep_unknown` is set; unknown faces kept
    this way count as members of `wanted` for ordinal purposes.

    When `ordinal` is given, the faces of the wanted gender are ranked with
    `sort_key` -- the very key `adetailer.mask.sort_bboxes` would use -- and
    only the `ordinal`-th (1-based) one is kept. Fewer faces than the ordinal
    means nothing is kept.

    Parameters
    ----------
        results: list[tuple[str, float]]
            per-bbox (label, probability), as `classify_faces` returns
        bboxes: list[list[float]]
            the bboxes the results belong to, in detection order
        wanted: str
            "female" or "male"
        ordinal: int | None
            1-based ordinal within `wanted`, or None to keep them all
        min_confidence: float
            below this a face counts as unknown
        keep_unknown: bool
            whether this tab is the one handling unknown faces
        sort_key: Callable | None
            key for a single bbox, or None when the sort order is "None"

    Returns
    -------
        tuple[list[int], list[str]]
            indices to keep, and one human-readable log entry per face
    """
    # candidates: the faces belonging to this tab's gender
    candidates: list[int] = []
    texts: list[str] = []
    matches: list[bool] = []
    for j, (label, confidence) in enumerate(results):
        if confidence < min_confidence:
            matched = keep_unknown
            text = f"face {j + 1} unknown ({label} {confidence:.2f})"
        else:
            matched = label.lower() == wanted
            text = f"face {j + 1} {label} {confidence:.2f}"

        texts.append(text)
        matches.append(matched)
        if matched:
            candidates.append(j)

    # rank the candidates the way the final masks will be sorted, so
    # "Female 2" is the second woman in bounding-box sort order
    ranks: dict[int, int] = {}
    if ordinal is not None and candidates:
        order = (
            list(candidates)
            if sort_key is None
            else sorted(candidates, key=lambda i: sort_key(bboxes[i]))
        )
        ranks = {j: rank for rank, j in enumerate(order, start=1)}

    keep: list[int] = []
    logs: list[str] = []
    for j, matched in enumerate(matches):
        decision = matched
        note = ""
        if matched and ordinal is not None:
            rank = ranks[j]
            note = f" ({wanted} #{rank})"
            decision = rank == ordinal

        if decision:
            keep.append(j)
        logs.append(f"{texts[j]} -> {'keep' if decision else 'skip'}{note}")

    return keep, logs


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
