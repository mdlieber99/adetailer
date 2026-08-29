from __future__ import annotations

import math

import pytest
import torch
from PIL import Image

from adetailer.common import PredictOutput
from adetailer.gender import (
    ALL_PROMPTS,
    FEMALE_PROMPTS,
    ClipBundle,
    classify_faces,
    clear_cache,
    crop_with_margin,
    extract_embeds,
)
from adetailer.mask import filter_by_indices

MODEL_NAME = "fake/clip"
DEVICE = "cpu"

FEMALE_VECTOR = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
MALE_VECTOR = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
TIE_VECTOR = [1.0] * 6


@pytest.fixture(autouse=True)
def _clear_model_cache():
    clear_cache()
    yield
    clear_cache()


class FakeProcessor:
    """Records the crops it is given, so batching can be asserted."""

    def __init__(self):
        self.image_calls: list[list[Image.Image]] = []

    def __call__(self, images=None, text=None, return_tensors=None, padding=None):
        images = list(images)
        self.image_calls.append(images)
        return {"pixel_values": torch.zeros(len(images), 3, 2, 2)}


class FakeModel:
    """Returns fixed per-crop feature vectors instead of running CLIP."""

    def __init__(self, vectors: list[list[float]]):
        self.vectors = vectors
        # exp(log(100)) == 100, the usual CLIP logit scale
        self.logit_scale = torch.tensor(math.log(100.0))

    def get_image_features(self, *, pixel_values):
        n = pixel_values.shape[0]
        return torch.tensor(self.vectors[:n], dtype=torch.float32)


def make_loader(vectors: list[list[float]]):
    """Build a replacement for `gender._load_model` returning fixed logits.

    `text_embeds` is the identity matrix, so the image-text logits are just
    `logit_scale * normalize(vector)`, in the order of `ALL_PROMPTS`.
    """
    processor = FakeProcessor()

    def _loader(model_name: str, device: str) -> ClipBundle:
        return ClipBundle(
            model=FakeModel(vectors),
            processor=processor,
            device=device,
            dtype=torch.float32,
            text_embeds=torch.eye(len(ALL_PROMPTS), dtype=torch.float32),
        )

    return _loader, processor


# --- crop_with_margin ---------------------------------------------------


def test_crop_with_margin_adds_25_percent():
    image = Image.new("RGB", (100, 100))
    crop = crop_with_margin(image, [40, 40, 60, 60])
    # 25% of a 20px box is 5px on each side
    assert crop.size == (30, 30)


def test_crop_with_margin_clamps_to_bottom_right():
    image = Image.new("RGB", (100, 100))
    crop = crop_with_margin(image, [80, 80, 100, 100])
    # left/top = 75, right/bottom would be 105 but is clamped to 100
    assert crop.size == (25, 25)


def test_crop_with_margin_clamps_to_top_left():
    image = Image.new("RGB", (100, 100))
    crop = crop_with_margin(image, [-40, -40, 20, 20])
    # left/top would be -55 but is clamped to 0, right/bottom = 35
    assert crop.size == (35, 35)


def test_crop_with_margin_never_empty():
    image = Image.new("RGB", (10, 10))
    crop = crop_with_margin(image, [9.6, 9.6, 9.9, 9.9])
    assert crop.size[0] >= 1
    assert crop.size[1] >= 1


def test_crop_with_margin_bad_bbox():
    image = Image.new("RGB", (10, 10))
    with pytest.raises(ValueError, match="bbox length must be 4"):
        crop_with_margin(image, [1, 2, 3])


# --- extract_embeds -----------------------------------------------------


def test_extract_embeds_tensor():
    "transformers 4.x returns a plain tensor."
    t = torch.ones(2, 4)
    assert extract_embeds(t) is t


def test_extract_embeds_output_object():
    "transformers 5.x returns an output object carrying pooler_output."

    class Output:
        def __init__(self, pooled):
            self.pooler_output = pooled
            self.last_hidden_state = torch.zeros(2, 3, 4)

    pooled = torch.ones(2, 4)
    assert extract_embeds(Output(pooled)) is pooled


def test_extract_embeds_tuple():
    t = torch.ones(2, 4)
    assert extract_embeds((t, None)) is t


def test_extract_embeds_unknown():
    with pytest.raises(TypeError, match="unexpected CLIP feature output"):
        extract_embeds(object())


# --- classify_faces -----------------------------------------------------


def test_classify_faces_no_bboxes():
    assert (
        classify_faces(
            Image.new("RGB", (10, 10)), [], model_name=MODEL_NAME, device=DEVICE
        )
        == []
    )


def test_classify_faces_group_summation(monkeypatch):
    from adetailer import gender

    loader, _ = make_loader([FEMALE_VECTOR, MALE_VECTOR])
    monkeypatch.setattr(gender, "_load_model", loader)

    image = Image.new("RGB", (100, 100))
    result = classify_faces(
        image,
        [[10, 10, 40, 40], [50, 50, 80, 80]],
        model_name=MODEL_NAME,
        device=DEVICE,
    )

    assert [label for label, _ in result] == ["female", "male"]
    for _, confidence in result:
        assert confidence > 0.9
        assert confidence <= 1.0


def test_classify_faces_probabilities_sum_within_one(monkeypatch):
    from adetailer import gender

    loader, _ = make_loader([TIE_VECTOR])
    monkeypatch.setattr(gender, "_load_model", loader)

    result = classify_faces(
        Image.new("RGB", (100, 100)),
        [[10, 10, 40, 40]],
        model_name=MODEL_NAME,
        device=DEVICE,
    )

    # every prompt is equally likely: both groups get 3/6
    label, confidence = result[0]
    assert label == "female"  # ties go to female
    assert confidence == pytest.approx(0.5, abs=1e-4)
    assert len(FEMALE_PROMPTS) * 2 == len(ALL_PROMPTS)


def test_classify_faces_runs_one_batch(monkeypatch):
    from adetailer import gender

    loader, processor = make_loader([FEMALE_VECTOR, MALE_VECTOR, FEMALE_VECTOR])
    monkeypatch.setattr(gender, "_load_model", loader)

    bboxes = [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]]
    result = classify_faces(
        Image.new("RGB", (100, 100)), bboxes, model_name=MODEL_NAME, device=DEVICE
    )

    assert len(result) == 3
    assert len(processor.image_calls) == 1
    assert len(processor.image_calls[0]) == 3


def test_classify_faces_model_is_cached(monkeypatch):
    from adetailer import gender

    calls = []
    loader, _ = make_loader([FEMALE_VECTOR])

    def counting_loader(model_name: str, device: str):
        calls.append((model_name, device))
        return loader(model_name, device)

    monkeypatch.setattr(gender, "_load_model", counting_loader)

    image = Image.new("RGB", (100, 100))
    for _ in range(3):
        classify_faces(image, [[0, 0, 10, 10]], model_name=MODEL_NAME, device=DEVICE)

    assert len(calls) == 1


# --- filter_by_indices --------------------------------------------------


def _fake_pred(n: int) -> PredictOutput:
    return PredictOutput(
        bboxes=[[float(i), float(i), float(i + 10), float(i + 10)] for i in range(n)],
        masks=[Image.new("L", (10, 10), i) for i in range(n)],
        confidences=[0.1 * i for i in range(n)],
        preview=Image.new("RGB", (10, 10)),
    )


def test_filter_by_indices_keeps_selected():
    pred = _fake_pred(3)
    preview = pred.preview

    result = filter_by_indices(pred, [0, 2])

    assert len(result.bboxes) == 2
    assert result.bboxes[0][0] == 0.0
    assert result.bboxes[1][0] == 2.0
    assert len(result.masks) == 2
    assert result.confidences == pytest.approx([0.0, 0.2])
    assert result.preview is preview


def test_filter_by_indices_empty():
    result = filter_by_indices(_fake_pred(3), [])
    assert result.bboxes == []
    assert result.masks == []
    assert result.confidences == []


def test_filter_by_indices_all():
    result = filter_by_indices(_fake_pred(2), [0, 1])
    assert len(result.bboxes) == 2
    assert len(result.masks) == 2
    assert len(result.confidences) == 2


def test_filter_by_indices_ignores_out_of_range():
    result = filter_by_indices(_fake_pred(2), [1, 5, -1])
    assert len(result.bboxes) == 1
    assert result.bboxes[0][0] == 1.0
