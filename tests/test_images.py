"""Tests for the ImageProvider contract, via the Fake implementation.

No network calls here - engine.images.openrouter.OpenRouterImageProvider is
verified manually against the real API instead (scripts/probe_image_models.py
renders the same prompt through every candidate model for comparison).
"""


async def test_generate_returns_bytes(fake_image_provider):
    image = await fake_image_provider.generate("a phone snapshot")
    assert isinstance(image, bytes)
    assert image == fake_image_provider.image_bytes


async def test_generate_records_the_call(fake_image_provider):
    await fake_image_provider.generate("candid backstage shot")
    assert fake_image_provider.calls == [
        {"prompt": "candid backstage shot", "reference_images": None}
    ]


async def test_generate_records_reference_images(fake_image_provider):
    refs = [b"face-bytes"]
    await fake_image_provider.generate("selfie", reference_images=refs)
    assert fake_image_provider.calls[-1]["reference_images"] == refs


async def test_generate_is_deterministic(fake_image_provider):
    first = await fake_image_provider.generate("same prompt")
    second = await fake_image_provider.generate("same prompt")
    assert first == second


# --- payload assembly for OpenRouterImageProvider ---
#
# The HTTP round trip stays untested (thin wrapper, project convention), but
# the reference-image encoding rules are real logic: they decide whether a
# frame costs an extra $0.01, and a wrong media type fails silently at the
# provider.

import base64

import pytest

from engine.images.openrouter import _build_payload, _data_url

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 16


def test_data_url_detects_png():
    assert _data_url(PNG_BYTES).startswith("data:image/png;base64,")


def test_data_url_detects_jpeg_regardless_of_any_filename():
    """The image models return JPEG, so a face artifact named
    face.png may well be a JPEG - media type must come from the bytes."""
    assert _data_url(JPEG_BYTES).startswith("data:image/jpeg;base64,")


def test_data_url_rejects_something_that_is_not_an_image():
    with pytest.raises(ValueError):
        _data_url(b"this is a text file, not an image")


def test_payload_without_references_has_no_input_references_key():
    """Absent, not empty: references are billed per image, and a frame that
    doesn't use one must not start carrying the field."""
    payload = _build_payload("some/model", "a phone snapshot", None, 3)
    assert payload == {"model": "some/model", "prompt": "a phone snapshot"}


def test_payload_with_empty_reference_list_also_omits_the_key():
    payload = _build_payload("some/model", "a phone snapshot", [], 3)
    assert "input_references" not in payload


def test_payload_encodes_references_in_the_shape_the_api_expects():
    payload = _build_payload("some/model", "a selfie", [PNG_BYTES], 3)
    assert payload["input_references"] == [
        {"type": "image_url", "image_url": {"url": _data_url(PNG_BYTES)}}
    ]


def test_payload_truncates_to_the_models_reference_limit():
    payload = _build_payload("some/model", "a selfie", [PNG_BYTES, JPEG_BYTES, PNG_BYTES], 1)
    assert len(payload["input_references"]) == 1
