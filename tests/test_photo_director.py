"""Tests for engine.photo_director.PhotoDirector"""

import pytest

from engine.photo_director import PhotoDirector, PhotoModeUnavailable
from engine.photo_store import PhotoStore


@pytest.fixture
def setting_path(tmp_path):
    path = tmp_path / "setting"
    (path / "characters" / "zoe").mkdir(parents=True)
    (path / "characters" / "zoe" / "bible.md").write_text("Zoe, mid-twenties.", encoding="utf-8")
    (path / "relationship.md").write_text("They met at the circus.", encoding="utf-8")
    (path / "runtime").mkdir(parents=True)
    (path / "runtime" / "continuity.md").write_text("It's late at night.", encoding="utf-8")
    return path


@pytest.fixture
def prompt_file(tmp_path):
    path = tmp_path / "photo_prompt.md"
    path.write_text("Describe a frame for {{self_name}}.", encoding="utf-8")
    return path


@pytest.fixture
def photo_store(tmp_path):
    return PhotoStore(photos_dir=tmp_path / "photos", jsonl_path=tmp_path / "photos.jsonl")


@pytest.fixture
def director(fake_llm, fake_image_provider, photo_store, chat_logger, setting_path, prompt_file):
    return PhotoDirector(
        llm=fake_llm,
        image_provider=fake_image_provider,
        photo_store=photo_store,
        chat_logger=chat_logger,
        setting_path=setting_path,
        self_slug="zoe",
        role_names={"user": "Misha", "assistant": "Zoe"},
        config={"enabled": True, "modes": ["shot", "scene"], "max_chars": 5000},
        prompt_file=prompt_file,
        describer_model="fake-describer",
        template_vars={"self_name": "Zoe", "other_name": "Misha"},
    )


AVAILABLE_RESPONSE = '{"available": true, "prompt": "a phone snapshot of Zoe", "note": "candid"}'
REFUSAL_RESPONSE = '{"available": false, "prompt": "", "note": "she is in the dark and won\'t turn on the light"}'


async def test_available_frame_generates_and_stores_image(director, fake_llm, fake_image_provider):
    fake_llm.content = AVAILABLE_RESPONSE

    entry = await director.generate(mode="shot", hint="")

    assert entry["available"] is True
    assert entry["prompt"] == "a phone snapshot of Zoe"
    assert entry["note"] == "candid"
    assert entry["file"] is not None
    assert entry["image_model"] == fake_image_provider.model_name
    assert fake_image_provider.calls[-1]["prompt"] == "a phone snapshot of Zoe"


async def test_refusal_branch_does_not_call_image_provider(director, fake_llm, fake_image_provider):
    fake_llm.content = REFUSAL_RESPONSE

    entry = await director.generate(mode="shot", hint="")

    assert entry["available"] is False
    assert entry["file"] is None
    assert "dark" in entry["note"]
    assert fake_image_provider.calls == []


async def test_malformed_json_does_not_raise_and_is_recorded_as_unavailable(
    director, fake_llm, fake_image_provider
):
    fake_llm.content = "not json at all, the model rambled instead"

    entry = await director.generate(mode="shot", hint="")

    assert entry["available"] is False
    assert entry["file"] is None
    assert fake_image_provider.calls == []


async def test_json_wrapped_in_markdown_fence_is_still_parsed(director, fake_llm):
    fake_llm.content = f"```json\n{AVAILABLE_RESPONSE}\n```"

    entry = await director.generate(mode="shot", hint="")

    assert entry["available"] is True
    assert entry["prompt"] == "a phone snapshot of Zoe"


async def test_mode_not_in_configured_modes_is_rejected(director, fake_llm, fake_image_provider, photo_store):
    fake_llm.content = AVAILABLE_RESPONSE

    with pytest.raises(PhotoModeUnavailable):
        await director.generate(mode="selfie", hint="")

    assert fake_llm.calls == []
    assert fake_image_provider.calls == []
    assert photo_store.last() is None


async def test_disabled_globally_is_rejected_even_for_a_configured_mode(
    fake_llm, fake_image_provider, photo_store, chat_logger, setting_path, prompt_file
):
    director = PhotoDirector(
        llm=fake_llm,
        image_provider=fake_image_provider,
        photo_store=photo_store,
        chat_logger=chat_logger,
        setting_path=setting_path,
        self_slug="zoe",
        role_names={"user": "Misha", "assistant": "Zoe"},
        config={"enabled": False, "modes": ["shot", "scene"], "max_chars": 5000},
        prompt_file=prompt_file,
        describer_model="fake-describer",
    )
    fake_llm.content = AVAILABLE_RESPONSE

    with pytest.raises(PhotoModeUnavailable):
        await director.generate(mode="shot", hint="")

    assert fake_llm.calls == []
    assert photo_store.last() is None


async def test_hint_is_passed_through_to_the_stored_record(director, fake_llm):
    fake_llm.content = AVAILABLE_RESPONSE

    entry = await director.generate(mode="shot", hint="that night on the roof")

    assert entry["hint"] == "that night on the roof"


async def test_context_includes_bible_relationship_continuity_and_manuscript(
    director, fake_llm, chat_logger
):
    chat_logger.log_message(role="user", content="do you remember the roof?")
    fake_llm.content = AVAILABLE_RESPONSE

    await director.generate(mode="shot", hint="")

    sent = fake_llm.calls[-1]["messages"]
    user_content = sent[1]["content"]
    assert "Zoe, mid-twenties." in user_content
    assert "They met at the circus." in user_content
    assert "It's late at night." in user_content
    assert "do you remember the roof?" in user_content


async def test_debug_prompt_path_is_written_when_given(director, fake_llm, tmp_path):
    debug_path = tmp_path / "debug_out" / "last_photo_prompt.txt"
    director.debug_prompt_path = debug_path
    fake_llm.content = AVAILABLE_RESPONSE

    await director.generate(mode="shot", hint="")

    assert debug_path.exists()


async def test_temperature_and_max_tokens_come_from_config(
    fake_llm, fake_image_provider, photo_store, chat_logger, setting_path, prompt_file
):
    director = PhotoDirector(
        llm=fake_llm,
        image_provider=fake_image_provider,
        photo_store=photo_store,
        chat_logger=chat_logger,
        setting_path=setting_path,
        self_slug="zoe",
        role_names={"user": "Misha", "assistant": "Zoe"},
        config={
            "enabled": True,
            "modes": ["shot"],
            "max_chars": 5000,
            "llm": {"temperature": 0.4, "max_tokens": 321, "reasoning_effort": "minimal"},
        },
        prompt_file=prompt_file,
        describer_model="fake-describer",
    )
    fake_llm.content = AVAILABLE_RESPONSE

    await director.generate(mode="shot", hint="")

    call = fake_llm.calls[-1]
    assert call["temperature"] == 0.4
    assert call["max_tokens"] == 321
    assert call["reasoning_effort"] == "minimal"


async def test_reasoning_effort_comes_from_config_and_defaults_to_none(director, fake_llm):
    """Verified live against google/gemini-3.8-flash (2026-09-14): without an
    explicit reasoning_effort, the model burns most of max_tokens on hidden
    reasoning before emitting JSON and the response truncates. The `director`
    fixture's config has no llm.reasoning_effort key, so this also pins the
    default (None -> provider decides) for setups that don't need it."""
    fake_llm.content = AVAILABLE_RESPONSE
    await director.generate(mode="shot", hint="")
    assert fake_llm.calls[-1]["reasoning_effort"] is None


# --- again() ---


async def test_again_with_no_prior_frame_returns_none(director):
    assert await director.again() is None


async def test_again_resubmits_last_prompt_without_calling_the_describer(
    director, fake_llm, fake_image_provider
):
    fake_llm.content = AVAILABLE_RESPONSE
    await director.generate(mode="shot", hint="")
    calls_after_first = len(fake_llm.calls)

    entry = await director.again()

    assert len(fake_llm.calls) == calls_after_first  # describer not called again
    assert entry["prompt"] == "a phone snapshot of Zoe"
    assert entry["available"] is True
    assert entry["id"] == 2


async def test_again_with_modifier_appends_to_the_last_prompt(director, fake_llm):
    fake_llm.content = AVAILABLE_RESPONSE
    await director.generate(mode="shot", hint="")

    entry = await director.again("more light")

    assert entry["prompt"] == "a phone snapshot of Zoe more light"


async def test_again_skips_a_trailing_refusal_and_reuses_the_last_delivered_frame(
    director, fake_llm
):
    fake_llm.content = AVAILABLE_RESPONSE
    await director.generate(mode="shot", hint="")
    fake_llm.content = REFUSAL_RESPONSE
    await director.generate(mode="shot", hint="")

    entry = await director.again()

    assert entry["prompt"] == "a phone snapshot of Zoe"


async def test_again_raises_if_disabled_globally(
    fake_llm, fake_image_provider, photo_store, chat_logger, setting_path, prompt_file
):
    director = PhotoDirector(
        llm=fake_llm,
        image_provider=fake_image_provider,
        photo_store=photo_store,
        chat_logger=chat_logger,
        setting_path=setting_path,
        self_slug="zoe",
        role_names={"user": "Misha", "assistant": "Zoe"},
        config={"enabled": False, "modes": ["shot"], "max_chars": 5000},
        prompt_file=prompt_file,
        describer_model="fake-describer",
    )

    with pytest.raises(PhotoModeUnavailable):
        await director.again()


# --- face reference ---

FACE_BYTES = b"\x89PNG\r\n\x1a\nnot-a-real-png-but-enough-for-a-fake-provider"


def _director_with(
    fake_llm, fake_image_provider, photo_store, chat_logger, setting_path, prompt_file, modes
):
    return PhotoDirector(
        llm=fake_llm,
        image_provider=fake_image_provider,
        photo_store=photo_store,
        chat_logger=chat_logger,
        setting_path=setting_path,
        self_slug="zoe",
        role_names={"user": "Misha", "assistant": "Zoe"},
        config={"enabled": True, "modes": modes, "max_chars": 5000},
        prompt_file=prompt_file,
        describer_model="fake-describer",
    )


def _write_face(setting_path, filename="face.png"):
    path = setting_path / "characters" / "zoe" / filename
    path.write_bytes(FACE_BYTES)
    return path


async def test_selfie_sends_the_face_as_a_reference(
    fake_llm, fake_image_provider, photo_store, chat_logger, setting_path, prompt_file
):
    _write_face(setting_path)
    director = _director_with(
        fake_llm, fake_image_provider, photo_store, chat_logger, setting_path, prompt_file,
        ["selfie"],
    )
    fake_llm.content = AVAILABLE_RESPONSE

    entry = await director.generate(mode="selfie", hint="")

    assert fake_image_provider.calls[-1]["reference_images"] == [FACE_BYTES]
    assert entry["reference"] == "face.png"


async def test_shot_never_sends_a_reference_even_when_a_face_exists(
    fake_llm, fake_image_provider, photo_store, chat_logger, setting_path, prompt_file
):
    """Her face isn't in frame by definition - attaching one would just pay
    the provider's per-reference fee for nothing."""
    _write_face(setting_path)
    director = _director_with(
        fake_llm, fake_image_provider, photo_store, chat_logger, setting_path, prompt_file,
        ["shot"],
    )
    fake_llm.content = AVAILABLE_RESPONSE

    entry = await director.generate(mode="shot", hint="")

    assert fake_image_provider.calls[-1]["reference_images"] is None
    assert entry["reference"] is None


async def test_scene_uses_a_face_when_present(
    fake_llm, fake_image_provider, photo_store, chat_logger, setting_path, prompt_file
):
    _write_face(setting_path)
    director = _director_with(
        fake_llm, fake_image_provider, photo_store, chat_logger, setting_path, prompt_file,
        ["scene"],
    )
    fake_llm.content = AVAILABLE_RESPONSE

    await director.generate(mode="scene", hint="")

    assert fake_image_provider.calls[-1]["reference_images"] == [FACE_BYTES]


async def test_scene_still_works_without_a_face(
    fake_llm, fake_image_provider, photo_store, chat_logger, setting_path, prompt_file
):
    """/scene shipped and ran live before faces existed; requiring one would
    break a working mode in every setting that hasn't got a face yet."""
    director = _director_with(
        fake_llm, fake_image_provider, photo_store, chat_logger, setting_path, prompt_file,
        ["scene"],
    )
    fake_llm.content = AVAILABLE_RESPONSE

    entry = await director.generate(mode="scene", hint="")

    assert entry["available"] is True
    assert entry["reference"] is None
    assert fake_image_provider.calls[-1]["reference_images"] is None


async def test_selfie_without_a_face_fails_loudly_at_construction(
    fake_llm, fake_image_provider, photo_store, chat_logger, setting_path, prompt_file
):
    """At startup, not at the first command - the CLAUDE.md empty-bible trap."""
    with pytest.raises(ValueError, match="face"):
        _director_with(
            fake_llm, fake_image_provider, photo_store, chat_logger, setting_path, prompt_file,
            ["shot", "selfie"],
        )


async def test_a_jpeg_face_artifact_is_found_too(
    fake_llm, fake_image_provider, photo_store, chat_logger, setting_path, prompt_file
):
    """The image models return JPEG and there's nothing here to convert with."""
    _write_face(setting_path, "face.jpeg")
    director = _director_with(
        fake_llm, fake_image_provider, photo_store, chat_logger, setting_path, prompt_file,
        ["selfie"],
    )
    fake_llm.content = AVAILABLE_RESPONSE

    entry = await director.generate(mode="selfie", hint="")

    assert entry["reference"] == "face.jpeg"


async def test_context_tells_the_describer_whether_a_reference_is_attached(
    fake_llm, fake_image_provider, photo_store, chat_logger, setting_path, prompt_file
):
    _write_face(setting_path)
    director = _director_with(
        fake_llm, fake_image_provider, photo_store, chat_logger, setting_path, prompt_file,
        ["scene", "shot"],
    )
    fake_llm.content = AVAILABLE_RESPONSE

    await director.generate(mode="scene", hint="")
    assert "REFERENCE: attached" in fake_llm.calls[-1]["messages"][1]["content"]

    await director.generate(mode="shot", hint="")
    assert "REFERENCE: none attached" in fake_llm.calls[-1]["messages"][1]["content"]


async def test_a_refusal_records_no_reference(
    fake_llm, fake_image_provider, photo_store, chat_logger, setting_path, prompt_file
):
    """Same rule as image_model/cost: input fields are recorded only for a
    call that actually happened."""
    _write_face(setting_path)
    director = _director_with(
        fake_llm, fake_image_provider, photo_store, chat_logger, setting_path, prompt_file,
        ["selfie"],
    )
    fake_llm.content = REFUSAL_RESPONSE

    entry = await director.generate(mode="selfie", hint="")

    assert entry["available"] is False
    assert entry["reference"] is None


async def test_again_reattaches_the_face_of_the_mode_it_repeats(
    fake_llm, fake_image_provider, photo_store, chat_logger, setting_path, prompt_file
):
    _write_face(setting_path)
    director = _director_with(
        fake_llm, fake_image_provider, photo_store, chat_logger, setting_path, prompt_file,
        ["selfie"],
    )
    fake_llm.content = AVAILABLE_RESPONSE
    await director.generate(mode="selfie", hint="")

    entry = await director.again("more light")

    assert fake_image_provider.calls[-1]["reference_images"] == [FACE_BYTES]
    assert entry["reference"] == "face.png"


async def test_again_picks_up_a_face_added_after_the_original_frame(
    fake_llm, fake_image_provider, photo_store, chat_logger, setting_path, prompt_file
):
    """Re-rolling an old frame against a newly chosen face is the natural way
    to check a replacement, so the reference is resolved fresh, not copied."""
    director = _director_with(
        fake_llm, fake_image_provider, photo_store, chat_logger, setting_path, prompt_file,
        ["scene"],
    )
    fake_llm.content = AVAILABLE_RESPONSE
    first = await director.generate(mode="scene", hint="")
    assert first["reference"] is None

    _write_face(setting_path)
    entry = await director.again()

    assert entry["reference"] == "face.png"
