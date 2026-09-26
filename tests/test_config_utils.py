"""Tests for engine.config_utils"""

from engine.config_utils import (
    list_settings,
    read_setting_characters,
    resolve_images_config,
    resolve_llm_config,
    write_active_setting,
)


def test_no_override_file_returns_base_config(tmp_path):
    base = {"provider": "openrouter", "model": "google/gemini-3.8-flash", "temperature": 0.85}

    result = resolve_llm_config(tmp_path, "some_setting", base)

    assert result == base


def test_override_replaces_only_specified_keys(tmp_path):
    setting_dir = tmp_path / "soul" / "settings" / "some_setting"
    setting_dir.mkdir(parents=True)
    (setting_dir / "llm.yaml").write_text("model: mistral-large\n", encoding="utf-8")
    base = {"provider": "openrouter", "model": "google/gemini-3.8-flash", "temperature": 0.85}

    result = resolve_llm_config(tmp_path, "some_setting", base)

    assert result == {"provider": "openrouter", "model": "mistral-large", "temperature": 0.85}


def test_override_can_add_and_replace_multiple_keys(tmp_path):
    setting_dir = tmp_path / "soul" / "settings" / "some_setting"
    setting_dir.mkdir(parents=True)
    (setting_dir / "llm.yaml").write_text(
        "provider: mistral\nmodel: mistral-large\nreasoning_effort: high\n", encoding="utf-8"
    )
    base = {"provider": "openrouter", "model": "google/gemini-3.8-flash", "temperature": 0.85}

    result = resolve_llm_config(tmp_path, "some_setting", base)

    assert result == {
        "provider": "mistral",
        "model": "mistral-large",
        "temperature": 0.85,
        "reasoning_effort": "high",
    }


def test_empty_override_file_returns_base_config(tmp_path):
    setting_dir = tmp_path / "soul" / "settings" / "some_setting"
    setting_dir.mkdir(parents=True)
    (setting_dir / "llm.yaml").write_text("", encoding="utf-8")
    base = {"provider": "openrouter", "model": "google/gemini-3.8-flash"}

    result = resolve_llm_config(tmp_path, "some_setting", base)

    assert result == base


def test_list_settings_returns_sorted_directory_names(tmp_path):
    settings_dir = tmp_path / "soul" / "settings"
    (settings_dir / "zeta").mkdir(parents=True)
    (settings_dir / "alpha").mkdir(parents=True)
    (settings_dir / "not_a_dir.txt").write_text("", encoding="utf-8")

    assert list_settings(tmp_path) == ["alpha", "zeta"]


def test_list_settings_missing_directory_returns_empty(tmp_path):
    assert list_settings(tmp_path) == []


def test_write_active_setting_replaces_setting_and_pairing_keeping_comments(tmp_path):
    config_path = tmp_path / "soul" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "# comment kept\nsetting: old_setting\ncharacters:\n  self: old_self\n  other: old_other\n",
        encoding="utf-8",
    )

    write_active_setting(tmp_path, "new_setting", "new_self", "new_other")

    text = config_path.read_text(encoding="utf-8")
    assert "setting: new_setting" in text
    assert "self: new_self" in text
    assert "other: new_other" in text
    assert "# comment kept" in text


def test_read_setting_characters_returns_declared_pairing(tmp_path):
    setting_dir = tmp_path / "soul" / "settings" / "some_setting"
    setting_dir.mkdir(parents=True)
    (setting_dir / "characters.yaml").write_text("self: amy\nother: adam\n", encoding="utf-8")

    assert read_setting_characters(tmp_path, "some_setting") == {"self": "amy", "other": "adam"}


def test_read_setting_characters_missing_file_returns_empty(tmp_path):
    assert read_setting_characters(tmp_path, "some_setting") == {}


def test_read_setting_characters_partial_declaration_returns_only_declared(tmp_path):
    setting_dir = tmp_path / "soul" / "settings" / "some_setting"
    setting_dir.mkdir(parents=True)
    (setting_dir / "characters.yaml").write_text("self: amy\n", encoding="utf-8")

    assert read_setting_characters(tmp_path, "some_setting") == {"self": "amy"}


def test_resolve_images_no_override_file_disables_photos_for_this_setting(tmp_path):
    base = {"enabled": True, "provider": "openrouter", "model": "some/image-model"}

    result = resolve_images_config(tmp_path, "some_setting", base)

    assert result == {**base, "modes": []}


def test_resolve_images_override_declares_modes_and_replaces_model(tmp_path):
    setting_dir = tmp_path / "soul" / "settings" / "some_setting"
    setting_dir.mkdir(parents=True)
    (setting_dir / "images.yaml").write_text(
        "modes: [shot, scene]\nmodel: other/image-model\n", encoding="utf-8"
    )
    base = {"enabled": True, "provider": "openrouter", "model": "some/image-model"}

    result = resolve_images_config(tmp_path, "some_setting", base)

    assert result == {
        "enabled": True,
        "provider": "openrouter",
        "model": "other/image-model",
        "modes": ["shot", "scene"],
        "llm": {},
    }


def test_resolve_images_llm_block_merges_key_by_key(tmp_path):
    setting_dir = tmp_path / "soul" / "settings" / "some_setting"
    setting_dir.mkdir(parents=True)
    (setting_dir / "images.yaml").write_text(
        "modes: [shot]\nllm:\n  model: cheap/describer\n", encoding="utf-8"
    )
    base = {
        "enabled": True,
        "model": "some/image-model",
        "llm": {"provider": "openrouter", "model": "google/gemini-3.8-flash", "temperature": 0.8},
    }

    result = resolve_images_config(tmp_path, "some_setting", base)

    assert result["llm"] == {"provider": "openrouter", "model": "cheap/describer", "temperature": 0.8}
    assert result["modes"] == ["shot"]


def test_resolve_images_empty_override_file_disables_photos(tmp_path):
    setting_dir = tmp_path / "soul" / "settings" / "some_setting"
    setting_dir.mkdir(parents=True)
    (setting_dir / "images.yaml").write_text("", encoding="utf-8")
    base = {"enabled": True, "model": "some/image-model"}

    result = resolve_images_config(tmp_path, "some_setting", base)

    assert result["modes"] == []
