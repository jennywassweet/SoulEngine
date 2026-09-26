"""Tests for engine.prompt_builder.PromptBuilder"""

import pytest

from engine.prompt_builder import PromptBuilder


@pytest.fixture
def project(tmp_path):
    """Set up a minimal soul/settings/<setting>/ + config/main_prompt.md tree under tmp_path"""
    setting_path = tmp_path / "soul" / "settings" / "test_setting"
    (setting_path / "characters" / "zoya").mkdir(parents=True)
    (setting_path / "characters" / "misha").mkdir(parents=True)
    (setting_path / "characters" / "zoya" / "bible.md").write_text(
        "Zoya. Designer.", encoding="utf-8"
    )
    (setting_path / "characters" / "misha" / "bible.md").write_text(
        "Misha. Programmer.", encoding="utf-8"
    )
    (setting_path / "relationship.md").write_text(
        "They met online.", encoding="utf-8"
    )
    (setting_path / "runtime").mkdir(parents=True)
    (setting_path / "runtime" / "continuity.md").write_text(
        "Currently on a train.", encoding="utf-8"
    )
    (tmp_path / "config").mkdir()
    main_prompt = tmp_path / "config" / "main_prompt.md"
    main_prompt.write_text(
        "Continue the exchange between {{self_name}} and {{other_name}}.",
        encoding="utf-8",
    )
    return tmp_path, main_prompt


@pytest.fixture
def builder(project):
    base_path, main_prompt = project
    return PromptBuilder(
        base_path=base_path,
        setting="test_setting",
        characters_config={"self": "zoya", "other": "misha"},
        main_prompt_file=main_prompt,
        budget={"total_chars": 24000, "reserved_for_response": 4000},
    )


def test_substitutes_character_names_in_system_frame(builder, chat_logger):
    messages, _ = builder.build_prompt(current_message="hi", chat_logger=chat_logger)
    system_content = messages[0]["content"]
    assert "Zoya" in system_content
    assert "Misha" in system_content
    assert "{{" not in system_content


def test_returns_exactly_two_messages(builder, chat_logger):
    messages, _ = builder.build_prompt(current_message="hi", chat_logger=chat_logger)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_user_message_includes_all_static_blocks(builder, chat_logger):
    messages, _ = builder.build_prompt(current_message="hi", chat_logger=chat_logger)
    user_content = messages[1]["content"]
    assert "Designer" in user_content
    assert "Programmer" in user_content
    assert "met online" in user_content
    assert "on a train" in user_content


def test_manuscript_includes_history_in_order_with_names(builder, chat_logger):
    chat_logger.log_message(role="user", content="hey")
    chat_logger.log_message(role="assistant", content="hi there")

    messages, _ = builder.build_prompt(current_message="how are you", chat_logger=chat_logger)
    user_content = messages[1]["content"]

    hey_pos = user_content.find("Misha: hey")
    hi_pos = user_content.find("Zoya: hi there")
    current_pos = user_content.find("Misha: how are you")

    assert hey_pos != -1 and hi_pos != -1 and current_pos != -1
    assert hey_pos < hi_pos < current_pos


def test_manuscript_respects_tight_budget(builder, chat_logger):
    builder.budget = {"total_chars": 200, "reserved_for_response": 0}
    for i in range(50):
        chat_logger.log_message(role="user", content=f"message number {i}" * 5)

    messages, debug_info = builder.build_prompt(current_message="hi", chat_logger=chat_logger)

    assert debug_info["total_chars"] <= sum(len(m["content"]) for m in messages)
    # budget is tight enough that not all 50 historical messages fit
    manuscript_lines = messages[1]["content"].count("Misha: message number")
    assert manuscript_lines < 50


def test_debug_info_reports_block_sizes(builder, chat_logger):
    _, debug_info = builder.build_prompt(current_message="hi", chat_logger=chat_logger)
    assert set(debug_info["blocks"].keys()) == {
        "system",
        "self_bible",
        "other_bible",
        "relationship",
        "continuity",
        "recalled",
        "manuscript",
    }
    assert all(v >= 0 for v in debug_info["blocks"].values())


# --- recalled block ---

def test_recalled_block_absent_when_empty(builder, chat_logger):
    messages, debug_info = builder.build_prompt(current_message="hi", chat_logger=chat_logger, recalled="")
    assert debug_info["blocks"]["recalled"] == 0
    # nothing in the user content should be attributable to an empty block
    # (no stray blank-block separator artifacts)
    assert "\n\n\n\n" not in messages[1]["content"]


def test_recalled_block_included_when_present(builder, chat_logger):
    messages, debug_info = builder.build_prompt(
        current_message="hi", chat_logger=chat_logger, recalled="Misha: remember the trip?"
    )
    assert "remember the trip" in messages[1]["content"]
    assert debug_info["blocks"]["recalled"] == len("Misha: remember the trip?")


def test_recalled_block_counted_in_budget(builder, chat_logger):
    builder.budget = {"total_chars": 200, "reserved_for_response": 0}
    for i in range(50):
        chat_logger.log_message(role="user", content=f"message number {i}" * 5)

    _, without_recalled = builder.build_prompt(current_message="hi", chat_logger=chat_logger)
    _, with_recalled = builder.build_prompt(
        current_message="hi", chat_logger=chat_logger, recalled="x" * 100
    )

    assert with_recalled["blocks"]["manuscript"] <= without_recalled["blocks"]["manuscript"]


def test_recalled_block_sits_between_continuity_and_manuscript(builder, chat_logger):
    chat_logger.log_message(role="user", content="hey")
    messages, _ = builder.build_prompt(
        current_message="hi", chat_logger=chat_logger, recalled="Misha: an old excerpt"
    )
    user_content = messages[1]["content"]

    continuity_pos = user_content.find("on a train")
    recalled_pos = user_content.find("an old excerpt")
    manuscript_pos = user_content.find("Misha: hey")

    assert continuity_pos < recalled_pos < manuscript_pos
