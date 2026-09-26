"""Tests for engine.compressor.Compressor"""

import json

import pytest

from engine.compressor import Compressor


@pytest.fixture
def prompt_file(tmp_path):
    path = tmp_path / "compression_prompt.md"
    path.write_text(
        "Compress. Summary limit: {{summary_max_words}}. Continuity limit: {{continuity_max_words}}.",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def compressor(fake_llm, state_manager, chat_logger, tmp_path, prompt_file):
    return Compressor(
        llm=fake_llm,
        chat_logger=chat_logger,
        state_manager=state_manager,
        config={
            "trigger_chars": 100,
            "chunk_size_messages": 3,
            "summary_max_words": 200,
            "continuity_max_words": 570,
        },
        prompt_file=prompt_file,
        summaries_path=tmp_path / "summaries.jsonl",
        continuity_path=tmp_path / "continuity.md",
        role_names={"user": "Misha", "assistant": "Zoya"},
        template_vars={"self_name": "Zoya", "other_name": "Misha"},
    )


def _fill_chat_log(chat_logger, n):
    for i in range(n):
        chat_logger.log_message(role="user", content=f"message {i}")


# --- should_compress() ---

def test_should_compress_below_threshold(compressor, state_manager):
    state_manager.update_frontmatter({"total_chars_since_last_compression": 50})
    assert compressor.should_compress() is False


def test_should_compress_at_or_above_threshold(compressor, state_manager):
    state_manager.update_frontmatter({"total_chars_since_last_compression": 150})
    assert compressor.should_compress() is True


# --- compress() ---

async def test_compress_success_writes_summary_under_summary_key(
    compressor, fake_llm, chat_logger, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _fill_chat_log(chat_logger, 5)
    fake_llm.content = "===SUMMARY===\nshort recap\n===CONTINUITY===\nupdated continuity"

    result = await compressor.compress()

    assert result["success"] is True
    lines = (tmp_path / "summaries.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    # regression guard: prompt_builder reads this back via the "summary" key
    assert entry["summary"] == "short recap"
    assert "content" not in entry


async def test_compress_deletes_compressed_messages_keeps_remainder(
    compressor, fake_llm, chat_logger, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _fill_chat_log(chat_logger, 5)  # chunk_size_messages = 3
    fake_llm.content = "===SUMMARY===\nrecap\n===CONTINUITY===\ncontinuity"

    result = await compressor.compress()

    remaining = chat_logger.read_all()
    assert result["chat_log_remaining"] == 2
    assert [e["content"] for e in remaining] == ["message 3", "message 4"]


async def test_compress_preserves_message_id_of_retained_messages(
    compressor, fake_llm, chat_logger, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _fill_chat_log(chat_logger, 5)  # chunk_size_messages = 3, ids auto-assigned 1..5
    fake_llm.content = "===SUMMARY===\nrecap\n===CONTINUITY===\ncontinuity"

    await compressor.compress()

    remaining = chat_logger.read_all()
    assert [e["message_id"] for e in remaining] == [4, 5]


async def test_compress_more_chunk_than_messages_clears_log(
    compressor, fake_llm, chat_logger, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    compressor.chunk_size_messages = 100
    _fill_chat_log(chat_logger, 2)
    fake_llm.content = "===SUMMARY===\nrecap\n===CONTINUITY===\ncontinuity"

    result = await compressor.compress()

    assert result["success"] is True
    assert result["chat_log_remaining"] == 0
    assert chat_logger.read_all() == []


async def test_compress_no_messages_returns_error(compressor):
    result = await compressor.compress()
    assert result == {"success": False, "error": "no_messages"}


async def test_compress_parse_failure(compressor, fake_llm, chat_logger, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _fill_chat_log(chat_logger, 3)
    fake_llm.content = "no markers here"

    result = await compressor.compress()

    assert result["success"] is False
    assert result["error"] == "parse_failed"
    assert result["raw_response"] == "no markers here"


async def test_compress_trims_summary_exceeding_word_limit(
    compressor, fake_llm, chat_logger, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    compressor.summary_max_words = 3
    _fill_chat_log(compressor.chat_logger, 3)
    fake_llm.content = "===SUMMARY===\none two three four five\n===CONTINUITY===\nfine"

    result = await compressor.compress()

    assert result["summary"] == "one two three"


# --- continuity read/write ---

def test_write_then_read_continuity_round_trips(compressor):
    compressor._write_continuity("some continuity text")
    assert compressor._read_continuity() == "some continuity text"


def test_continuity_initialized_empty_by_default(compressor):
    """Empty, not a placeholder: the file's content reaches the prompt as-is."""
    assert compressor._read_continuity() == ""
