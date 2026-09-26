"""Tests for engine.command_executor.CommandExecutor"""

import pytest

from engine.command_executor import CommandExecutor
from engine.command_parser import ParsedCommand
from engine.memory_store import MemoryStore


@pytest.fixture
def executor(tmp_path, chat_logger):
    return CommandExecutor(
        chat_logger=chat_logger,
        state_path=tmp_path / "state.md",
        user_model_path=tmp_path / "user_model.md",
        continuity_path=tmp_path / "continuity.md",
        summaries_path=tmp_path / "data" / "summaries.jsonl",
        opening_line_path=tmp_path / "opening_line.md",
        debug_dir=tmp_path / "debug",
    )


@pytest.fixture
def memory_store(tmp_path):
    return MemoryStore(
        jsonl_path=tmp_path / "memory.jsonl",
        npy_path=tmp_path / "memory_vectors.npy",
        model_name="fake-embed",
    )


@pytest.fixture
def executor_with_memory(tmp_path, chat_logger, memory_store):
    return CommandExecutor(
        chat_logger=chat_logger,
        state_path=tmp_path / "state.md",
        user_model_path=tmp_path / "user_model.md",
        continuity_path=tmp_path / "continuity.md",
        summaries_path=tmp_path / "data" / "summaries.jsonl",
        opening_line_path=tmp_path / "opening_line.md",
        debug_dir=tmp_path / "debug",
        memory_store=memory_store,
    )


async def test_reset_clears_chat_log(executor, chat_logger):
    chat_logger.log_message(role="user", content="hello")
    await executor.execute(ParsedCommand(name="reset"))
    assert chat_logger.read_all() == []


async def test_reset_deletes_state_and_continuity_files(executor, tmp_path):
    (tmp_path / "state.md").write_text("---\nhits: 5\n---\n", encoding="utf-8")
    (tmp_path / "continuity.md").write_text("some memory", encoding="utf-8")
    (tmp_path / "user_model.md").write_text("---\nuser_id: 1\n---\n", encoding="utf-8")

    await executor.execute(ParsedCommand(name="reset"))

    assert not (tmp_path / "state.md").exists()
    assert not (tmp_path / "continuity.md").exists()
    assert not (tmp_path / "user_model.md").exists()


async def test_reset_without_opening_line_leaves_chat_log_empty(executor, chat_logger):
    response = await executor.execute(ParsedCommand(name="reset"))
    assert chat_logger.read_all() == []
    assert "no opening_line.md" in response


async def test_reset_with_opening_line_reseeds_chat_log(executor, chat_logger, tmp_path):
    (tmp_path / "opening_line.md").write_text("Hey, it's been a while.\n", encoding="utf-8")

    response = await executor.execute(ParsedCommand(name="reset"))

    entries = chat_logger.read_all()
    assert len(entries) == 1
    assert entries[0]["role"] == "assistant"
    assert entries[0]["content"] == "Hey, it's been a while."
    assert "starting fresh" in response


async def test_reset_truncates_summaries_and_clears_debug_logs(executor, tmp_path):
    summaries_path = tmp_path / "data" / "summaries.jsonl"
    summaries_path.parent.mkdir(parents=True)
    summaries_path.write_text('{"old": true}\n', encoding="utf-8")

    debug_dir = tmp_path / "debug"
    debug_dir.mkdir()
    (debug_dir / "prompt_log.jsonl").write_text("stale", encoding="utf-8")

    await executor.execute(ParsedCommand(name="reset"))

    assert summaries_path.read_text(encoding="utf-8") == ""
    assert not (debug_dir / "prompt_log.jsonl").exists()


async def test_reset_clears_memory_store(executor_with_memory, memory_store):
    memory_store.add(
        entries=[{"message_id": 1, "timestamp": "2026-01-01T00:00:00", "role": "user", "content": "hi"}],
        vectors=[[1.0, 0.0]],
    )
    assert len(memory_store) == 1

    await executor_with_memory.execute(ParsedCommand(name="reset"))

    assert len(memory_store) == 0


async def test_unknown_command_returns_message_without_touching_files(executor, chat_logger):
    chat_logger.log_message(role="user", content="hello")
    response = await executor.execute(ParsedCommand(name="banana"))
    assert "Unknown control command" in response
    assert len(chat_logger.read_all()) == 1


# --- /back ---

def _seed_pairs(chat_logger, n: int) -> None:
    """Log n complete (user, assistant) exchanges."""
    for i in range(n):
        chat_logger.log_message(role="user", content=f"user {i}")
        chat_logger.log_message(role="assistant", content=f"assistant {i}")


async def test_back_removes_exactly_n_pairs(executor, chat_logger):
    _seed_pairs(chat_logger, 3)

    response = await executor.execute(ParsedCommand(name="back", args=["1"]))

    remaining = chat_logger.read_all()
    assert [e["content"] for e in remaining] == ["user 0", "assistant 0", "user 1", "assistant 1"]
    assert "Rolled back 1 exchange" in response


async def test_back_multiple_pairs(executor, chat_logger):
    _seed_pairs(chat_logger, 3)

    await executor.execute(ParsedCommand(name="back", args=["2"]))

    remaining = chat_logger.read_all()
    assert [e["content"] for e in remaining] == ["user 0", "assistant 0"]


async def test_back_drops_dangling_tail_without_counting_it(executor, chat_logger):
    _seed_pairs(chat_logger, 2)
    chat_logger.log_message(role="user", content="unanswered")  # simulates a failed LLM call

    response = await executor.execute(ParsedCommand(name="back", args=["1"]))

    remaining = chat_logger.read_all()
    # the dangling "unanswered" is gone, and exactly one real pair was removed
    assert [e["content"] for e in remaining] == ["user 0", "assistant 0"]
    assert "Rolled back 1 exchange" in response


async def test_back_more_than_available_removes_everything(executor, chat_logger):
    _seed_pairs(chat_logger, 2)

    response = await executor.execute(ParsedCommand(name="back", args=["10"]))

    assert chat_logger.read_all() == []
    assert "Rolled back 2 exchange" in response


async def test_back_on_empty_log_returns_message_without_crashing(executor):
    response = await executor.execute(ParsedCommand(name="back", args=["1"]))
    assert "Nothing to roll back" in response


async def test_back_rejects_invalid_args(executor, chat_logger):
    _seed_pairs(chat_logger, 1)
    for bad_args in ([], ["0"], ["-1"], ["abc"], ["1", "2"]):
        response = await executor.execute(ParsedCommand(name="back", args=bad_args))
        assert "Usage:" in response
    # nothing was touched by any of the rejected calls
    assert len(chat_logger.read_all()) == 2


async def test_back_removes_ids_from_memory_store(executor_with_memory, chat_logger, memory_store):
    _seed_pairs(chat_logger, 2)
    for entry in chat_logger.read_all():
        memory_store.add(entries=[entry], vectors=[[1.0, 0.0]])
    assert len(memory_store) == 4

    await executor_with_memory.execute(ParsedCommand(name="back", args=["1"]))

    # only the two oldest messages (ids 1-2) should remain archived
    assert memory_store.all_ids() == [1, 2]


async def test_back_recomputes_total_chars_since_last_compression(executor, chat_logger, tmp_path):
    _seed_pairs(chat_logger, 2)

    await executor.execute(ParsedCommand(name="back", args=["1"]))

    from engine.state_manager import StateManager
    state_manager = StateManager(tmp_path / "state.md")
    assert state_manager.get_frontmatter()["total_chars_since_last_compression"] == chat_logger.get_total_chars()


@pytest.fixture
def project_root(tmp_path):
    """A fake project tree with two settings and soul/config.yaml, for /list and /switch."""
    for setting, self_slug, other_slug in (("alpha", "amy", "adam"), ("beta", "bill", "bea")):
        setting_dir = tmp_path / "soul" / "settings" / setting
        (setting_dir / "characters" / self_slug).mkdir(parents=True)
        (setting_dir / "characters" / other_slug).mkdir(parents=True)
        (setting_dir / "characters.yaml").write_text(
            f"self: {self_slug}\nother: {other_slug}\n", encoding="utf-8"
        )

    (tmp_path / "soul" / "config.yaml").write_text(
        "# top comment\n"
        "setting: alpha\n"
        "characters:\n"
        "  self: amy\n"
        "  other: adam\n"
        "llm:\n"
        "  provider: openrouter\n"
        "  model: shared-model\n"
        "  temperature: 0.7\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def switch_executor(project_root, chat_logger):
    return CommandExecutor(
        chat_logger=chat_logger,
        state_path=project_root / "state.md",
        user_model_path=project_root / "user_model.md",
        continuity_path=project_root / "continuity.md",
        summaries_path=project_root / "data" / "summaries.jsonl",
        opening_line_path=project_root / "opening_line.md",
        debug_dir=project_root / "debug",
        base_path=project_root,
        setting="alpha",
    )


async def test_list_marks_current_setting(switch_executor):
    response = await switch_executor.execute(ParsedCommand(name="list"))

    assert "alpha ← current" in response
    assert "beta" in response


async def test_switch_to_unknown_setting_is_rejected(switch_executor, project_root):
    response = await switch_executor.execute(ParsedCommand(name="switch", args=["nope"]))

    assert "Unknown setting" in response
    assert not switch_executor.pending_restart
    assert "setting: alpha" in (project_root / "soul" / "config.yaml").read_text()


async def test_switch_without_characters_file_is_rejected(switch_executor, project_root):
    (project_root / "soul" / "settings" / "beta" / "characters.yaml").unlink()

    response = await switch_executor.execute(ParsedCommand(name="switch", args=["beta"]))

    assert "characters.yaml" in response
    assert not switch_executor.pending_restart
    assert "setting: alpha" in (project_root / "soul" / "config.yaml").read_text()


async def test_switch_with_declared_character_missing_on_disk_is_rejected(
    switch_executor, project_root
):
    (project_root / "soul" / "settings" / "beta" / "characters.yaml").write_text(
        "self: ghost\nother: bea\n", encoding="utf-8"
    )

    response = await switch_executor.execute(ParsedCommand(name="switch", args=["beta"]))

    assert "ghost" in response
    assert not switch_executor.pending_restart
    assert "setting: alpha" in (project_root / "soul" / "config.yaml").read_text()


async def test_switch_writes_setting_and_its_own_pairing(switch_executor, project_root):
    response = await switch_executor.execute(ParsedCommand(name="switch", args=["beta"]))

    text = (project_root / "soul" / "config.yaml").read_text()
    assert "setting: beta" in text
    assert "self: bill" in text
    assert "other: bea" in text
    assert "# top comment" in text
    assert switch_executor.pending_restart
    assert "shared-model" in response


async def test_switch_reports_per_setting_llm_override(switch_executor, project_root):
    (project_root / "soul" / "settings" / "beta" / "llm.yaml").write_text(
        "model: beta-only-model\n", encoding="utf-8"
    )

    response = await switch_executor.execute(ParsedCommand(name="switch", args=["beta"]))

    assert "beta-only-model" in response


async def test_switch_without_args_lists_settings(switch_executor):
    response = await switch_executor.execute(ParsedCommand(name="switch"))

    assert "Usage: /switch" in response
    assert not switch_executor.pending_restart
