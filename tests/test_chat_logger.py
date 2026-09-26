"""Tests for engine.chat_logger.ChatLogger"""

from engine.chat_logger import ChatLogger


def test_log_message_auto_computes_char_count(chat_logger):
    chat_logger.log_message(role="user", content="hello")
    entries = chat_logger.read_all()
    assert len(entries) == 1
    assert entries[0]["content"] == "hello"
    assert entries[0]["char_count"] == len("hello")
    assert entries[0]["role"] == "user"


def test_log_message_includes_user_id_when_given(chat_logger):
    chat_logger.log_message(role="user", content="hi", user_id="42")
    assert chat_logger.read_all()[0]["user_id"] == "42"


def test_log_message_omits_user_id_when_not_given(chat_logger):
    chat_logger.log_message(role="assistant", content="hi")
    assert "user_id" not in chat_logger.read_all()[0]


def test_read_all_empty_log(chat_logger):
    assert chat_logger.read_all() == []


def test_read_all_preserves_order(chat_logger):
    chat_logger.log_message(role="user", content="one")
    chat_logger.log_message(role="assistant", content="two")
    chat_logger.log_message(role="user", content="three")
    contents = [e["content"] for e in chat_logger.read_all()]
    assert contents == ["one", "two", "three"]


def test_read_last_n(chat_logger):
    for i in range(5):
        chat_logger.log_message(role="user", content=str(i))
    last = chat_logger.read_last_n(2)
    assert [e["content"] for e in last] == ["3", "4"]


def test_read_last_n_more_than_available(chat_logger):
    chat_logger.log_message(role="user", content="only one")
    assert len(chat_logger.read_last_n(10)) == 1


def test_read_last_n_chars_respects_budget(chat_logger):
    chat_logger.log_message(role="user", content="a" * 10)
    chat_logger.log_message(role="assistant", content="b" * 10)
    chat_logger.log_message(role="user", content="c" * 10)

    result = chat_logger.read_last_n_chars(15)

    assert sum(e["char_count"] for e in result) <= 15
    # only the most recent message fits under budget
    assert [e["content"] for e in result] == ["c" * 10]


def test_read_last_n_chars_keeps_chronological_order(chat_logger):
    chat_logger.log_message(role="user", content="a" * 5)
    chat_logger.log_message(role="assistant", content="b" * 5)
    chat_logger.log_message(role="user", content="c" * 5)

    result = chat_logger.read_last_n_chars(100)

    assert [e["content"] for e in result] == ["a" * 5, "b" * 5, "c" * 5]


def test_get_total_chars(chat_logger):
    chat_logger.log_message(role="user", content="abc")
    chat_logger.log_message(role="assistant", content="de")
    assert chat_logger.get_total_chars() == 5


def test_clear(chat_logger):
    chat_logger.log_message(role="user", content="something")
    chat_logger.clear()
    assert chat_logger.read_all() == []


# --- message_id (with state_manager wired, as the `chat_logger` fixture is) ---

def test_log_message_auto_assigns_monotonic_message_id(chat_logger):
    chat_logger.log_message(role="user", content="one")
    chat_logger.log_message(role="assistant", content="two")
    ids = [e["message_id"] for e in chat_logger.read_all()]
    assert ids == [1, 2]


def test_log_message_explicit_message_id_overrides_auto_assign(chat_logger):
    chat_logger.log_message(role="user", content="one", message_id=99)
    assert chat_logger.read_all()[0]["message_id"] == 99


def test_log_message_explicit_message_id_does_not_consume_counter(chat_logger):
    chat_logger.log_message(role="user", content="one", message_id=99)
    chat_logger.log_message(role="user", content="two")
    assert chat_logger.read_all()[1]["message_id"] == 1


def test_log_message_explicit_none_message_id_is_omitted(chat_logger):
    """_delete_old_messages rewrites pre-message_id entries by passing
    message_id=None explicitly — that must be preserved as absent, not
    treated as "not given" and auto-assigned a fresh id."""
    chat_logger.log_message(role="user", content="legacy", message_id=None)
    assert "message_id" not in chat_logger.read_all()[0]


def test_message_id_survives_logger_restart(tmp_path, state_manager):
    log_file = tmp_path / "chat_log.jsonl"
    ChatLogger(log_file, state_manager=state_manager).log_message(role="user", content="one")

    # New ChatLogger instance, same backing files - simulates a process restart.
    restarted = ChatLogger(log_file, state_manager=state_manager)
    restarted.log_message(role="user", content="two")

    ids = [e["message_id"] for e in restarted.read_all()]
    assert ids == [1, 2]


def test_rewrite_replaces_contents_preserving_ids(chat_logger):
    chat_logger.log_message(role="user", content="one")
    chat_logger.log_message(role="assistant", content="two")
    chat_logger.log_message(role="user", content="three")
    entries = chat_logger.read_all()

    chat_logger.rewrite([entries[0], entries[2]])

    remaining = chat_logger.read_all()
    assert [e["content"] for e in remaining] == ["one", "three"]
    assert [e["message_id"] for e in remaining] == [1, 3]


def test_rewrite_preserves_original_timestamps(chat_logger):
    """Compression and /back rewrite the log; surviving messages must keep
    the time they were actually sent, not the time of the rewrite."""
    chat_logger.log_message(role="user", content="one")
    chat_logger.log_message(role="assistant", content="two")
    original = chat_logger.read_all()

    chat_logger.rewrite(original)

    assert [e["timestamp"] for e in chat_logger.read_all()] == [
        e["timestamp"] for e in original
    ]


def test_log_message_defaults_timestamp_to_now(chat_logger):
    chat_logger.log_message(role="user", content="one", timestamp="2020-01-01T00:00:00")
    chat_logger.log_message(role="user", content="two")
    entries = chat_logger.read_all()
    assert entries[0]["timestamp"] == "2020-01-01T00:00:00"
    assert entries[1]["timestamp"] != "2020-01-01T00:00:00"


def test_rewrite_empty_list_clears_log(chat_logger):
    chat_logger.log_message(role="user", content="one")
    chat_logger.rewrite([])
    assert chat_logger.read_all() == []


def test_rewrite_leaves_original_contents_if_interrupted_before_replace(chat_logger, monkeypatch):
    """rewrite() writes to a temp file and os.replace()s it in — if the
    process dies before the replace, the original log must survive intact
    rather than being left cleared (the old clear()-then-append-loop could
    lose the whole log if killed mid-loop)."""
    chat_logger.log_message(role="user", content="one")
    chat_logger.log_message(role="assistant", content="two")

    def boom(*args, **kwargs):
        raise OSError("simulated crash before rename")

    monkeypatch.setattr("engine.chat_logger.os.replace", boom)

    try:
        chat_logger.rewrite([{"role": "user", "content": "replacement"}])
    except OSError:
        pass

    entries = chat_logger.read_all()
    assert [e["content"] for e in entries] == ["one", "two"]


def test_message_id_omitted_without_state_manager(tmp_path):
    logger = ChatLogger(tmp_path / "chat_log.jsonl")
    logger.log_message(role="user", content="hello")
    assert "message_id" not in logger.read_all()[0]


def test_log_message_returns_the_assigned_message_id(chat_logger):
    assert chat_logger.log_message(role="user", content="one") == 1
    assert chat_logger.log_message(role="user", content="two", message_id=99) == 99
    assert chat_logger.log_message(role="user", content="three", message_id=None) is None
