"""Tests for engine.command_parser.parse_command"""

from engine.command_parser import parse_command


def test_recognizes_known_command():
    result = parse_command("/reset")
    assert result is not None
    assert result.name == "reset"
    assert result.args == []


def test_is_case_insensitive():
    result = parse_command("/RESET")
    assert result.name == "reset"


def test_strips_surrounding_whitespace():
    result = parse_command("  /reset  ")
    assert result.name == "reset"


def test_captures_args():
    result = parse_command("/reset now")
    assert result.name == "reset"
    assert result.args == ["now"]


def test_plain_text_is_not_a_command():
    assert parse_command("hey how are you") is None


def test_unrecognized_slash_command_returns_none():
    assert parse_command("/debug") is None


def test_bare_slash_returns_none():
    assert parse_command("/") is None


def test_recognizes_back_with_count_arg():
    result = parse_command("/back 2")
    assert result.name == "back"
    assert result.args == ["2"]
