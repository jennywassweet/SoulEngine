"""Tests for engine.response_parser.parse_markers"""

from engine.response_parser import parse_markers


def test_parses_two_markers():
    text = "===STATE===\nstate text\n===USER_MODEL===\nuser model text"
    result = parse_markers(text, ["===STATE===", "===USER_MODEL==="])
    assert result == {"STATE": "state text", "USER_MODEL": "user model text"}


def test_missing_marker_returns_none():
    text = "===STATE===\nonly state here"
    result = parse_markers(text, ["===STATE===", "===USER_MODEL==="])
    assert result["STATE"] == "only state here"
    assert result["USER_MODEL"] is None


def test_last_marker_content_runs_to_end_of_text():
    text = "===A===\nfirst\n===B===\nsecond part\nwith more lines"
    result = parse_markers(text, ["===A===", "===B==="])
    assert result["B"] == "second part\nwith more lines"


def test_content_is_stripped():
    text = "===A===   \n  padded content  \n\n===B===\nrest"
    result = parse_markers(text, ["===A===", "===B==="])
    assert result["A"] == "padded content"


def test_no_markers_found_all_none():
    result = parse_markers("random text without markers", ["===A===", "===B==="])
    assert result == {"A": None, "B": None}


def test_uses_last_occurrence_when_marker_mentioned_earlier_in_prose():
    # A model narrating its own instructions ("the ===B=== marker means...")
    # before actually using them must not have that mention mistaken for
    # the real, structural marker.
    text = (
        "Let me think. The ===A=== marker holds private notes, and "
        "===B=== holds the reply. Final answer:\n"
        "===A===\nreal thought\n===B===\nreal reply"
    )
    result = parse_markers(text, ["===A===", "===B==="])
    assert result == {"A": "real thought", "B": "real reply"}
