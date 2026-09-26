"""Tests for engine.state_manager.StateManager"""

from engine.state_manager import StateManager


def test_read_missing_file_returns_empty(tmp_path):
    sm = StateManager(tmp_path / "missing.md")
    frontmatter, content = sm.read()
    assert frontmatter == {}
    assert content == ""


def test_write_then_read_round_trip(state_manager):
    state_manager.write({"count": 3, "name": "Zoe"}, "Hello world")
    frontmatter, content = state_manager.read()
    assert frontmatter == {"count": 3, "name": "Zoe"}
    assert content == "Hello world"


def test_get_frontmatter_and_get_content(state_manager):
    state_manager.write({"a": 1}, "body text")
    assert state_manager.get_frontmatter() == {"a": 1}
    assert state_manager.get_content() == "body text"


def test_update_frontmatter_merges_without_clobbering(state_manager):
    state_manager.write({"a": 1, "b": 2}, "content")
    state_manager.update_frontmatter({"b": 20, "c": 30})
    frontmatter, content = state_manager.read()
    assert frontmatter["a"] == 1
    assert frontmatter["b"] == 20
    assert frontmatter["c"] == 30
    assert content == "content"


def test_increment_counter_from_zero(state_manager):
    new_value = state_manager.increment_counter("hits")
    assert new_value == 1
    assert state_manager.get_frontmatter()["hits"] == 1


def test_increment_counter_from_existing_value(state_manager):
    state_manager.write({"hits": 5}, "")
    new_value = state_manager.increment_counter("hits", amount=3)
    assert new_value == 8
    assert state_manager.get_frontmatter()["hits"] == 8


def test_increment_counter_sets_last_updated(state_manager):
    state_manager.increment_counter("hits")
    assert "last_updated" in state_manager.get_frontmatter()


def test_reset_counter(state_manager):
    state_manager.write({"hits": 42}, "")
    state_manager.reset_counter("hits")
    assert state_manager.get_frontmatter()["hits"] == 0


def test_update_content_preserves_frontmatter(state_manager):
    state_manager.write({"a": 1}, "old content")
    state_manager.update_content("new content")
    frontmatter, content = state_manager.read()
    assert frontmatter["a"] == 1
    assert content == "new content"
    assert "last_updated" in frontmatter
