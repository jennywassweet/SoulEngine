"""Tests for engine.debug_logger.save_prompt_to_file"""

from engine.debug_logger import save_prompt_to_file


def test_creates_parent_directories(tmp_path):
    target = tmp_path / "nested" / "dir" / "prompt.txt"
    save_prompt_to_file([{"role": "user", "content": "hi"}], target)
    assert target.exists()


def test_formats_with_default_role_names(tmp_path):
    target = tmp_path / "prompt.txt"
    save_prompt_to_file(
        [{"role": "system", "content": "sys msg"}, {"role": "user", "content": "user msg"}],
        target,
    )
    text = target.read_text(encoding="utf-8")
    assert "SYSTEM" in text
    assert "sys msg" in text
    assert "USER" in text
    assert "user msg" in text


def test_formats_with_custom_role_names(tmp_path):
    target = tmp_path / "prompt.txt"
    save_prompt_to_file(
        [{"role": "assistant", "content": "hello"}],
        target,
        role_names={"assistant": "Zoe"},
    )
    text = target.read_text(encoding="utf-8")
    assert "Zoe" in text
    assert "hello" in text


def test_unknown_role_falls_back_to_uppercase(tmp_path):
    target = tmp_path / "prompt.txt"
    save_prompt_to_file([{"role": "tool", "content": "result"}], target)
    text = target.read_text(encoding="utf-8")
    assert "TOOL" in text
