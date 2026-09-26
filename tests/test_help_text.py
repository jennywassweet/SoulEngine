"""Guards HELP_TEXT against drifting out of sync with the commands that exist.

The list shipped for months missing every control command — /reset, /back,
/list, /switch — because they live in a different dispatcher than the debug
commands the list was written from.
"""

from engine.command_parser import KNOWN_COMMANDS
from engine.orchestrator import HELP_TEXT, PHOTO_COMMANDS


def test_help_text_lists_every_control_command():
    missing = [name for name in KNOWN_COMMANDS if f"/{name}" not in HELP_TEXT]
    assert not missing, f"HELP_TEXT is missing control commands: {missing}"


def test_help_text_lists_every_photo_command():
    """Photo commands (/shot, /scene, /selfie, /again, /photos) live in the
    debug dispatcher, not command_parser.py's KNOWN_COMMANDS — the test
    above can't see them, so this guards them separately."""
    missing = [name for name in PHOTO_COMMANDS if f"/{name}" not in HELP_TEXT]
    assert not missing, f"HELP_TEXT is missing photo commands: {missing}"
