"""
Pre-parser for control commands that must never reach the LLM.

Sits between the connector and the orchestrator: a recognized command is
routed straight to CommandExecutor and never touches chat_log.jsonl or the
prompt. This is a different concern from the `/debug`, `/state`, `/config`
etc. dispatcher in orchestrator.py — those are introspection commands that
happen to also be gated on a leading "/"; control commands like /reset
change runtime state and are handled here so that distinction stays visible
instead of both living in the same catch-all.
"""

from dataclasses import dataclass, field

KNOWN_COMMANDS = {"reset", "back", "list", "switch"}


@dataclass
class ParsedCommand:
    name: str
    args: list[str] = field(default_factory=list)


def parse_command(text: str) -> ParsedCommand | None:
    """
    Parse a raw incoming message into a control command.

    Returns None for anything that isn't a recognized control command —
    including other leading-"/" text, which the orchestrator's own debug
    dispatcher may still handle.
    """
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None

    parts = stripped[1:].split()
    if not parts:
        return None

    name = parts[0].lower()
    if name not in KNOWN_COMMANDS:
        return None

    return ParsedCommand(name=name, args=parts[1:])
