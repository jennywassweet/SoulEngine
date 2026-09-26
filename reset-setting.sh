#!/usr/bin/env bash
# Reset a setting's runtime state (chat log, summaries, state, user_model,
# continuity, retrieval memory, debug logs) back to a clean start. If the
# setting has an opening_line.md, chat_log.jsonl is reseeded with it as the
# first message from `self` — everything else starts empty.
#
# This delegates to engine/command_executor.py rather than reimplementing
# the wipe in bash: the same code runs whether you reset from the shell or
# by sending /reset in chat. (It used to be a second bash copy, which had
# already drifted — it reseeded the opening line without a message_id and
# left memory_vectors.npy behind while deleting the id counter in state.md,
# so fresh messages would reuse ids still held by archived vectors.)
#
# Usage:
#   ./reset-setting.sh <setting>
set -euo pipefail
cd "$(dirname "$0")"

SETTINGS_DIR="soul/settings"

if [ $# -lt 1 ]; then
    echo "Usage: ./reset-setting.sh <setting>"
    echo "Available settings:"
    for d in "$SETTINGS_DIR"/*/; do
        [ -d "$d" ] && echo "  - $(basename "$d")"
    done
    exit 1
fi

SETTING="$1"

if [ ! -d "$SETTINGS_DIR/$SETTING" ]; then
    echo "Error: $SETTINGS_DIR/$SETTING does not exist."
    exit 1
fi

uv run python - "$SETTING" <<'EOF'
import asyncio
import sys
from pathlib import Path

import yaml

from engine.chat_logger import ChatLogger
from engine.config_utils import runtime_dir, setting_dir
from engine.command_executor import CommandExecutor
from engine.command_parser import ParsedCommand
from engine.memory_store import MemoryStore
from engine.state_manager import StateManager

setting = sys.argv[1]
base = Path.cwd()
soul_path = setting_dir(base, setting)
runtime_path = runtime_dir(base, setting)
runtime_path.mkdir(parents=True, exist_ok=True)

memory_config = yaml.safe_load((base / "soul" / "config.yaml").read_text(encoding="utf-8")).get("memory", {})

state_path = runtime_path / "state.md"
state_manager = StateManager(state_path)
chat_logger = ChatLogger(runtime_path / "chat_log.jsonl", state_manager=state_manager)

memory_store = MemoryStore(
    jsonl_path=runtime_path / "memory.jsonl",
    npy_path=runtime_path / "memory_vectors.npy",
    model_name=memory_config.get("model", "openai/text-embedding-3-small"),
)

executor = CommandExecutor(
    chat_logger=chat_logger,
    state_path=state_path,
    user_model_path=runtime_path / "user_model.md",
    continuity_path=runtime_path / "continuity.md",
    summaries_path=runtime_path / "summaries.jsonl",
    opening_line_path=soul_path / "opening_line.md",
    debug_dir=base / "debug",
    memory_store=memory_store,
)

print(asyncio.run(executor.execute(ParsedCommand(name="reset"))))
EOF
