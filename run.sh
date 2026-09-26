#!/usr/bin/env bash
# Start the SoulEngine Telegram bot locally (no Docker).
# Ctrl+C to stop.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f .env ]; then
    echo "Error: .env not found. Copy .env.example to .env and fill in your credentials first."
    exit 1
fi

set -a
source .env
set +a

uv run python -m engine.main
