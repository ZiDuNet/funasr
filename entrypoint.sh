#!/bin/bash
set -e

FUNASR_HOST="${FUNASR_HOST:-0.0.0.0}"
FUNASR_PORT="${FUNASR_PORT:-8000}"
FUNASR_DEVICE="${FUNASR_DEVICE:-cpu}"
FUNASR_MODEL="${FUNASR_MODEL:-sensevoice}"
FUNASR_WORKERS="${FUNASR_WORKERS:-1}"

export FUNASR_DATA_DIR="${FUNASR_DATA_DIR:-/data}"

exec python /app/server.py \
    --host "$FUNASR_HOST" \
    --port "$FUNASR_PORT" \
    --device "$FUNASR_DEVICE" \
    --model "$FUNASR_MODEL" \
    --workers "$FUNASR_WORKERS"
