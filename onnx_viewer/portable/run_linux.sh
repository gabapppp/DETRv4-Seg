#!/usr/bin/env bash
set -e
cd -- "$(dirname -- "$0")"
if command -v python3.12 >/dev/null 2>&1; then
  exec python3.12 run_viewer.py "$@"
fi
exec python3 run_viewer.py "$@"
