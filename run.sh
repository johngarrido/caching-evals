#!/usr/bin/env bash
# Convenience runner: syncs deps and runs the cache eval on Vertex AI.
#
# Usage:
#   gcloud auth application-default login     # once (default auth: ADC)
#   export VERTEX_API_KEY=...                 # optional: use a Vertex AI API key instead
#   ./run.sh --project my-project             # both arms, 5 trials each
#   ./run.sh --project my-project --reps 10
#   ./run.sh --dry-run                        # no API calls, synthetic run for the workbench
#   ./run.sh --help
set -euo pipefail
cd "$(dirname "$0")"

uv sync --quiet

uv run python cache_eval.py "$@"
