#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROUTE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../../../" && pwd)"

cd "${ROUTE_DIR}"
exec "${REPO_ROOT}/Isaaclab/isaaclab.sh" -p "${REPO_ROOT}/scripts/reinforcement_learning/rsl_rl/train.py" "$@" \
    --task Rough-Deeprobotics-LightHW-Blind-v0
