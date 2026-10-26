#!/usr/bin/env bash
set -euo pipefail
r5a_auto_v5_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${R5A_PYTHON:-}" ]]; then export R6_PYTHON="$R5A_PYTHON"; fi
if [[ -n "${R5A_ISAAC_SIM:-}" ]]; then export R6_ISAAC_SIM="$R5A_ISAAC_SIM"; fi
if [[ "${1:-help}" == help ]]; then
    printf '%s\n' 'Independent autonomous backflip: check | train | evaluate | play | verify | collect' \
        'Start: bash r5a_auto_v5.sh train' \
        'Resume: bash r5a_auto_v5.sh train --checkpoint PATH --updates 3000' \
        'Review: copy the printed [USB REVIEW ZIP] and its .sha256 file.'
    exit 0
fi
exec bash "$r5a_auto_v5_root/r6.sh" python "$r5a_auto_v5_root/r5a_auto_v5/cli.py" "$@"
