#!/usr/bin/env bash
set -euo pipefail
r5a_auto_v3_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../../../.." && pwd)"
if [[ -n "${R5A_PYTHON:-}" ]]; then export R6_PYTHON="$R5A_PYTHON"; fi
if [[ -n "${R5A_ISAAC_SIM:-}" ]]; then export R6_ISAAC_SIM="$R5A_ISAAC_SIM"; fi
if [[ "${1:-help}" == help ]]; then
    printf '%s\n' 'Independent autonomous backflip: check | train | evaluate | play | verify | collect' \
        'Start: bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_auto_v3.sh train' \
        'Resume: bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_auto_v3.sh train --checkpoint PATH --updates 2000' \
        'Review: copy the printed [USB REVIEW ZIP] and its .sha256 file.'
    exit 0
fi
exec bash "$r5a_auto_v3_root/r6.sh" python "$r5a_auto_v3_root/model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/r5a_auto_v3/cli.py" "$@"
