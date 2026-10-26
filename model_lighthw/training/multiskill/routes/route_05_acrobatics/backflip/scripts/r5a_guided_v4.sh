#!/usr/bin/env bash
set -euo pipefail
r5a_guided_v4_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../../../.." && pwd)"
if [[ -n "${R5A_PYTHON:-}" ]]; then export R6_PYTHON="$R5A_PYTHON"; fi
if [[ -n "${R5A_ISAAC_SIM:-}" ]]; then export R6_ISAAC_SIM="$R5A_ISAAC_SIM"; fi
if [[ "${1:-help}" == help ]]; then
    printf '%s\n' 'Guided v4 landing refinement from successful v3: check | train | evaluate | play | verify | collect' \
        'Start: bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_guided_v4.sh train (automatic baseline verification, then 3000 updates)' \
        'Resume: bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_guided_v4.sh train --checkpoint PATH --updates 3000' \
        'Review: copy the printed [USB REVIEW ZIP] and its .sha256 file.'
    exit 0
fi
exec bash "$r5a_guided_v4_root/r6.sh" python "$r5a_guided_v4_root/model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/r5a_guided_v4/station.py" "$@"
