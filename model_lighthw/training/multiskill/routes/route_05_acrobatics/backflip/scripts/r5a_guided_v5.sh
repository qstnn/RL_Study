#!/usr/bin/env bash
set -euo pipefail
r5a_guided_v5_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../../../.." && pwd)"
if [[ -n "${R5A_PYTHON:-}" ]]; then export R6_PYTHON="$R5A_PYTHON"; fi
if [[ -n "${R5A_ISAAC_SIM:-}" ]]; then export R6_ISAAC_SIM="$R5A_ISAAC_SIM"; fi
if [[ "${1:-help}" == help ]]; then
    printf '%s\n' 'Guided v5 landing refinement from successful v3: check | audit | prepare | train | evaluate | play | verify | collect' \
        'Audit only: bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_guided_v5.sh audit --v4-session PATH (CPU, no simulation or PPO)' \
        'Start: bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_guided_v5.sh train --v4-session PATH (contact/reward audit, baseline verification, then 3000 updates)' \
        'Resume: bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_guided_v5.sh train --checkpoint PATH --v4-session ORIGINAL_V4_SESSION --updates 3000' \
        'Review: copy the printed [USB REVIEW ZIP] and its .sha256 file.'
    exit 0
fi
exec bash "$r5a_guided_v5_root/r6.sh" python "$r5a_guided_v5_root/model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/r5a_guided_v5/station.py" "$@"
