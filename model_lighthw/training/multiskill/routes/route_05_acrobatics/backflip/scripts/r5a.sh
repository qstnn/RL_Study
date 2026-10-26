#!/usr/bin/env bash
set -euo pipefail
r5a_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../../../.." && pwd)"
r5a_command="${1:-help}"
if [[ "$r5a_command" == help || "$r5a_command" == --help ]]; then
    printf '%s\n' 'R5A J1 v3: highjump | highjump-check | highjump-probe | highjump-evaluate | highjump-play' \
        'R5A J1: prepare | doctor | check | smoke | train-smoke | refine | migrate | train | play | evaluate | review | collect | verify' \
        'J1 v2: bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a.sh refine (migrate, frozen baseline, up to 300 updates, evaluate, collect)' \
        'Playback: bash model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a.sh play --checkpoint PATH --play-episodes 10' \
        'Set R5A_PYTHON and R5A_ISAAC_SIM only when installation paths are not detected.'
    exit 0
fi
shift
if [[ -n "${R5A_PYTHON:-}" ]]; then export R6_PYTHON="$R5A_PYTHON"; fi
if [[ -n "${R5A_ISAAC_SIM:-}" ]]; then export R6_ISAAC_SIM="$R5A_ISAAC_SIM"; fi
case "$r5a_command" in
    prepare|doctor|python) exec bash "$r5a_root/r6.sh" "$r5a_command" "$@" ;;
esac
r5a_entry="$r5a_root/model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/launch.py"
r5a_highjump="$r5a_root/model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/highjump/scripts/launch.py"
case "$r5a_command" in
    highjump) exec bash "$r5a_root/r6.sh" python "$r5a_root/model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/offline/r5a_highjump.py" "$@" ;;
    highjump-check) exec bash "$r5a_root/r6.sh" python "$r5a_highjump" check "$@" ;;
    highjump-probe) exec bash "$r5a_root/r6.sh" python "$r5a_highjump" smoke --headless --num_envs 4 "$@" ;;
    highjump-evaluate) exec bash "$r5a_root/r6.sh" python "$r5a_highjump" evaluate --headless "$@" ;;
    highjump-play)
        if [[ $# -eq 0 ]]; then
            r5a_marker="$r5a_root/offline_sessions/route_05a/LATEST_HIGHJUMP_CHECKPOINT.txt"
            if [[ ! -f "$r5a_marker" ]]; then printf '%s\n' 'No selected highjump model; specify --checkpoint PATH.' >&2; exit 2; fi
            IFS= read -r r5a_selected < "$r5a_marker"
            set -- --checkpoint "$r5a_root/$r5a_selected"
        fi
        exec bash "$r5a_root/r6.sh" python "$r5a_highjump" play "$@" ;;
    refine) exec bash "$r5a_root/r6.sh" python "$r5a_root/model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/offline/r5a_refine.py" "$@" ;;
    migrate) exec bash "$r5a_root/r6.sh" python "$(dirname -- "$r5a_entry")/migrate.py" "$@" ;;
    train|play|review|collect|routes|verify)
        exec bash "$r5a_root/r6.sh" python "$r5a_root/offline/station.py" "$r5a_command" --route 05a "$@" ;;
    check) exec bash "$r5a_root/r6.sh" python "$r5a_entry" check "$@" ;;
    smoke) exec bash "$r5a_root/r6.sh" python "$r5a_entry" smoke --headless --num_envs 4 "$@" ;;
    train-smoke) exec bash "$r5a_root/r6.sh" python "$r5a_entry" train --headless --num_envs 32 --max_iterations 2 "$@" ;;
    evaluate) exec bash "$r5a_root/r6.sh" python "$r5a_entry" evaluate --headless "$@" ;;
    *) printf '%s\n' "[R5A ERROR] Unknown command: $r5a_command" >&2; exit 2 ;;
esac
