#!/usr/bin/env bash
# All changes below apply only to this subprocess; .bashrc and existing conda envs are not edited.
set -eo pipefail
r6_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
r6_command="${1:-help}"
if [[ "$r6_command" == help || "$r6_command" == --help ]]; then
    printf '%s\n' 'R6 USB: bash r6.sh prepare | doctor | check | smoke | train-smoke | train [options] | play [options] | evaluate [options] | review --run PATH | collect --session PATH' 'Override paths with R6_PYTHON=/absolute/lab/bin/python and R6_ISAAC_SIM=/absolute/isaac-sim.'
    exit 0
fi
shift
r6_python="${R6_PYTHON:-}"
if [[ -z "$r6_python" ]]; then
    for r6_candidate in "$HOME/.conda/envs/lab/bin/python" "$HOME/miniconda3/envs/lab/bin/python" "$HOME/anaconda3/envs/lab/bin/python"; do
        if [[ -x "$r6_candidate" ]]; then r6_python="$r6_candidate"; break; fi
    done
fi
if [[ ! -x "$r6_python" ]]; then printf '%s\n' '[R6 USB ERROR] Set R6_PYTHON to the absolute Python 3.11 lab/bin/python path.' >&2; exit 2; fi
r6_sim="${R6_ISAAC_SIM:-}"
if [[ -z "$r6_sim" ]]; then
    for r6_candidate in "$HOME/IsaacCodeAgent/IsaacLab/_isaac_sim" "$HOME/IsaacCodeAgent/isaac-sim" "$r6_root/Isaaclab/_isaac_sim" "$HOME/RL_lab/isaac-sim" "$HOME/RL_Lab/isaac-sim"; do
        if [[ -f "$r6_candidate/setup_conda_env.sh" && -f "$r6_candidate/python_packages/isaacsim/__init__.py" && -d "$r6_candidate/kit/kernel/py/carb" ]]; then r6_sim="$r6_candidate"; break; fi
    done
fi
if [[ ! -f "$r6_sim/setup_conda_env.sh" || ! -f "$r6_sim/python_packages/isaacsim/__init__.py" || ! -d "$r6_sim/kit/kernel/py/carb" ]]; then
    printf '%s\n' '[R6 USB ERROR] Set R6_ISAAC_SIM to the complete Isaac Sim installation containing setup_conda_env.sh, python_packages/isaacsim and kit/kernel/py/carb.' >&2; exit 2
fi
r6_sim="$(readlink -f -- "$r6_sim")"
r6_deps="$("$r6_python" -I "$r6_root/offline/manage.py" path)"
export R6_DEPS="$r6_deps" R6_PROJECT_ROOT="$r6_root" R6_PYTHON="$r6_python" R6_ISAAC_SIM="$r6_sim"
export ISAACSIM_PATH="$r6_sim" ISAACLAB_PATH="$r6_root/Isaaclab" ISAACSIM_PYTHON_EXE="$r6_python"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TERM=xterm GIT_PYTHON_REFRESH=quiet
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
unset PYTHONHOME
export PYTHONPATH=""
export LD_LIBRARY_PATH="$(dirname "$(dirname "$r6_python")")/lib"
source "$r6_sim/setup_conda_env.sh"
export PYTHONPATH="$r6_root/offline/bootstrap:$r6_root/source/rl_training:$r6_root/Isaaclab/source/isaaclab:$r6_root/Isaaclab/source/isaaclab_assets:$r6_root/Isaaclab/source/isaaclab_tasks:$r6_root/Isaaclab/source/isaaclab_rl:$r6_deps:$r6_deps/cmeel.prefix/lib/python3.11/site-packages:$PYTHONPATH"
if [[ "$r6_command" == prepare ]]; then
    "$r6_python" -I "$r6_root/offline/manage.py" prepare
    exec "$r6_python" "$r6_root/offline/doctor.py"
fi
if [[ ! -f "$(dirname "$r6_deps")/ready.json" ]]; then
    printf '%s\n' '[R6 USB ERROR] Run bash r6.sh prepare once to install the bundled wheels offline.' >&2; exit 2
fi
if [[ "$r6_command" == doctor ]]; then exec "$r6_python" "$r6_root/offline/doctor.py" "$@"; fi
if [[ "$r6_command" == python ]]; then exec "$r6_python" "$@"; fi
case "$r6_command" in
    train|play|review|collect|routes|verify) exec "$r6_python" "$r6_root/offline/station.py" "$r6_command" --route 06 "$@" ;;
esac
case "$r6_command" in
    check|smoke|train-smoke|train|play|evaluate) ;;
    *) printf '%s\n' "[R6 USB ERROR] Unknown command: $r6_command" >&2; exit 2 ;;
esac
"$r6_python" "$r6_root/offline/doctor.py" --brief
r6_entry="$r6_root/model_lighthw/training/multiskill/run.py"
case "$r6_command" in
    check) exec "$r6_python" "$r6_entry" check --route 06 "$@" ;;
    smoke) exec "$r6_python" "$r6_entry" smoke --route 06 --headless --num_envs 32 "$@" ;;
    train-smoke) exec "$r6_python" "$r6_entry" train --route 06 --headless --num_envs 32 --max_iterations 2 "$@" ;;
    train) exec "$r6_python" "$r6_entry" train --route 06 --headless --num_envs 1024 --max_iterations 1000 "$@" ;;
    play|evaluate) exec "$r6_python" "$r6_entry" "$r6_command" --route 06 "$@" ;;
esac
