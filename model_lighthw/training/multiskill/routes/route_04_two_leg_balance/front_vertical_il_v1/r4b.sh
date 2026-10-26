#!/usr/bin/env bash
set -eo pipefail
r4b_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
r4b_repo="$(cd -- "$r4b_dir/../../../../../.." && pwd)"
r4b_python="${R4_PYTHON:-$HOME/miniconda3/envs/lab/bin/python}"
r4b_sim="${R4_ISAAC_SIM:-$r4b_repo/../isaac-sim}"
r4b_lab="${R4_ISAACLAB:-$r4b_repo/../Isaaclab}"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 TERM=xterm
case "${1:---help}" in
    check|-h|--help) ;;
    *)
        export LD_LIBRARY_PATH="$(dirname "$(dirname "$r4b_python")")/lib:${LD_LIBRARY_PATH:-}"
        export ISAACSIM_PATH="$r4b_sim" ISAACLAB_PATH="$r4b_lab" ISAACSIM_PYTHON_EXE="$r4b_python"
        source "$r4b_sim/setup_conda_env.sh"
        ;;
esac
export PYTHONPATH="$r4b_repo/source/rl_training:$r4b_lab/source/isaaclab:$r4b_lab/source/isaaclab_assets:$r4b_lab/source/isaaclab_tasks:$r4b_lab/source/isaaclab_rl:${PYTHONPATH:-}"
exec "$r4b_python" -u "$r4b_dir/run.py" "$@"
