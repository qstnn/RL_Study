#!/usr/bin/env bash
set -eo pipefail
r4_eval_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
r4_eval_repo="$(cd -- "$r4_eval_dir/../../../../../../../.." && pwd)"
r4_eval_python="${R4_PYTHON:-$HOME/miniconda3/envs/lab/bin/python}"
r4_eval_sim="${R4_ISAAC_SIM:-$r4_eval_repo/../isaac-sim}"
r4_eval_lab="${R4_ISAACLAB:-$r4_eval_repo/../Isaaclab}"
export TERM=xterm PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
case "${1:---help}" in
  evaluate|worker)
    export LD_LIBRARY_PATH="$(dirname "$(dirname "$r4_eval_python")")/lib:${LD_LIBRARY_PATH:-}"
    export ISAACSIM_PATH="$r4_eval_sim" ISAACLAB_PATH="$r4_eval_lab" ISAACSIM_PYTHON_EXE="$r4_eval_python"
    source "$r4_eval_sim/setup_conda_env.sh"
    ;;
esac
export PYTHONPATH="$r4_eval_repo/source/rl_training:$r4_eval_lab/source/isaaclab:$r4_eval_lab/source/isaaclab_assets:$r4_eval_lab/source/isaaclab_tasks:$r4_eval_lab/source/isaaclab_rl:$r4_eval_dir/../structured_v1:$r4_eval_dir/../../:${PYTHONPATH:-}"
exec "$r4_eval_python" "$r4_eval_dir/run_eval.py" "$@"
