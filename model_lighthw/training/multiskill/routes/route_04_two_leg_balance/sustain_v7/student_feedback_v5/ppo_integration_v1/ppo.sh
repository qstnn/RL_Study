#!/usr/bin/env bash
set -eo pipefail
r4_ppo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
r4_ppo_repo="$(cd -- "$r4_ppo_dir/../../../../../../../.." && pwd)"
r4_ppo_python="${R4_PYTHON:-$HOME/miniconda3/envs/lab/bin/python}"
r4_ppo_sim="${R4_ISAAC_SIM:-$r4_ppo_repo/../isaac-sim}"
r4_ppo_lab="${R4_ISAACLAB:-$r4_ppo_repo/../Isaaclab}"
export TERM=xterm PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
case "${1:---help}" in
  smoke|resume-smoke|formal-smoke|train|worker)
    export LD_LIBRARY_PATH="$(dirname "$(dirname "$r4_ppo_python")")/lib:${LD_LIBRARY_PATH:-}"
    export ISAACSIM_PATH="$r4_ppo_sim" ISAACLAB_PATH="$r4_ppo_lab" ISAACSIM_PYTHON_EXE="$r4_ppo_python"
    source "$r4_ppo_sim/setup_conda_env.sh"
    ;;
esac
export PYTHONPATH="$r4_ppo_repo/source/rl_training:$r4_ppo_lab/source/isaaclab:$r4_ppo_lab/source/isaaclab_assets:$r4_ppo_lab/source/isaaclab_tasks:$r4_ppo_lab/source/isaaclab_rl:$r4_ppo_dir/../structured_v1:$r4_ppo_dir/../../:$r4_ppo_dir/../ppo_design:${PYTHONPATH:-}"
exec "$r4_ppo_python" "$r4_ppo_dir/run_ppo.py" "$@"
