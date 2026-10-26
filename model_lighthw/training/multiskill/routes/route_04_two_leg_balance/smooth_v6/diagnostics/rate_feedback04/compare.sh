#!/usr/bin/env bash
set -eo pipefail
r4_rate_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
r4_rate_repo="$(cd -- "$r4_rate_dir/../../../../../../../.." && pwd)"
r4_rate_python="${R4_PYTHON:-$HOME/miniconda3/envs/lab/bin/python}"
r4_rate_sim="${R4_ISAAC_SIM:-$r4_rate_repo/../isaac-sim}"
r4_rate_lab="${R4_ISAACLAB:-$r4_rate_repo/../Isaaclab}"
export TERM=xterm PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
case "${1:-run}" in
  --verify|--check|--help) ;;
  *)
    export LD_LIBRARY_PATH="$(dirname "$(dirname "$r4_rate_python")")/lib:${LD_LIBRARY_PATH:-}"
    export ISAACSIM_PATH="$r4_rate_sim" ISAACLAB_PATH="$r4_rate_lab" ISAACSIM_PYTHON_EXE="$r4_rate_python"
    source "$r4_rate_sim/setup_conda_env.sh"
    ;;
esac
export PYTHONPATH="$r4_rate_repo/source/rl_training:$r4_rate_lab/source/isaaclab:$r4_rate_lab/source/isaaclab_assets:$r4_rate_lab/source/isaaclab_tasks:$r4_rate_lab/source/isaaclab_rl:${PYTHONPATH:-}"
exec "$r4_rate_python" "$r4_rate_dir/compare.py" "$@"
