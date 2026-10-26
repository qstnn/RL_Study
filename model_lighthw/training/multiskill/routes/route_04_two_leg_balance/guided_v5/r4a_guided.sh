#!/usr/bin/env bash
set -eo pipefail
r4_guided_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
r4_guided_repo="$(cd -- "$r4_guided_dir/../../../../../.." && pwd)"
r4_guided_python="${R4_PYTHON:-$HOME/miniconda3/envs/lab/bin/python}"
r4_guided_sim="${R4_ISAAC_SIM:-$r4_guided_repo/../isaac-sim}"
r4_guided_lab="${R4_ISAACLAB:-$r4_guided_repo/../Isaaclab}"
export TERM=xterm PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
case "${1:-check}" in
  check|collect|pretrain|--help) ;;
  *)
    export LD_LIBRARY_PATH="$(dirname "$(dirname "$r4_guided_python")")/lib:${LD_LIBRARY_PATH:-}"
    export ISAACSIM_PATH="$r4_guided_sim" ISAACLAB_PATH="$r4_guided_lab" ISAACSIM_PYTHON_EXE="$r4_guided_python"
    source "$r4_guided_sim/setup_conda_env.sh"
    ;;
esac
export PYTHONPATH="$r4_guided_repo/source/rl_training:$r4_guided_lab/source/isaaclab:$r4_guided_lab/source/isaaclab_assets:$r4_guided_lab/source/isaaclab_tasks:$r4_guided_lab/source/isaaclab_rl:${PYTHONPATH:-}"
exec "$r4_guided_python" "$r4_guided_dir/run.py" "${@:-check}"
