#!/usr/bin/env bash
set -eo pipefail
r4_trigger_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
r4_trigger_repo="$(cd -- "$r4_trigger_dir/../../../../../../../.." && pwd)"
r4_trigger_python="${R4_PYTHON:-$HOME/miniconda3/envs/lab/bin/python}"
r4_trigger_sim="${R4_ISAAC_SIM:-$r4_trigger_repo/../isaac-sim}"
r4_trigger_lab="${R4_ISAACLAB:-$r4_trigger_repo/../Isaaclab}"
export TERM=xterm PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
case "${1:-run}" in
  --verify|--check|--help) ;;
  *)
    export LD_LIBRARY_PATH="$(dirname "$(dirname "$r4_trigger_python")")/lib:${LD_LIBRARY_PATH:-}"
    export ISAACSIM_PATH="$r4_trigger_sim" ISAACLAB_PATH="$r4_trigger_lab" ISAACSIM_PYTHON_EXE="$r4_trigger_python"
    source "$r4_trigger_sim/setup_conda_env.sh"
    ;;
esac
export PYTHONPATH="$r4_trigger_repo/source/rl_training:$r4_trigger_lab/source/isaaclab:$r4_trigger_lab/source/isaaclab_assets:$r4_trigger_lab/source/isaaclab_tasks:$r4_trigger_lab/source/isaaclab_rl:${PYTHONPATH:-}"
exec "$r4_trigger_python" "$r4_trigger_dir/compare.py" "$@"
