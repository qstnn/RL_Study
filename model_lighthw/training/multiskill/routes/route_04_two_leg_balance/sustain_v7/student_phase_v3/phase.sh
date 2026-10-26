#!/usr/bin/env bash
set -eo pipefail
r4_segment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
r4_segment_repo="$(cd -- "$r4_segment_dir/../../../../../../.." && pwd)"
r4_segment_python="${R4_PYTHON:-$HOME/miniconda3/envs/lab/bin/python}"
r4_segment_sim="${R4_ISAAC_SIM:-$r4_segment_repo/../isaac-sim}"
r4_segment_lab="${R4_ISAACLAB:-$r4_segment_repo/../Isaaclab}"
export TERM=xterm PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
case "${1:---help}" in
  collect|evaluate|worker)
    export LD_LIBRARY_PATH="$(dirname "$(dirname "$r4_segment_python")")/lib:${LD_LIBRARY_PATH:-}"
    export ISAACSIM_PATH="$r4_segment_sim" ISAACLAB_PATH="$r4_segment_lab" ISAACSIM_PYTHON_EXE="$r4_segment_python"
    source "$r4_segment_sim/setup_conda_env.sh"
    ;;
esac
export PYTHONPATH="$r4_segment_repo/source/rl_training:$r4_segment_lab/source/isaaclab:$r4_segment_lab/source/isaaclab_assets:$r4_segment_lab/source/isaaclab_tasks:$r4_segment_lab/source/isaaclab_rl:${PYTHONPATH:-}"
exec "$r4_segment_python" "$r4_segment_dir/run.py" "$@"
