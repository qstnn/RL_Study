#!/usr/bin/env bash
set -eo pipefail
r4_play_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
r4_play_repo="$(cd -- "$r4_play_dir/../../../../../../../.." && pwd)"
r4_play_python="${R4_PYTHON:-$HOME/miniconda3/envs/lab/bin/python}"
r4_play_sim="${R4_ISAAC_SIM:-$r4_play_repo/../isaac-sim}"
r4_play_lab="${R4_ISAACLAB:-$r4_play_repo/../Isaaclab}"
export TERM=xterm PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
r4_play_cpu_only=false
for r4_play_arg in "$@"; do
    case "$r4_play_arg" in --check|-h|--help) r4_play_cpu_only=true ;; esac
done
if [[ "$r4_play_cpu_only" == false ]]; then
    export LD_LIBRARY_PATH="$(dirname "$(dirname "$r4_play_python")")/lib:${LD_LIBRARY_PATH:-}"
    export ISAACSIM_PATH="$r4_play_sim" ISAACLAB_PATH="$r4_play_lab" ISAACSIM_PYTHON_EXE="$r4_play_python"
    source "$r4_play_sim/setup_conda_env.sh"
fi
export PYTHONPATH="$r4_play_repo/source/rl_training:$r4_play_lab/source/isaaclab:$r4_play_lab/source/isaaclab_assets:$r4_play_lab/source/isaaclab_tasks:$r4_play_lab/source/isaaclab_rl:$r4_play_dir/../structured_v1:$r4_play_dir/../../:${PYTHONPATH:-}"
exec "$r4_play_python" -u "$r4_play_dir/play.py" "$@"
