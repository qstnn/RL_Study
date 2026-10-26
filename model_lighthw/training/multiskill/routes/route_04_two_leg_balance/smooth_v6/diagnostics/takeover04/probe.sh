#!/usr/bin/env bash
set -eo pipefail
r4_probe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
r4_probe_repo="$(cd -- "$r4_probe_dir/../../../../../../../.." && pwd)"
r4_probe_python="${R4_PYTHON:-$HOME/miniconda3/envs/lab/bin/python}"
r4_probe_sim="${R4_ISAAC_SIM:-$r4_probe_repo/../isaac-sim}"
r4_probe_lab="${R4_ISAACLAB:-$r4_probe_repo/../Isaaclab}"
export TERM=xterm PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
case "${1:-run}" in
  --verify|--check|--baseline-check|--help) ;;
  *)
    export LD_LIBRARY_PATH="$(dirname "$(dirname "$r4_probe_python")")/lib:${LD_LIBRARY_PATH:-}"
    export ISAACSIM_PATH="$r4_probe_sim" ISAACLAB_PATH="$r4_probe_lab" ISAACSIM_PYTHON_EXE="$r4_probe_python"
    source "$r4_probe_sim/setup_conda_env.sh"
    ;;
esac
export PYTHONPATH="$r4_probe_repo/source/rl_training:$r4_probe_lab/source/isaaclab:$r4_probe_lab/source/isaaclab_assets:$r4_probe_lab/source/isaaclab_tasks:$r4_probe_lab/source/isaaclab_rl:${PYTHONPATH:-}"
exec "$r4_probe_python" "$r4_probe_dir/probe.py" "$@"
