#!/usr/bin/env bash
set -eo pipefail
r4_scripts="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
r4_repo="$(cd -- "$r4_scripts/../../../../../.." && pwd)"
r4_python="${R4_PYTHON:-$HOME/miniconda3/envs/lab/bin/python}"
r4_sim="${R4_ISAAC_SIM:-$r4_repo/../isaac-sim}"
r4_lab="${R4_ISAACLAB:-$r4_repo/../Isaaclab}"
if [[ ! -x "$r4_python" ]]; then
    printf '%s\n' 'Set R4_PYTHON to the lab Python executable.' >&2; exit 2
fi
export TERM=xterm PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
# CPU检查不要求Isaac Sim启动。训练/评估显式绑定此仓库，避免另一个editable副本抢先导入。
if [[ "${1:-help}" != check && "${1:-help}" != help && "${1:-help}" != --help ]]; then
    if [[ ! -f "$r4_sim/setup_conda_env.sh" ]]; then
        printf '%s\n' 'Set R4_ISAAC_SIM to the complete Isaac Sim installation.' >&2; exit 2
    fi
    export LD_LIBRARY_PATH="$(dirname "$(dirname "$r4_python")")/lib:${LD_LIBRARY_PATH:-}"
    export ISAACSIM_PATH="$r4_sim" ISAACLAB_PATH="$r4_lab" ISAACSIM_PYTHON_EXE="$r4_python"
    source "$r4_sim/setup_conda_env.sh"
fi
export PYTHONPATH="$r4_repo/source/rl_training:$r4_lab/source/isaaclab:$r4_lab/source/isaaclab_assets:$r4_lab/source/isaaclab_tasks:$r4_lab/source/isaaclab_rl:${PYTHONPATH:-}"
exec "$r4_python" "$r4_scripts/run.py" "$@"
