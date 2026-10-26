#!/usr/bin/env bash
# 真实40 kg被动拖车物理复现；没有PPO或Actor训练。
set -eo pipefail
if (( $# > 1 )); then
  echo 'Usage: bash replay_loaded40.sh [fresh-output-directory]' >&2
  exit 2
fi
R1_V7_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate lab
# Isaac Sim's conda activation reads optional shell variables; enable nounset afterwards.
set -u
export TERM=xterm OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$R1_V7_ROOT"
R1_REPLAY_OUTPUT="${1:-$R1_V7_ROOT/results/loaded40_replay_$(date +%Y%m%d_%H%M%S)}"
python -u scripts/loaded_cart_worker.py \
  --reference-run results/2026-10-22_load_sharing_control/lower120_shift100_rear120_stairs173 \
  --reference-window 99 107 --cart-mass 40 --move-speed 0.3 \
  --rounded-cart-wheels --front-force-share 0.3 \
  --hitch-height 0.04 --robot-hitch-offset 0.20 \
  --ground-friction 3 --loaded-acceleration 0.05 \
  --trial-duration 30 --device cpu --output "$R1_REPLAY_OUTPUT"
python scripts/real_cart_audit.py --run "$R1_REPLAY_OUTPUT" --require-pass --require-hold
