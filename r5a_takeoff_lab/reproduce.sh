#!/usr/bin/env bash
set -euo pipefail
probe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
probe_root="$(dirname -- "$probe_dir")"
exec bash "$probe_root/r5a.sh" python "$probe_dir/run_all.py" "$@"
