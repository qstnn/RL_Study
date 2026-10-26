#!/usr/bin/env bash
# The same bundled Python/Isaac Sim/dependencies are reused for all station commands.
set -eo pipefail
station_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$station_root/r6.sh" python "$station_root/offline/station.py" "$@"
