#!/usr/bin/env bash
set -eo pipefail
r4_branch_scripts="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$r4_branch_scripts/r4_v4.sh" "$@" --skill R4a
