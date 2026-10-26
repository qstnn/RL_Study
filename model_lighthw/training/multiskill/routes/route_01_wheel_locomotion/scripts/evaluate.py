"""Route01 evaluate entry; shared lifecycle retains route-local outputs."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from run import main

if __name__ == "__main__":
    raise SystemExit(main("evaluate"))
