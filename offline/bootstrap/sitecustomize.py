"""Activate only the USB bundle's wheel installation, including cmeel's .pth."""
import os
import site
from pathlib import Path

_r6_deps = os.environ.get('R6_DEPS')
if _r6_deps and Path(_r6_deps).is_dir():
    site.addsitedir(_r6_deps)
