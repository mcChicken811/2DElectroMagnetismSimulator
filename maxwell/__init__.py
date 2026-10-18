"""Interactive, charge-conserving 2D Maxwell particle-in-cell simulator."""

import os
from pathlib import Path

# Keep application-controlled caches and temporary files inside this project.
ROOT = Path(__file__).resolve().parent.parent
for _name, _directory in (("NUMBA_CACHE_DIR", "numba"), ("TMPDIR", "tmp"),
                          ("XDG_CACHE_HOME", "xdg"), ("CFFIXED_USER_HOME", "cocoa")):
    _path = ROOT / ".cache" / _directory
    _path.mkdir(parents=True, exist_ok=True)
    os.environ[_name] = str(_path)

