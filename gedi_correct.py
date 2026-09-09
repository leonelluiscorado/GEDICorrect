"""Backward-compatible launcher for repository users.

Installed users should prefer ``gedicorrect run``.
"""

import sys
from pathlib import Path

try:
    from gedicorrect.cli import main
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from gedicorrect.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["run", *sys.argv[1:]]))
