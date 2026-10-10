"""Regenerate numerical reference data and run independent checks."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
for script in ("baci/check_protocol_calibration.py", "entropy/generate.py", "entropy/check.py"):
    subprocess.run([sys.executable, str(root / script)], cwd=root, check=True)
