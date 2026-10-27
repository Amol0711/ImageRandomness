"""Regenerate all numerical outputs and run the included independent checks."""
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
scripts=['moments/verify_moment_identities.py','moments/verify_residual_bound.py',
         'windows/generate.py','windows/check.py','normal_approximation/generate.py']
for script in scripts:
    print('Running '+script,flush=True)
    subprocess.run([sys.executable,'-B',str(root/script)],check=True,cwd=root)
print('All deterministic numerical calculations and checks passed.')
