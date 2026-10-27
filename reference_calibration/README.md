# Image-randomness reference calibration

This component contains null-reference calculations, critical values and
verification code.
The numerical calculations use no new Monte Carlo draws.

## Reproduce

Use Python 3.10 or later. Install the single pinned dependency and run

```sh
python -m pip install -r requirements.txt
python reproduce.py
```

The BACI module uses only the Python standard library. The entropy module uses
mpmath for complete finite sums at 60 and 80 decimal digits. Full entropy
regeneration takes longer than the other checks because every finite count
term is included. To check the saved results without repeating those large
moment sums, run

```sh
python baci/check_protocol_calibration.py
python entropy/check.py
```

The `baci/` directory includes exact grid moments, Gaussian intervals,
independent strict-majority probabilities, same-data majority/tolerance joint
probabilities and calibrated count thresholds. The `entropy/` directory
includes all 78 original interval inputs, corrected intervals under both
source-reported and recomputed moments, exact binary achieved sizes, and
39 exact symmetric binary acceptance rules.

Gaussian probabilities approximate finite-image distributions. Exact binary
probabilities and independent-component count guarantees are identified
separately. The component README files specify the models, units, strict or
closed interval conventions, technical references and verification methods.
Simulation observations are separate from these deterministic calculations.

No network access is required after dependency installation.
