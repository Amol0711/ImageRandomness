# Numerical reference API

This is the original general finite-law implementation. Run Python from the
`core/` directory to import it as `calibration`, or add `core` to the Python
path when running from the consolidated archive root. No installation of the
repository itself is required. The example below starts at the archive root.

```python
import sys
sys.path.insert(0, "core")
from calibration import iid_baci_moments

moments = iid_baci_moments(256, 256, q=256, repeats=50)
print(float(moments.mean))
print(float(moments.variance))
```

`iid_baci_moments`, `fixed_uaci_moments` and `weighted_edge_moments` default
to `exact=True`. This module's `fixed_baci_moments` defaults to `exact=False`;
its rational mode is intended for small reference examples. Statistic
evaluators and `known_null_rank_pvalue` also default to `exact=False`.
`repeats` counts independent repetitions of the complete statistic, not
windows or dependent observations.

For the exact fixed-image BACI algorithm, use
`fixed_image_and_power/fixed_algorithm/fixed_image_moments.py` from the archive
root. That separate module's `fixed_baci_moments` and `fixed_uaci_moments`
default to exact rational outputs and perform integer calculations even when
`exact=False` requests final float conversion. It reuses reflected laws and
evaluates vertex projections one row at a time. The general implementation
in this directory uses a full vertex-by-alphabet projection array and supports
arbitrary independent finite laws and weighted edges. The similarly named
fixed-image routines have different return fields and are not drop-in
replacements for one another.

| Function group | Purpose |
| --- | --- |
| `baci_statistic`, `baci_from_images` | Six-pair distances in every overlapping 2×2 window, normalized to a fraction |
| `iid_pair_constants`, `iid_baci_moments` | Independent uniform-image distance-law and grid moments |
| `fixed_baci_moments`, `weighted_edge_moments` | Nonidentical independent laws and weighted edge statistics |
| `fixed_uaci_moments`, `uaci_statistic`, `npcr_statistic` | Pixelwise magnitude and change statistics |
| `block_entropy`, `count_entropy`, `local_entropy_mean` | Base-2 plug-in entropy from pixels or category counts |
| `gaussian_interval`, `local_entropy_interval` | Approximate Gaussian intervals with explicit variance/standard-error inputs |
| `known_null_rank_pvalue` | Upper-tail rank p-value including ties and the extra evaluation observation |

For BACI moments, `variance` includes the full covariance.
`independent_pair_variance` treats all six distances in every window as
independent. `within_block_only_variance` retains dependence inside each
window but treats windows as independent. For fixed UACI, `pooled_variance`
is the pooled-pixel comparator, and `pooled_excess` is its difference from
the conditional variance.

The tests enumerate small finite state spaces independently, compare literal
window statistics with weighted-edge computations, check exact moments,
exercise nonidentical and signed weighted laws, and validate rank ties and
input domains. From the consolidated archive root, enter `core/` and run

```sh
cd core
python -B -m unittest discover -s tests -v
```
