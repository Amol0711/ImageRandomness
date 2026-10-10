# BACI aggregate null calibration

This module reproduces finite-grid BACI moments, Gaussian-surrogate intervals,
strict-majority probabilities, same-data combinations of majority and relative
tolerance rules, and calibrated independent-test rejection counts. It uses the
Python standard library and contains no measured cipher outputs.

Run from this directory with Python 3.10 or later.

```sh
python protocol_calibration.py
python check_protocol_calibration.py
```

The first command regenerates `data/`. The second runs mathematical and numerical
checks and regenerates the same files. Generation takes seconds on an ordinary
computer. No random sampling is required.

## Model and units

The numerical reference model has two independent 256 by 256 images whose pixels
are independent uniform symbols from 0 through 255. BACI is a proportion. Image
replication averages `R` independent whole-image BACI values. Outer replication
counts `m` independent averages. Their roles are different.

All interval endpoints and half-widths in `gaussian_intervals.csv` use percent
units. Half-widths are measured in percentage points. The relative tolerance
`epsilon=0.001` means 0.1 percent of the null mean and corresponds here to
approximately 0.026771242 percentage points. Probabilities are proportions.

## Files and assumptions

| File | Contents |
| --- | --- |
| `moments.json` | Exact rational mean and covariance calculations and their decimals |
| `gaussian_intervals.csv` | 16 intervals for four replication counts, two component levels, and two variance rules |
| `gaussian_strict_majority.csv` | 64 independent homogeneous strict-majority calculations |
| `gaussian_joint_rules.csv` | 32 calculations retaining the dependence between same-data majority and tolerance criteria |
| `independent_count_cutoffs.csv` | 12 conservative aggregate count thresholds and their binomial bounds |

The normal probabilities are analytical evaluations, subject to floating-point
rounding, for an explicitly specified Gaussian model. They approximate
finite-image BACI probabilities. Using exact means and variances does not make
normal-tail probabilities exact for the finite image distribution.

The count thresholds provide finite-sample error control when the component
decisions are independent and each has rejection probability at most the stated
bound. The cutoff selection uses exact rational arithmetic for the supplied
decimal levels. A nominal Gaussian interval does not itself establish the needed
finite-sample component bound. Independent-null rank tests can supply that bound
if their independent reference banks are also independent across component tests.
A shared reference bank generally invalidates a simple unconditional binomial
calculation even if fresh observations are independent.

The numerical majority table uses one homogeneous null law and independent outer
comparisons. It does not reanalyse the heterogeneous 15-case cipher experiment in
Zhang (2021). That source offers a majority criterion and a relative-error
criterion as alternatives. The same-data union and intersection are explicitly
specified additional combinations studied here; neither treats their events as
independent.

The general polynomial functions support heterogeneous independent comparisons.
Joint rule calculations allow either ordering of the two centered interval
radii at each comparison and allow that ordering to differ across comparisons.
Dependence between different outer comparisons requires a joint null model and
is not inferred from reported averages.

## Checks

Checks cover exact-support moment calculations against closed forms for six
alphabet sizes, Bernoulli convolution against enumeration, strict endpoint and
majority conventions, same-data joint events against exhaustive category
enumeration in both nesting directions, and aggregate cutoff bounds and
minimality using independent exact rational binomial sums. Extremely small
rejection probabilities are evaluated directly instead of subtracting a nearly
unit acceptance probability from one.

## Technical reference

Y. Zhang, “Statistical test criteria for sensitivity indexes of image
cryptosystems,” Information Sciences 550 (2021), 313–328,
https://doi.org/10.1016/j.ins.2020.10.026.
The replication definitions and decision criteria are in Section 4
and Remarks 3 and 4. The formulas implemented here retain overlapping-pair
covariance when calculating the full BACI variance.
