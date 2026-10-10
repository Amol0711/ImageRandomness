# Complete repeated-image BACI calibration

Null rejection percentages for averages of R independent 256 by 256 BACI image-pair statistics, with independent uniform 8-bit pixels in each random image. Each R uses 5000 evaluation averages and a separate independent bank of 9999 averages. Both nominal levels use those same evaluations. Tabulated applies Zhang's decimal endpoints with acceptance strictly between them. The three Gaussian rules use pair-occurrence independence, within-window covariance, or full covariance. Exact moments do not make their finite-sample tails Gaussian. Null rank includes upper-tail ties; its finite-sample guarantee is marginal over bank and observation. The CSV retains counts and Wilson 95% intervals; the rank intervals condition on the realized bank. Rates in this table are percentages; CSV rejection_rate and Wilson endpoints without a percent suffix are fractions.

| Independent whole-image repeats R | Nominal level (%) | Tabulated (%) | Pair independence (%) | Within-window (%) | Full covariance (%) | Null rank (%) |
| --- | --- | --- | --- | --- | --- | --- |
| 50 | 1 | 25.66 | 25.60 | 14.26 | 0.86 | 0.74 |
| 50 | 5 | 38.28 | 38.40 | 26.38 | 5.34 | 4.92 |
| 100 | 1 | 26.00 | 26.04 | 14.08 | 1.06 | 1.22 |
| 100 | 5 | 39.22 | 39.22 | 26.76 | 5.02 | 5.42 |

BACI means blocked average changing intensity. See ../README.md for the data and method crosswalk and Zhang (2021) citation.
