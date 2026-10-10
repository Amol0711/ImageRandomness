# Natural-reference moments

Exact conditional moments for fixed 256 by 256 grayscale crops and one predetermined histogram-preserving permutation of each crop. Comparison pixels are independent uniform 8-bit values. No Monte Carlo estimates are used. Means are displayed as percentages; all displayed variances use squared fraction units multiplied by 10^7. Full/Pair divides the full BACI variance by its pair-occurrence independence approximation. Pool/Fixed divides pooled UACI variance by fixed-coordinate UACI variance. UACI denotes unified average changing intensity. Original and permuted arrangements have exactly equal UACI moments. CSV means are fractions, variances and excesses are squared fractions, and ratios are dimensionless.

## BACI and spatial arrangement

| Image | Arrangement | BACI mean (%) | BACI variance x 10^7 | Full/Pair |
| --- | --- | --- | --- | --- |
| Camera | Original | 25.38598 | 3.44565 | 3.83864 |
| Camera | Permuted | 27.04868 | 4.69892 | 4.74105 |
| Clock | Original | 18.73263 | 1.93072 | 4.03663 |
| Clock | Permuted | 19.17995 | 2.49787 | 5.01799 |
| Grass | Original | 20.34961 | 2.57661 | 4.51052 |
| Grass | Permuted | 20.61927 | 2.95054 | 5.06470 |

## UACI and histogram preservation

| Image | UACI mean (%) | Fixed variance x 10^7 | Pooling excess x 10^7 | Pool/Fixed |
| --- | --- | --- | --- | --- |
| Camera | 33.80241 | 7.81335 | 0.90179 | 1.11542 |
| Clock | 26.68068 | 4.19386 | 0.18440 | 1.04397 |
| Grass | 27.64853 | 4.87347 | 0.18506 | 1.03797 |

Exact rational outputs, image provenance and permutation instructions remain in ../../fixed_image_and_power/natural_images/. See ../README.md.
