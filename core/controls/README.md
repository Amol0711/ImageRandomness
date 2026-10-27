# Exact numerical control

`binary_entropy.py` enumerates the probability mass function of the average
base-2 plug-in entropy of 30 independent, uniform binary blocks of two pixels
each. Each block entropy is either zero or one with equal probability.
The script uses exact rational binomial probabilities and no random draws.

Run from the repository root with a new output directory

```sh
python -B controls/binary_entropy.py --output outputs/binary-entropy
```

`pmf.csv` contains the count, average entropy, probability, and acceptance
indicators for two interval rules. `summary.json` contains the intervals and
exact rejection fractions. The linear rule uses single-block standard
deviation divided by the number of blocks. The square-root rule uses the
standard deviation divided by its square root. The latter is a Gaussian
reference interval; its exact rejection probability is computed here from the
discrete distribution.
