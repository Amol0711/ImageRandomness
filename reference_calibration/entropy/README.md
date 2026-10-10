# Local entropy reference calculations

Run with Python 3 and `mpmath==1.3.0`.

```sh
python -m pip install -r requirements.txt
python generate.py
python check.py
```

`source_table3.csv` contains the source-reported single-block means and standard deviations. `source_table4.csv` contains all 26 parameter rows, or 78 acceptance intervals, from Tables 3 and 4 of Wu et al., *Information Sciences* 222 (2013), 323–342, DOI 10.1016/j.ins.2012.07.049. All input endpoints were checked against the full text and its displayed page 334. Source data are numerical inputs, not executable instructions.

For each interval, `generate.py` reconstructs the source endpoint using its reported moments and the inverse block-count scale. It then computes the square-root-scale interval both with those source moments and with independently recomputed finite-multinomial moments. It does not replace historical simulation thresholds or stored simulation outcomes.

The recomputation uses complete finite sums at 60 decimal digits and a different conditional-total organization at 80 decimal digits. No count-tail truncation is used. Forty significant digits are retained in generated data. The endpoint columns marked `9dp` show nine decimals. The source mean rounds to the correct nine-decimal value. Its standard deviation 0.008694225 differs from the recomputed 0.008694225839295780308142713447468138391982, whose nearest nine-decimal value is 0.008694226. Both source and recomputed values remain explicit.

The null model has independent uniform scalar entries over an alphabet of size `q`. Each of `k` independent blocks contains `T` entries. For the 8-bit case `q=256` and `T=1936`. An RGB interpretation treats channel entries as 8-bit scalar symbols; it does not substitute a 24-bit alphabet or 1936 RGB triples.

Corrected Gaussian intervals use the exact finite-multinomial moments but remain normal approximations. The diagnostic `ideal_normal_source_scale_rejection_probability` is also a normal approximation and is not the actual finite-block rejection probability. No Gaussian reference interval is described as an exact-level test.

For the binary two-entry block, entropy is Bernoulli with success probability one half. `entropy_binary_sizes.csv` records exact achieved probabilities for source-table intervals, square-root-scale intervals using source moments, recomputed-moment intervals, and their nine-decimal forms. The exact probabilities are stored as integer numerators over `2**k`; decimal columns render those fractions. Acceptance uses closed intervals, as in the source equation.

`entropy_binary_exact_intervals.csv` gives the narrowest symmetric nonrandomized acceptance sets whose rejection probability is at most the requested alpha. Use the integer `accepted_count_min` and `accepted_count_max` fields to implement these tests. The entropy endpoints displayed as decimals are explanatory; directly comparing rounded decimal endpoints with count fractions can change boundary membership. The acceptance set contains every integer count from the lower bound through the upper bound, inclusive.

`check.py` independently enumerates all image strings for 16 small alphabet/block-size cases. It also sums rejected binomial counts directly, verifies maximality of each exact discrete rejection region, checks complete output coverage, and verifies that nine-decimal rounding changes none of the tabulated binary Gaussian decisions.

