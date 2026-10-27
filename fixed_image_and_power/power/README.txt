PAIRED POWER EXPERIMENT

This directory contains simulation code, the registered design, raw integer
statistics and numerical outputs for a comparison of blocked average changing
intensity (BACI), unified average changing intensity (UACI) and number of pixels
change rate (NPCR).

Reproduction

  python3 run.py
  python3 check.py

The original run used Python 3.12.14, NumPy 2.3.5 and SciPy 1.17.0.
The design file fixes all dimensions, sample sizes, alternative strengths,
random streams and decision rules. Run from any working directory. To retain
the supplied outputs, use python3 run.py --output new_results.
To rebuild summaries without generating new images, use
python3 run.py --reuse-raw.

The root seed and stream identifiers feed NumPy SeedSequence and PCG64.
The registered batch size is part of the reproducible random-generation
schedule and must remain unchanged for byte-for-byte raw-array reproduction.

Reference model and decisions

The reference model consists of two independent 64 by 64 images with independent
uniform integer pixels from 0 through 255. The three two-sided scores use exact
null expectations. An independent bank of 99,999 image pairs supplies the
thresholds. Independent uniform tie-breakers randomize discrete boundaries.
At significance levels 0.05 and 0.01, the rule has the stated exact marginal
null rejection probability over independent calibration-bank and test draws.
A particular realized bank can have a different conditional rejection
probability. The 20,000 independent null-validation draws measure that value.
Intervals describe conditional simulation uncertainty for the realized bank.

Alternatives

Identity contamination sets V equal to U independently at each pixel with
the configured probability. Other pixels remain independent uniform draws.

Horizontal copying partitions each row into disjoint adjacent pairs. At each
pair, independently with the configured probability, both image pixels at
the second position copy their respective first-position pixels. This keeps
the distribution of each individual difference unchanged and introduces
dependence within those pairs. The same generated image pair is used for all
three statistics. Zero strength uses the common null-validation sample.

Files

design.json records the plan fixed before simulation. run.py generates all
outputs and check.py verifies checksums, direct small-grid calculations,
rank boundaries, raw-array summaries and mean diagnostics.

results/raw/*.npz stores statistics and tie_uniforms. Each array has three
columns, in the order BACI, UACI and NPCR. The statistics columns are the
unnormalized BACI six-comparison sum, total absolute pixel difference and
number of unequal pixels. They are integers, preserving exact score ties.
Each row is one paired replication. The calibration and null-validation
arrays are separate from all alternative arrays.

results/thresholds.json records integer-score and uniform thresholds.
results/power_summary.csv reports rejection counts and 95 percent Wilson
intervals. results/paired_differences.csv reports BACI-minus-comparator
differences and discordant counts with conservative exact 95 percent intervals.
results/npcr_conditional_size.json evaluates NPCR's realized-bank size from its binomial null law.
results/sha256.json verifies the supplied output files.

The alternative strengths were not changed after observing the results.
The experiment compares sensitivity to its two declared departures and does
not establish universal superiority of any statistic.
