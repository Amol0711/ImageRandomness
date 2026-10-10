Fixed grayscale reference-image computations

This directory contains three metadata-free uint8 pixel arrays, a reproducible
fixed spatial permutation, exact numerical moment results, and verification
scripts. The images are fixed reference inputs. Comparison-image pixels follow
the independent uniform distribution on {0,...,255}. No encrypted image or
security claim is involved. No Monte Carlo sampling is used in these tables.

Requirements
Python 3.10 or later and NumPy. Pillow is needed only to recreate pixel arrays
from the upstream PNG files. The sibling fixed_algorithm/fixed_image_moments.py
module is required to compute the moments.

Reproduce
    python natural_images/prepare_inputs.py
    python natural_images/run_natural_images.py

The commands also work with absolute script paths. The first command verifies
the bundled inputs and recreates input_manifest.json. The second command writes
decimal CSV values, exact rational JSON values, and independent verification
results into natural_images/outputs/.

Retrieve upstream bytes and recreate the crops when desired
    python natural_images/prepare_inputs.py --fetch

This optional command checks the upstream source hashes before decoding. It
needs an internet connection and Pillow. It does not change any source pixel
value except by selecting the stated center crop.

Input arrays
All arrays are centered 256x256 crops from scikit-image 0.25.2 grayscale images.
Zero-based row and column origins are camera (128,128), clock (22,72), and grass
(128,128). No resizing, interpolation, contrast adjustment or gray conversion
is applied. Source PNG dimensions and byte hashes, pixel hashes, URLs, third-
party credits and licenses are in input_manifest.json and THIRD_PARTY.txt.

Permutation control
The same predetermined permutation is applied to the row-major pixel vector
of each crop. Sort indices by the SHA-256 digest of ASCII
fixed-image-permutation-v1, one zero byte, and the original index encoded as
four unsigned big-endian bytes. Break any digest tie by the original index.
This rule is independent of image content and software random-number streams.
Each resulting input remains a fixed image. Its histogram is identical to that
of the original crop. It is one control arrangement, not an estimated average
over random permutations.

Output units and diagnostics
BACI and UACI means are fractions in [0,1]. Every variance and variance excess
uses squared fraction units. The CSV contains no percentage-unit variances.
BACI full_to_pair uses the occurrence-wise approximation that treats the six
pair occurrences in every 2x2 window as independent. The additional full_to_
independent_edge diagnostic keeps squared multiplicities of unique pixel pairs
but omits covariance between distinct edges sharing a pixel. Full variance
includes both effects. UACI pooled_excess is pooled_variance minus the exact
fixed-image variance. Every result is computed with integer arithmetic and
retained as a Fraction before the final decimal conversion.

Independent checks
The reproduction script checks every BACI mean and occurrence-wise variance
against direct summation over the six pairs of every window, using unreduced
conditional laws formed by enumeration. It checks UACI moments by direct
enumeration of all 256 comparison intensities for each reference intensity.
It verifies all full-variance projection decompositions, positivity, histogram
equalities, and exact equality of UACI moments across each permutation pair.

Licensing
The pixel inputs retain their public-domain/CC0 source status and third-party
provenance. This directory contains no manuscript, proofs, review documents,
or manuscript-author metadata. Third-party source credits are retained.
