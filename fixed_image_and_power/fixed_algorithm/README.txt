EXACT FIXED-IMAGE MOMENTS

This module computes conditional moments for a fixed integer image and an
independent comparison image whose pixels are uniform on {0,...,q-1}.
BACI uses every overlapping 2x2 window and all six unordered pair distances
inside each window. UACI uses ordinary absolute pixel differences. Outputs
use fraction units in [0,1], not percentage units. Variances use squared
fraction units. This directory contains numerical code and reproducible
numerical examples only.

Requirements
Python 3.10 or newer and NumPy. The calculations do not need SciPy, image
libraries, symbolic software, random-number generators, or a network connection.

Quick example
Place this directory on the Python import path, or run Python from it.

    import numpy as np
    from fixed_image_moments import fixed_baci_moments, fixed_uaci_moments

    fixed = np.array([[0, 64], [128, 255]], dtype=np.uint8)
    b = fixed_baci_moments(fixed, q=256, repeats=1, exact=True)
    u = fixed_uaci_moments(fixed, q=256, repeats=1, exact=True)
    print(b.mean, b.variance, b.pair_occurrence_variance)
    print(u.mean, u.variance, u.pooled_variance, u.pooled_excess)

Callable contract
fixed_baci_moments(image, q=256, *, repeats=1, exact=True)
fixed_uaci_moments(image, q=256, *, repeats=1, exact=True)

* q is an integer at least 2. repeats is an integer at least 1.
* image contains integer values from 0 through q-1. Boolean, floating, object,
  out-of-range and empty inputs are rejected. Integer lists are accepted when
  NumPy converts them to an integer array. Pixel storage must fit signed int64.
* BACI requires one 2D image with both dimensions at least 2. It implements
  overlapping 2x2 windows only. It does not accept color channels or batches.
* UACI accepts one nonempty 1D or 2D image.
* repeats refers to independent repetitions of the entire comparison image,
  with the same fixed reference. Mean outputs do not change. All variance
  outputs are divided by repeats exactly once.
* exact=True returns fractions.Fraction values for every moment. exact=False
  converts those final fractions to Python floats. Both options perform the
  same exact integer computation before conversion.

BACI return fields
mean                         normalized E[B]
variance                     full Var(B), retaining all covariance
pair_occurrence_variance     sum_e c_e Var(h_e) / (repeats*d^2)
independent_edge_variance     sum_e c_e^2 Var(h_e) / (repeats*d^2)
vertex_projection_variance   nonnegative vertex component of full variance
edge_residual_variance       nonnegative edge component of full variance
raw_mean                     E[S], the unnormalized sum before division by d
normalization                d=6*(M-1)*(N-1)*(q-1)
pixels, windows              M*N and (M-1)*(N-1)
repeats                      supplied independent whole-image count
reflected_classes            distinct min(v,q-1-v) values in the fixed image
arithmetic                   which exact integer array path was used

Each unique edge e connects two pixels that share a window, c_e is its number
of window occurrences, and h_e is the absolute difference of their difference
values. The two diagnostic variances omit dependence and are not substitutes
for the full variance. The two nonnegative variance components sum exactly
to variance. raw_mean is suitable for exact centering of integer BACI sums.

UACI return fields
mean                         mean absolute difference divided by q-1
variance                     fixed-coordinate variance
pooled_variance              independent empirical-intensity pooling variance
pooled_excess                pooled_variance minus variance, always nonnegative
pixels, repeats              number of pixels and independent image count

The pooled law redraws a reference intensity independently at each coordinate
from the reference histogram. It is a different sampling design from holding
the image fixed. Its excess is zero for a reference whose pixel intensities
all belong to one reflected pair {a,q-1-a}; constancy is not necessary.

Computational method and integer safety
Identical conditional laws are reused, including reflected intensities.
Conditional absolute-distance expectations use prefix sums. Pair tables
are computed once per reflected class pair. Vertex projections are evaluated
one row at a time; an image-sized array of q-component projections is never
stored. Local int64 arrays are used only when the conservative bound
288*q^4*(q-1)^2 <= 2^63-1 guarantees safe intermediates. Larger alphabets use
arbitrary-precision integer arrays. Every large nonnegative int64 reduction
is divided into provably safe partial sums; Python integers combine all
partial and grid totals. There is no fixed numerical q upper limit apart
from array-index capacity, but large alphabets require more time and memory.

For n=M*N and t reflected classes, BACI uses O((n+t^2)*q) arithmetic operations
and O(n+N*q+t*q+t^2+q) auxiliary storage. UACI uses O(n+q) operations and
O(n+q) auxiliary storage, including possible conversion of the input image to
signed int64. Its extra storage is O(q) when no conversion is needed. These
are operation and stored-number counts, not integer bit-complexity bounds.

Reproduction
Run python run_examples.py. This regenerates fixed_pattern_moments.json for
five deterministic 256x256 patterns with q=256. There are no random draws.
The JSON records exact numerators, denominators and decimal conversions.
Use --output PATH to select a different output location.
The independent verification companion is algorithm_checks/verify_fixed_algorithm.py.
From the parent computational directory, run

    python algorithm_checks/verify_fixed_algorithm.py

Its default module path points to this sibling module. Its default JSON output is written in the working directory.

Interpretation
Exact moments do not specify an exact finite-sample acceptance interval.
Gaussian tail approximations and independent-null rank calibration are
separate procedures. The model conditions on the complete supplied image;
it does not estimate or infer independence from the pixel data.
