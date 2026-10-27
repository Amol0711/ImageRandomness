"""Alphabet-enumerating moment algorithms for independent discrete variables.

The fixed-reference model is D_i = |v_i-U_i| with independent U_i uniform on
range(q), and S = sum_{(i,j,c)} c*|D_i-D_j|.  ``enumerate_exact`` computes the
same raw moments as moment_engine using integer prefix sums over the alphabet.
It does not call the piecewise-polynomial engine.  ``enumerate_float`` uses the
same finite-sum identities in NumPy float64 arithmetic; its output is therefore
an approximation to the mathematically exact moments.  No quantile is computed.

Only occurring, non-isolated folded reference classes are materialized.  Only
class pairs that occur on nonzero edges are cached.  Vertex projections are
processed one at a time.  If r classes and K class pairs occur, the arithmetic
cost is O(q*(r+K+n_active+e)+n+e), after graph validation, and the additional
storage is O(q*r+q+n+K).  These are arithmetic counts, not bit-complexity bounds.
Hash-based dictionaries provide expected/amortized cache lookup costs.

``moments_independent`` extends the float64 algorithm to known, independent,
position-specific PMFs on the common support range(q).  It uses one profile per
active vertex and caches only actual edge pairs, without constructing a q by q
kernel matrix.  It does not model dependent variables.
"""
from dataclasses import dataclass
from fractions import Fraction
import math
import sys
from time import perf_counter

from moment_engine import GraphDesign, prepare_graph


DEFAULT_WORKSPACE_BYTES = 512 * 1024**2


class MemoryBudgetExceeded(MemoryError):
    """The prespecified workspace estimate exceeds the requested budget."""

    def __init__(self, estimate_bytes, limit_bytes):
        self.estimate_bytes = estimate_bytes
        self.limit_bytes = limit_bytes
        super().__init__(
            f"estimated alphabet workspace {estimate_bytes} bytes exceeds "
            f"budget {limit_bytes} bytes")


@dataclass(frozen=True)
class ExactAlphabetMoments:
    q: int
    mean_numerator: int
    variance_numerator: int
    diagnostics: dict

    @property
    def mean(self):
        return Fraction(self.mean_numerator, self.q**2)

    @property
    def variance(self):
        return Fraction(self.variance_numerator, self.q**4)

    def normalized(self, divisor):
        divisor = _normalizer(divisor)
        return self.mean/divisor, self.variance/divisor**2


@dataclass(frozen=True)
class FloatMoments:
    mean: float
    variance: float
    diagnostics: dict

    def normalized(self, divisor):
        divisor = float(_normalizer(divisor))
        return self.mean/divisor, self.variance/divisor**2


@dataclass(frozen=True)
class RationalMoments:
    mean: Fraction
    variance: Fraction
    diagnostics: dict

    def normalized(self, divisor):
        divisor = _normalizer(divisor)
        return self.mean/divisor, self.variance/divisor**2


def _normalizer(value):
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise TypeError("normalizer must be an integer or Fraction")
    if value <= 0:
        raise ValueError("normalizer must be positive")
    return Fraction(value)


def _integer(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _graph(n, edges_or_design):
    if isinstance(edges_or_design, GraphDesign):
        if edges_or_design.n != n:
            raise ValueError("graph vertex count differs from the input length")
        return edges_or_design, True
    return prepare_graph(n, edges_or_design), False


def _fixed_inputs(reference, q, edges_or_design):
    _integer(q, "alphabet size", 2)
    reference = tuple(reference)
    for v in reference:
        _integer(v, "reference intensity", 0)
        if v >= q:
            raise ValueError("reference intensity must be less than q")
    design, reused = _graph(len(reference), edges_or_design)
    classes = tuple(min(v, q-1-v) for v in reference)
    active_classes = tuple(sorted({classes[i] for i, adj in
                                   enumerate(design.adjacency) if adj}))
    pairs = tuple(sorted({_pair_key(classes[i], classes[j])
                          for i, j, _ in design.edges}))
    return classes, active_classes, pairs, design, reused


def _pair_key(a, b):
    return (a, b) if a <= b else (b, a)


def _int_bytes(bits):
    """CPython integer storage approximation, including object overhead."""
    digits = max(1, (bits + sys.int_info.bits_per_digit-1)
                 // sys.int_info.bits_per_digit)
    return sys.getsizeof(0) + (digits-1)*sys.int_info.sizeof_digit


def _workspace(q, n, r, k, design, mode):
    if mode not in ("exact", "float64"):
        raise ValueError("mode must be 'exact' or 'float64'")
    # The estimate excludes caller-owned inputs and an already prepared graph.
    # Conservative object/temporary allowances make the guard useful before
    # array allocation, but this estimate is not a measured peak or RSS limit.
    metadata = 4096 + 128*n + 1024*r + 768*k
    if not design.edges:
        profiles = scratch = 0
    elif mode == "float64":
        profiles = 16*q*r
        scratch = 96*q
    else:
        qbits = q.bit_length()
        weighted_bits = 2*qbits + max(1, design.absolute_weight_sum.bit_length())
        profiles = q*r*(16 + _int_bytes(2*qbits+1)) + 128*r
        scratch = q*(16 + _int_bytes(weighted_bits+2))
        metadata += k*(2*_int_bytes(6*qbits+4))
    return {"total_bytes": metadata+profiles+scratch,
            "profile_bytes": profiles, "scratch_bytes": scratch,
            "metadata_bytes": metadata, "q": q, "vertices": n,
            "classes": r, "class_pairs": k, "mode": mode,
            "scope": "estimated additional array/cache workspace; excludes "
                     "caller inputs, prepared graph, allocator and interpreter overhead"}


def estimate_workspace(reference, q, edges_or_design, *, mode="float64"):
    """Estimate live array/cache storage without allocating alphabet arrays.

    Graph validation and small O(n+e) class/pair metadata are constructed first.
    This is a prespecified omission rule, not an operating-system memory cap.
    """
    classes, active, pairs, design, _ = _fixed_inputs(reference, q, edges_or_design)
    return _workspace(q, len(classes), len(active), len(pairs), design, mode)


def _budget(estimate, limit):
    if limit is not None:
        _integer(limit, "max_workspace_bytes", 0)
        if estimate["total_bytes"] > limit:
            raise MemoryBudgetExceeded(estimate["total_bytes"], limit)


def _exact_profile(m, q):
    """Return integer mass counts and q times the conditional mean distance."""
    counts = [0]*q
    # Enumerating U directly is independent of the piecewise engine's law
    # construction. Folded m and its reflected reference have identical laws.
    for u in range(q):
        counts[abs(m-u)] += 1
    first = sum(x*a for x, a in enumerate(counts))
    second = sum(x*x*a for x, a in enumerate(counts))
    prefix_count = prefix_sum = 0
    profile = [0]*q
    for x, a in enumerate(counts):
        prefix_count += a
        prefix_sum += x*a
        profile[x] = x*(2*prefix_count-q) + first - 2*prefix_sum
    return counts, profile, first, second


def _exact_pair(left, right, q):
    a, f, first_a, second_a = left
    b, g, first_b, second_b = right
    mean = sum(ax*gx for ax, gx in zip(a, g))
    first_projection = sum(ax*gx*gx for ax, gx in zip(a, g))
    second_projection = sum(bx*fx*fx for bx, fx in zip(b, f))
    second = q*(second_a+second_b)-2*first_a*first_b
    residual = q*q*second-q*(first_projection+second_projection)+mean*mean
    if residual < 0:
        raise ArithmeticError("negative exact edge residual variance")
    return mean, residual


def _diagnostics(mode, q, classes, active, pairs, design, reused, workspace):
    return {"method": mode, "q": q, "vertices": design.n,
            "edges": len(design.edges), "active_vertices": sum(bool(a) for a in design.adjacency),
            "reference_classes": len(set(classes)), "profile_evaluations": len(active),
            "pair_evaluations": len(pairs), "reused_graph_design": reused,
            "workspace_estimate_bytes": workspace["total_bytes"],
            "workspace_estimate_scope": workspace["scope"],
            "profile_alphabet_entries": q*len(active),
            "vertex_projections_streamed": True}


def enumerate_exact(reference, q, edges_or_design, *,
                    max_workspace_bytes=DEFAULT_WORKSPACE_BYTES):
    """Exact raw E[S] and Var(S) from O(q) integer alphabet prefix sums.

    Numerators have denominators q**2 and q**4, respectively.  Timing includes
    validation, profile/pair preprocessing, and graph-query accumulation.
    ``max_workspace_bytes=None`` explicitly disables the estimate-based guard.
    """
    started = perf_counter()
    classes, active, pairs, design, reused = _fixed_inputs(reference, q, edges_or_design)
    workspace = _workspace(q, len(classes), len(active), len(pairs), design, "exact")
    _budget(workspace, max_workspace_bytes)
    validated = perf_counter()
    profiles = {m: _exact_profile(m, q) for m in active}
    profiled = perf_counter()
    pair_moments = {key: _exact_pair(profiles[key[0]], profiles[key[1]], q)
                    for key in pairs}
    prepared = perf_counter()
    mean = residual_total = 0
    for i, j, c in design.edges:
        edge_mean, residual = pair_moments[_pair_key(classes[i], classes[j])]
        mean += c*edge_mean
        residual_total += c*c*residual
    vertex_total = 0
    for i, adj in enumerate(design.adjacency):
        if not adj:
            continue
        weighted = [0]*q
        center = 0
        for j, c in adj:
            profile = profiles[classes[j]][1]
            center += c*pair_moments[_pair_key(classes[i], classes[j])][0]
            for x in range(q):
                weighted[x] += c*profile[x]
        counts = profiles[classes[i]][0]
        contribution = q*sum(a*w*w for a, w in zip(counts, weighted))-center*center
        if contribution < 0:
            raise ArithmeticError("negative exact vertex projection variance")
        vertex_total += contribution
    finished = perf_counter()
    diagnostics = _diagnostics("integer alphabet enumeration", q, classes, active,
                               pairs, design, reused, workspace)
    diagnostics.update(validation_seconds=validated-started,
                       profile_seconds=profiled-validated,
                       pair_seconds=prepared-profiled,
                       preprocessing_seconds=prepared-started,
                       query_seconds=finished-prepared,
                       total_seconds=finished-started,
                       vertex_variance_numerator=vertex_total,
                       residual_variance_numerator=residual_total,
                       arithmetic="exact arbitrary-precision integers")
    return ExactAlphabetMoments(q, mean, vertex_total+residual_total, diagnostics)


def _float_profile(p, x, np):
    first = float(np.dot(p, x))
    second = float(np.dot(p, x*x))
    cdf = np.cumsum(p)
    prefix_first = np.cumsum(p*x)
    gamma = x*(2*cdf-1)+first-2*prefix_first
    return p, gamma, first, second


def _float_pair(left, right, np, diagnostics, tolerance_factor):
    p, gamma_p, first_p, second_p = left
    r, gamma_r, first_r, second_r = right
    left_mean = float(np.dot(p, gamma_r))
    right_mean = float(np.dot(r, gamma_p))
    mean = (left_mean+right_mean)/2
    diagnostics["max_pair_mean_asymmetry"] = max(
        diagnostics["max_pair_mean_asymmetry"], abs(left_mean-right_mean))
    centered = gamma_r-mean
    projection_left = float(np.dot(p, centered*centered))
    centered = gamma_p-mean
    projection_right = float(np.dot(r, centered*centered))
    second = math.fsum((second_p, second_r, -2*first_p*first_r))
    residual = math.fsum((second, -mean*mean, -projection_left, -projection_right))
    # This explicit scale follows the terms that enter the subtraction.  The
    # residual identity has genuine cancellation, even though vertex terms are
    # centered. This is a prespecified diagnostic tolerance, not a proved bound
    # for all prefix/dot-product roundoff. Such errors can depend on q.
    scale = math.fsum((abs(second_p), abs(second_r), abs(2*first_p*first_r),
                       mean*mean, projection_left, projection_right))
    tolerance = tolerance_factor*np.finfo(np.float64).eps*scale
    diagnostics["max_residual_roundoff_tolerance"] = max(
        diagnostics["max_residual_roundoff_tolerance"], tolerance)
    if not all(map(math.isfinite, (mean, residual, tolerance))):
        raise FloatingPointError("nonfinite float64 pair moment")
    if abs(residual) <= tolerance:
        diagnostics["cancellation_sensitive_pairs"] += 1
    if residual < 0:
        diagnostics["minimum_unclipped_residual"] = min(
            diagnostics["minimum_unclipped_residual"], residual)
        if residual < -tolerance:
            raise FloatingPointError(
                f"negative edge residual {residual} exceeds explicit roundoff "
                f"tolerance {tolerance}")
        diagnostics["negative_residual_clipped_count"] += 1
        diagnostics["total_residual_clipping_magnitude"] += -residual
        residual = 0.0
    return mean, residual


def _floating_diagnostics(tolerance_factor):
    if isinstance(tolerance_factor, bool) or not isinstance(tolerance_factor, (int, float)):
        raise TypeError("residual_tolerance_factor must be a finite nonnegative number")
    if not math.isfinite(tolerance_factor) or tolerance_factor < 0:
        raise ValueError("residual_tolerance_factor must be finite and nonnegative")
    return {"arithmetic": "NumPy float64 approximation to exact finite-sum identities",
            "residual_tolerance_factor": tolerance_factor,
            "max_pair_mean_asymmetry": 0.0,
            "max_residual_roundoff_tolerance": 0.0,
            "cancellation_sensitive_pairs": 0,
            "negative_residual_clipped_count": 0,
            "minimum_unclipped_residual": 0.0,
            "total_residual_clipping_magnitude": 0.0,
            "weights_above_exact_float_integer_range": 0}


def _float_query(classes, profiles, pair_moments, design, q, np, diagnostics):
    def float_weight(c):
        try:
            value = float(c)
        except OverflowError as exc:
            raise FloatingPointError("edge weight exceeds float64 range") from exc
        if not math.isfinite(value):
            raise FloatingPointError("nonfinite float64 edge weight")
        return value

    diagnostics["weights_above_exact_float_integer_range"] = sum(
        abs(c) > 2**53 for _, _, c in design.edges)
    # Generators avoid allocating another edge-sized contribution list.
    mean = math.fsum(float_weight(c)*pair_moments[_pair_key(classes[i], classes[j])][0]
                     for i, j, c in design.edges)
    residual_total = math.fsum(float_weight(c)**2*pair_moments[_pair_key(classes[i], classes[j])][1]
                              for i, j, c in design.edges)
    vertex_terms = []
    for i, adj in enumerate(design.adjacency):
        if not adj:
            continue
        centered_sum = np.zeros(q, dtype=np.float64)
        for j, c in adj:
            edge_mean = pair_moments[_pair_key(classes[i], classes[j])][0]
            centered_sum += float_weight(c)*(profiles[classes[j]][1]-edge_mean)
        vertex_terms.append(float(np.dot(profiles[classes[i]][0], centered_sum*centered_sum)))
    vertex_total = math.fsum(vertex_terms)
    variance = math.fsum((vertex_total, residual_total))
    if not math.isfinite(mean) or not math.isfinite(variance):
        raise FloatingPointError("nonfinite float64 graph moment")
    diagnostics.update(vertex_variance=vertex_total, residual_variance=residual_total)
    return mean, variance


def enumerate_float(reference, q, edges_or_design, *,
                    max_workspace_bytes=DEFAULT_WORKSPACE_BYTES,
                    residual_tolerance_factor=128.0):
    """Float64 alphabet baseline with explicit residual-roundoff diagnostics.

    Vertex projections are accumulated in centered form. Negative edge residual
    variances are rounded to zero only within ``factor * eps * term_scale``;
    the count/magnitude are reported, and larger negative values raise an error.
    Positive residuals, including tiny cancellation-sensitive values, are kept.
    """
    import numpy as np
    started = perf_counter()
    classes, active, pairs, design, reused = _fixed_inputs(reference, q, edges_or_design)
    workspace = _workspace(q, len(classes), len(active), len(pairs), design, "float64")
    _budget(workspace, max_workspace_bytes)
    diagnostics = _floating_diagnostics(residual_tolerance_factor)
    validated = perf_counter()
    profiles = {}
    if active:
        x = np.arange(q, dtype=np.float64)
        for m in active:
            # Directly enumerate U and aggregate the q possible distances.
            distances = np.abs(np.arange(q, dtype=np.int64)-m)
            p = np.bincount(distances, minlength=q).astype(np.float64)/q
            profiles[m] = _float_profile(p, x, np)
        del distances, p, x
    profiled = perf_counter()
    pair_moments = {key: _float_pair(profiles[key[0]], profiles[key[1]], np,
                                    diagnostics, residual_tolerance_factor)
                    for key in pairs}
    prepared = perf_counter()
    mean, variance = _float_query(classes, profiles, pair_moments, design, q, np, diagnostics)
    finished = perf_counter()
    diagnostics.update(_diagnostics("float64 alphabet enumeration", q, classes,
                                    active, pairs, design, reused, workspace))
    diagnostics.update(validation_seconds=validated-started,
                       profile_seconds=profiled-validated,
                       pair_seconds=prepared-profiled,
                       preprocessing_seconds=prepared-started,
                       query_seconds=finished-prepared, total_seconds=finished-started)
    return FloatMoments(mean, variance, diagnostics)


def moments_independent(pmf, edges_or_design, *,
                        normalization_tolerance=1e-12,
                        max_workspace_bytes=DEFAULT_WORKSPACE_BYTES,
                        residual_tolerance_factor=128.0):
    """Moments of sum c_ij*|D_i-D_j| for independent position-specific PMFs.

    ``pmf[i][x]`` is P(D_i=x), with a nonempty common finite support range(q).
    Entries must be finite and nonnegative. Row sums must differ from one by
    at most ``normalization_tolerance``; accepted rows are explicitly divided
    by their sums, and the largest adjustment is reported. No arbitrary input
    measure is silently converted into a probability distribution.

    The input is converted to float64. Rational inputs consequently lose exact
    arithmetic. Profiles are built only for vertices incident to nonzero edges.
    Storage is O(n*q+e), including the float64 PMF copy, and arithmetic cost is
    O(q*(n+e)) after topology validation. Identical rows are not deduplicated.
    """
    import numpy as np
    started = perf_counter()
    if isinstance(normalization_tolerance, bool) or not isinstance(normalization_tolerance, (int, float)):
        raise TypeError("normalization_tolerance must be a finite nonnegative number")
    if not math.isfinite(normalization_tolerance) or normalization_tolerance < 0:
        raise ValueError("normalization_tolerance must be finite and nonnegative")
    # Determine shape before a potentially large dtype conversion or copy.
    if hasattr(pmf, "shape"):
        if len(pmf.shape) != 2:
            raise ValueError("pmf must be a two-dimensional array")
        n, q = map(int, pmf.shape)
    else:
        n = len(pmf)
        if not n:
            raise ValueError("empty PMF input must supply a two-dimensional shape")
        q = len(pmf[0])
        if any(len(row) != q for row in pmf):
            raise ValueError("PMF rows must have equal length")
    if q < 1:
        raise ValueError("the PMF support must be nonempty")
    design, reused = _graph(n, edges_or_design)
    active = tuple(i for i, adj in enumerate(design.adjacency) if adj)
    pairs = tuple((i, j) for i, j, _ in design.edges)
    workspace = _workspace(q, n, len(active), len(pairs), design, "float64")
    # _workspace counted two arrays per active profile. Here the PMF matrix is
    # copied once for every vertex and each active profile stores a row view.
    workspace["profile_bytes"] = 8*q*(n+len(active))
    workspace["total_bytes"] = (workspace["profile_bytes"] + workspace["scratch_bytes"]
                                + workspace["metadata_bytes"])
    _budget(workspace, max_workspace_bytes)
    probabilities = np.array(pmf, dtype=np.float64, copy=True)
    if probabilities.shape != (n, q):
        raise ValueError("pmf must be a rectangular two-dimensional array")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("PMF entries must be finite and nonnegative")
    row_sums = probabilities.sum(axis=1)
    maximum_adjustment = float(np.max(np.abs(row_sums-1))) if n else 0.0
    if np.any(row_sums <= 0) or maximum_adjustment > normalization_tolerance:
        raise ValueError("PMF rows must sum to one within normalization_tolerance")
    probabilities /= row_sums[:, None]
    diagnostics = _floating_diagnostics(residual_tolerance_factor)
    validated = perf_counter()
    x = np.arange(q, dtype=np.float64) if active else None
    profiles = {i: _float_profile(probabilities[i], x, np) for i in active}
    del x
    profiled = perf_counter()
    pair_moments = {key: _float_pair(profiles[key[0]], profiles[key[1]], np,
                                    diagnostics, residual_tolerance_factor)
                    for key in pairs}
    prepared = perf_counter()
    mean, variance = _float_query(tuple(range(n)), profiles, pair_moments, design, q, np, diagnostics)
    finished = perf_counter()
    diagnostics.update(_diagnostics("independent PMF float64 enumeration", q, tuple(range(n)),
                                    active, pairs, design, reused, workspace))
    diagnostics.update(normalization_tolerance=normalization_tolerance,
                       maximum_input_row_sum_adjustment=maximum_adjustment,
                       pmf_rows_explicitly_normalized=True,
                       validation_seconds=validated-started,
                       profile_seconds=profiled-validated,
                       pair_seconds=prepared-profiled,
                       preprocessing_seconds=prepared-started,
                       query_seconds=finished-prepared, total_seconds=finished-started)
    return FloatMoments(mean, variance, diagnostics)


def _rational_profile(row):
    first = sum((p*x for x, p in enumerate(row)), Fraction())
    second = sum((p*x*x for x, p in enumerate(row)), Fraction())
    cdf = prefix_first = Fraction()
    gamma = []
    for x, p in enumerate(row):
        cdf += p
        prefix_first += x*p
        gamma.append(x*(2*cdf-1)+first-2*prefix_first)
    return row, gamma, first, second


def _rational_pair(left, right):
    p, gamma_p, first_p, second_p = left
    r, gamma_r, first_r, second_r = right
    mean = sum((px*gx for px, gx in zip(p, gamma_r)), Fraction())
    projection_left = sum((px*(gx-mean)**2 for px, gx in zip(p, gamma_r)), Fraction())
    projection_right = sum((rx*(gx-mean)**2 for rx, gx in zip(r, gamma_p)), Fraction())
    second = second_p+second_r-2*first_p*first_r
    residual = second-mean*mean-projection_left-projection_right
    if residual < 0:
        raise ArithmeticError("negative rational edge residual variance")
    return mean, residual


def moments_independent_exact(pmf, edges_or_design, *,
                              max_workspace_bytes=DEFAULT_WORKSPACE_BYTES):
    """Exact moments for independent rational-valued PMFs on range(q).

    Entries must be Python integers or ``fractions.Fraction`` objects, finite,
    nonnegative, and sum to exactly one in every row. Floats are deliberately
    rejected, and rows are never renormalized. ``RationalMoments`` supplies
    Fraction-valued ``mean`` and ``variance`` plus explicit normalization.

    Prefix profiles and streamed, centered vertex projections require
    O(q*(n+e)) rational arithmetic operations after graph validation. The
    workspace estimate accounts for a common denominator obtained from the
    input rational laws; growth of integer bit lengths affects both space and
    runtime. No alphabet-pair kernel matrix is formed.
    """
    started = perf_counter()
    if hasattr(pmf, "shape"):
        if len(pmf.shape) != 2:
            raise ValueError("pmf must be a two-dimensional array")
        n, q = map(int, pmf.shape)
    else:
        n = len(pmf)
        if not n:
            raise ValueError("empty PMF input must supply a two-dimensional shape")
        q = len(pmf[0])
        if any(len(row) != q for row in pmf):
            raise ValueError("PMF rows must have equal length")
    if q < 1:
        raise ValueError("the PMF support must be nonempty")
    design, reused = _graph(n, edges_or_design)
    active = tuple(i for i, adj in enumerate(design.adjacency) if adj)
    active_set = set(active)
    pairs = tuple((i, j) for i, j, _ in design.edges)
    workspace = _workspace(q, n, len(active), len(pairs), design, "exact")
    # Reject clearly oversized inputs before even making a rational row copy.
    workspace["profile_bytes"] = 16*q*n + 64*q*len(active)
    workspace["total_bytes"] = (workspace["profile_bytes"]+workspace["scratch_bytes"]
                                +workspace["metadata_bytes"])
    _budget(workspace, max_workspace_bytes)
    denominator = 1
    for i, row in enumerate(pmf):
        total = Fraction()
        for p in row:
            if isinstance(p, bool) or not isinstance(p, (int, Fraction)):
                raise TypeError("exact PMF entries must be integers or Fraction objects")
            if p < 0:
                raise ValueError("PMF entries must be nonnegative")
            total += p
            if i in active_set and isinstance(p, Fraction):
                denominator = math.lcm(denominator, p.denominator)
        if total != 1:
            raise ValueError("exact PMF rows must sum to exactly one")
    # A global common denominator D bounds intermediate centered-square
    # denominators by D**5. This intentionally conservative bit-size allowance
    # is an estimate of additional storage, not an operating-system cap.
    denominator_bits = denominator.bit_length()
    qbits = q.bit_length()
    weight_bits = max(1, design.absolute_weight_sum.bit_length())
    fraction_header = sys.getsizeof(Fraction())
    profile_scalar = fraction_header+2*_int_bytes(denominator_bits+qbits+2)
    moment_scalar = fraction_header+2*_int_bytes(5*denominator_bits+2*qbits+2*weight_bits+8)
    workspace["profile_bytes"] = (8*q*n + 8*q*len(active)
                                  +q*len(active)*(2*profile_scalar))
    workspace["scratch_bytes"] = q*(16+2*moment_scalar) if active else 0
    workspace["metadata_bytes"] += len(pairs)*2*moment_scalar
    workspace["total_bytes"] = sum(workspace[key] for key in
                                    ("profile_bytes", "scratch_bytes", "metadata_bytes"))
    _budget(workspace, max_workspace_bytes)
    validated = perf_counter()
    # Only active row copies are needed; all rows were nevertheless validated.
    profiles = {i: _rational_profile(tuple(Fraction(p) for p in pmf[i])) for i in active}
    profiled = perf_counter()
    pair_moments = {key: _rational_pair(profiles[key[0]], profiles[key[1]])
                    for key in pairs}
    prepared = perf_counter()
    mean = sum((c*pair_moments[(i, j)][0] for i, j, c in design.edges), Fraction())
    residual_total = sum((c*c*pair_moments[(i, j)][1] for i, j, c in design.edges), Fraction())
    vertex_total = Fraction()
    for i in active:
        centered = [Fraction()]*q
        for j, c in design.adjacency[i]:
            edge_mean = pair_moments[_pair_key(i, j)][0]
            gamma = profiles[j][1]
            for x in range(q):
                centered[x] += c*(gamma[x]-edge_mean)
        vertex_total += sum((p*g*g for p, g in zip(profiles[i][0], centered)), Fraction())
    finished = perf_counter()
    diagnostics = _diagnostics("independent rational PMF enumeration", q, tuple(range(n)),
                               active, pairs, design, reused, workspace)
    diagnostics.update(arithmetic="exact arbitrary-precision rational numbers",
                       pmf_rows_explicitly_normalized=False,
                       input_common_denominator_bits=denominator_bits,
                       vertex_variance=vertex_total,
                       residual_variance=residual_total,
                       validation_seconds=validated-started,
                       profile_seconds=profiled-validated,
                       pair_seconds=prepared-profiled,
                       preprocessing_seconds=prepared-started,
                       query_seconds=finished-prepared, total_seconds=finished-started)
    return RationalMoments(mean, vertex_total+residual_total, diagnostics)
