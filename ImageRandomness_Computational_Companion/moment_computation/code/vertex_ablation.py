"""Exact vertex-kernel comparison for incident-projection aggregation.

Both kernels compute the same integer quantity

    sum_x a_0(x) * (sum_j c_0j f_j(x))**2,

where a_0 is the folded-law multiplicity and f_j is q times the conditional
absolute-distance profile. They receive the same prepared profiles and weights.
The sweep calls the production event-sweep kernel. The pairwise kernel expands
the square into diagonal and unordered cross products and integrates each
piecewise-quartic product separately. Neither kernel enumerates the alphabet.

Preparation is explicit and excluded from kernel timing. This is an ablation
of one vertex calculation, not an alternative complete moment algorithm. The
returned counters describe structural work, not CPU instructions. The separate
verification function directly enumerates small alphabets and checks signed
weights, boundary references, empty stars and exact cancellation.
"""
from __future__ import annotations

from dataclasses import dataclass
import json

from moment_engine import _integrate, _profile, _unique_sorted, _vertex_square_sum


@dataclass(frozen=True)
class VertexCase:
    q: int
    reference: tuple
    weights: tuple
    profiles: tuple
    neighbors: tuple

    @property
    def degree(self):
        return len(self.weights)

    @property
    def target(self):
        return self.profiles[0]


@dataclass(frozen=True)
class VertexResult:
    raw_square_sum: int
    diagnostics: dict


def prepare_vertex_case(reference, q, weights):
    """Prepare one central vertex and its weighted incident profiles."""
    if isinstance(q, bool) or not isinstance(q, int) or q < 2:
        raise ValueError("q must be an integer at least two")
    reference, weights = tuple(reference), tuple(weights)
    if len(reference) != len(weights) + 1:
        raise ValueError("one central reference and one per weight are required")
    if any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < q
           for v in reference):
        raise ValueError("reference values must be integers in range(q)")
    if any(isinstance(c, bool) or not isinstance(c, int) or c == 0
           for c in weights):
        raise ValueError("incident weights must be nonzero integers")
    return VertexCase(q, reference, weights,
                      tuple(_profile(v, q) for v in reference),
                      tuple((j, c) for j, c in enumerate(weights, start=1)))


def make_star_case(degree, q):
    """Deterministic unit-weight star with spread folded reference radii.

    At q=65536 and degrees 8 through 256, all leaf folded radii are distinct.
    The same preparation is supplied to both kernels.
    """
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 0:
        raise ValueError("degree must be a nonnegative integer")
    if isinstance(q, bool) or not isinstance(q, int) or q < 2:
        raise ValueError("q must be an integer at least two")
    leaves = tuple(j * (q - 1) // (2 * (degree + 1))
                   for j in range(1, degree + 1))
    return prepare_vertex_case((q // 3,) + leaves, q, (1,) * degree)


def sweep_vertex(case):
    """Production sorted-event kernel, with fresh local diagnostic counters."""
    counts = {"vertex_breakpoint_events": 0, "vertex_event_records": 0,
              "max_vertex_event_records": 0, "vertex_intervals": 0}
    value = _vertex_square_sum(case.target, case.neighbors,
                               case.profiles, counts)
    return VertexResult(value, {
        "kernel": "sorted_event_sweep", "degree": case.degree,
        "pair_products": 0,
        "interval_integrations": counts["vertex_intervals"],
        "raw_events": counts["vertex_event_records"],
        "neighbor_coefficient_events": counts["vertex_breakpoint_events"],
        "raw_square_sum_bits": abs(value).bit_length(),
    })


def _multiply_quadratics(left, right):
    a, b, c = left
    d, e, f = right
    return (a*d, a*e+b*d, a*f+b*e+c*d, b*f+c*e, c*f)


def pairwise_vertex(case):
    """Expand each diagonal and unordered incident-profile product exactly.

    Each product merges only the target and the two participating profiles'
    cuts, a bounded number independent of degree and alphabet size. The number
    of integrated products is d(d+1)/2, including d diagonal products.
    """
    value = intervals = products = 0
    neighbors = case.neighbors
    target = case.target
    for position, (j, weight_j) in enumerate(neighbors):
        left = case.profiles[j]
        for k, weight_k in neighbors[position:]:
            right = case.profiles[k]
            coefficient = weight_j * weight_k * (1 if j == k else 2)
            cuts = _unique_sorted(target.cuts + left.cuts + right.cuts)
            product_sum = 0
            for lo, end in zip(cuts, cuts[1:]):
                mass = target.mass_count(lo)
                if mass:
                    product = _multiply_quadratics(left.polynomial(lo),
                                                  right.polynomial(lo))
                    product_sum += mass * _integrate(product, lo, end - 1)
                    intervals += 1
            value += coefficient * product_sum
            products += 1
    return VertexResult(value, {
        "kernel": "incident_pair_expansion", "degree": case.degree,
        "pair_products": products, "interval_integrations": intervals,
        "raw_events": 0, "neighbor_coefficient_events": 0,
        "raw_square_sum_bits": abs(value).bit_length(),
    })


def _direct_small_sum(case):
    """Independent finite-alphabet calculation from the defining intensities."""
    total = 0
    for central_uniform in range(case.q):
        x = abs(case.reference[0] - central_uniform)
        combined = sum(
            weight * sum(abs(x - abs(case.reference[j] - u))
                         for u in range(case.q))
            for j, weight in enumerate(case.weights, start=1))
        total += combined * combined
    return total


def verify():
    """Check both kernels against direct small-alphabet enumeration."""
    checked = zero_cases = 0
    signed_pattern = (1, -2, 3, -1, 2)
    for q in range(2, 9):
        for degree in range(1, 9):
            leaves = tuple((3*j + degree) % q for j in range(1, degree + 1))
            weights = tuple(signed_pattern[j % len(signed_pattern)]
                            for j in range(degree))
            for center in (0, q // 2, q - 1):
                case = prepare_vertex_case((center,) + leaves, q, weights)
                direct = _direct_small_sum(case)
                sweep, pairwise = sweep_vertex(case), pairwise_vertex(case)
                if sweep.raw_square_sum != direct or pairwise.raw_square_sum != direct:
                    raise ArithmeticError("vertex kernels differ from direct enumeration")
                if pairwise.diagnostics["pair_products"] != degree*(degree+1)//2:
                    raise ArithmeticError("incident-product counter differs")
                if sweep.diagnostics["raw_events"] > 3*degree+5:
                    raise ArithmeticError("event bound exceeded")
                checked += 1
        for case in (prepare_vertex_case((q//2,), q, ()),
                     prepare_vertex_case((q//2, 0, 0), q, (1, -1))):
            if any(value != 0 for value in (_direct_small_sum(case),
                                            sweep_vertex(case).raw_square_sum,
                                            pairwise_vertex(case).raw_square_sum)):
                raise ArithmeticError("empty or cancellation case is nonzero")
            checked += 1
            zero_cases += 1
    return {"status": "passed", "direct_enumeration_cases": checked,
            "zero_or_cancellation_cases": zero_cases,
            "alphabet_values": list(range(2, 9)),
            "nonempty_degrees": list(range(1, 9)),
            "signed_weights": True,
            "scope": "Vertex kernels only; no timing measurements"}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
