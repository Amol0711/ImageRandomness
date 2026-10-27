"""Exact fixed-reference pair moments without alphabet enumeration.

Model: U_i are independent uniform integers in range(q), D_i=abs(v_i-U_i),
S=sum_{(i,j,c) in edges} c*abs(D_i-D_j). All graph weights are integers.
The algorithm returns raw S moments. No distribution or quantile is computed.

Only Python's standard library is required. Polynomial coefficients and all
accumulators are integers; Fraction is used only for output and normalization.
The default core is deterministic and uncached. Optional cache=True uses
hash-based profile/pair caches with expected/amortized lookup costs. Both modes
use deterministic sorted-event vertex sweeps, with no alphabet enumeration.

For n vertices, e nonzero edges, and degrees d_i, validated-input arithmetic
cost is O(n+e+sum_i d_i*log(1+d_i)); input graph validation separately costs
O(n+e_in*log(1+e_in)). These are arithmetic counts, not bit complexity bounds.
"""
from dataclasses import dataclass
from fractions import Fraction


def _integer(x, name, minimum=None):
    if not isinstance(x, int) or isinstance(x, bool):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and x < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return x


def _unique_sorted(values):
    """Deterministic distinct sorted values without a hash table."""
    out = []
    for x in sorted(values):
        if not out or x != out[-1]:
            out.append(x)
    return tuple(out)


def _normalizer(value):
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise TypeError('normalizer must be an integer or Fraction')
    value = Fraction(value)
    if value <= 0:
        raise ValueError('normalizer must be positive')
    return value


@dataclass(frozen=True, init=False)
class GraphDesign:
    """Immutable validated graph; normalization is explicit and optional.

    Construction canonicalizes and sorts all edge triplets, detects duplicate
    unordered pairs (including zero-weight entries), and drops zero weights.
    Reusing a GraphDesign avoids repeated graph validation and adjacency work.
    The normalizer never changes the raw output of a moment calculation.
    """
    n: int
    edges: tuple
    adjacency: tuple
    normalizer: Fraction
    input_edge_count: int
    weight_sum: int
    absolute_weight_sum: int

    def __init__(self, n, edges, normalizer=1):
        _integer(n, 'vertex count', 0)
        normalizer = _normalizer(normalizer)
        canonical = []
        for edge in edges:
            edge = tuple(edge)
            if len(edge) != 3:
                raise ValueError('each edge must contain i, j, weight')
            i,j,c = edge
            _integer(i, 'edge endpoint', 0)
            _integer(j, 'edge endpoint', 0)
            _integer(c, 'edge weight')
            if i >= n or j >= n:
                raise ValueError('edge endpoint is outside the reference')
            if i == j:
                raise ValueError('self-loops are not allowed')
            canonical.append((min(i,j),max(i,j),c))
        canonical.sort()
        for first,second in zip(canonical,canonical[1:]):
            if first[:2] == second[:2]:
                raise ValueError('repeated unordered edge')
        cleaned = tuple((i,j,c) for i,j,c in canonical if c)
        adjacent = [[] for _ in range(n)]
        for i,j,c in cleaned:
            adjacent[i].append((j,c))
            adjacent[j].append((i,c))
        object.__setattr__(self,'n',n)
        object.__setattr__(self,'edges',cleaned)
        object.__setattr__(self,'adjacency',tuple(tuple(x) for x in adjacent))
        object.__setattr__(self,'normalizer',normalizer)
        object.__setattr__(self,'input_edge_count',len(canonical))
        object.__setattr__(self,'weight_sum',sum(c for _,_,c in cleaned))
        object.__setattr__(self,'absolute_weight_sum',sum(abs(c) for _,_,c in cleaned))


def prepare_graph(n, edges, normalizer=1):
    """Validate a topology once for reuse across alphabets or references."""
    return GraphDesign(n,edges,normalizer)


def _power_sums(n):
    """Sums of x**r, x=0,...,n, for r=0,...,4, with 0**0=1."""
    if n < 0:
        return (0, 0, 0, 0, 0)
    s1 = n * (n + 1) // 2
    return (n + 1, s1, n * (n + 1) * (2 * n + 1) // 6,
            s1 * s1,
            n * (n + 1) * (2 * n + 1) * (3 * n * n + 3 * n - 1) // 30)


def _integrate(poly, lo, hi):
    if hi < lo:
        return 0
    upper, lower = _power_sums(hi), _power_sums(lo - 1)
    return sum(c * (u - l) for c, u, l in zip(poly, upper, lower))


def _square(p):
    a, b, c = p
    return (a*a, 2*a*b, b*b + 2*a*c, 2*b*c, c*c)


def _fpoly(a, b, x):
    """Integer coefficients of sum_{d=a}^b |x-d| on x's piece."""
    if b < a:
        return (0, 0, 0)
    count = b-a+1
    if x < a:
        return (count*(a+b)//2, -count, 0)
    if x <= b:
        return ((a*a+b*b+b-a)//2, -a-b, 1)
    return (-count*(a+b)//2, count, 0)


@dataclass(frozen=True)
class _Profile:
    m: int
    upper: int
    cuts: tuple
    initial: tuple
    events: tuple
    first_sum: int
    second_sum: int

    def polynomial(self, x):
        p, r = _fpoly(0, self.upper, x), _fpoly(1, self.m, x)
        return tuple(a+b for a, b in zip(p, r))

    def mass_count(self, x):
        return int(0 <= x <= self.upper) + int(1 <= x <= self.m)


def _profile(v, q):
    m = min(v, q-1-v)
    upper = q-1-m
    cuts = _unique_sorted((0, 1, m+1, upper+1, q))
    def pol(x):
        return tuple(a+b for a,b in zip(_fpoly(0,upper,x), _fpoly(1,m,x)))
    initial = pol(0)
    previous = initial
    events = []
    for x in cuts[1:-1]:
        current = pol(x)
        delta = tuple(a-b for a,b in zip(current,previous))
        if any(delta):
            events.append((x, delta))
        previous = current
    return _Profile(m, upper, cuts, initial, tuple(events),
                    _power_sums(upper)[1]+_power_sums(m)[1],
                    _power_sums(upper)[2]+_power_sums(m)[2])


def _projection_sums(target, neighbor, counts):
    """Return sum a_target*f_neighbor and sum a_target*f_neighbor**2."""
    cuts = _unique_sorted(target.cuts + neighbor.cuts)
    first = second = 0
    for lo, end in zip(cuts, cuts[1:]):
        mass = target.mass_count(lo)
        if mass:
            p = neighbor.polynomial(lo)
            first += mass * _integrate(p, lo, end-1)
            second += mass * _integrate(_square(p), lo, end-1)
            counts['projection_integrations'] += 1
    return first, second


def _vertex_square_sum(target, neighbors, profiles, counts):
    """sum_x a_target(x)*(sum_j c_ij f_j(x))**2 by breakpoint sweep."""
    if not neighbors:
        return 0
    active = [0, 0, 0]
    # Include target-mass breakpoints as zero updates. Sorting and scanning
    # the raw list gives deterministic O(d log(1+d)) work at this vertex.
    events = [(x,0,0,0) for x in target.cuts]
    for j, weight in neighbors:
        profile = profiles[j]
        for k in range(3):
            active[k] += weight * profile.initial[k]
        for x, delta in profile.events:
            events.append((x,weight*delta[0],weight*delta[1],weight*delta[2]))
            counts['vertex_breakpoint_events'] += 1
    events.sort()
    counts['vertex_event_records'] += len(events)
    counts['max_vertex_event_records'] = max(counts['max_vertex_event_records'],len(events))
    total = 0
    pos = 0
    while pos < len(events):
        lo = events[pos][0]
        while pos < len(events) and events[pos][0] == lo:
            event = events[pos]
            active[0] += event[1]
            active[1] += event[2]
            active[2] += event[3]
            pos += 1
        if pos == len(events):
            break
        end = events[pos][0]
        mass = target.mass_count(lo)
        if mass:
            total += mass * _integrate(_square(active), lo, end-1)
            counts['vertex_intervals'] += 1
    return total


@dataclass(frozen=True)
class ExactMoments:
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
        """Return (E[S/divisor], Var[S/divisor]); divisor must be positive."""
        divisor = _normalizer(divisor)
        return self.mean/divisor, self.variance/(divisor*divisor)


def moments_graph(reference, q, edges, *, cache=False):
    """Compute exact raw S moments for a finite simple integer-weighted graph.

    Input length is the vertex count; isolated vertices and empty graphs are
    allowed. Signed weights are allowed. Self-loops and repeated unordered
    edges are rejected. Zero-weight edges are checked and then discarded.
    Pass an existing GraphDesign as edges to reuse its validated topology.
    cache=False is deterministic and uncached. cache=True adds optional
    expected/amortized hash-based profile and pair caches. Structural
    diagnostic counters are not CPU-instruction counts. Raw S moments are
    returned regardless of a GraphDesign's explicit normalizer.
    """
    _integer(q, 'q', 2)
    reference = tuple(reference)
    n = len(reference)
    if not isinstance(cache,bool):
        raise TypeError('cache must be a boolean')
    for v in reference:
        _integer(v, 'reference intensity', 0)
        if v >= q:
            raise ValueError('reference intensity must be smaller than q')
    prepared = isinstance(edges,GraphDesign)
    design = edges if prepared else prepare_graph(n,edges)
    if design.n != n:
        raise ValueError('reference length does not match graph vertex count')
    profile_cache = {} if cache else None
    profiles = []
    classes = []
    profile_evaluations = profile_cache_hits = 0
    for v in reference:
        cls = min(v,q-1-v)
        if cache and cls in profile_cache:
            profile = profile_cache[cls]
            profile_cache_hits += 1
        else:
            profile = _profile(v,q)
            profile_evaluations += 1
            if cache:
                profile_cache[cls] = profile
        profiles.append(profile)
        classes.append(cls)
    diagnostics = dict(vertices=n, edges=len(design.edges),
                       input_edges=design.input_edge_count,
                       reused_graph_design=prepared, cache_enabled=cache,
                       cache_model='expected_hash' if cache else 'deterministic_uncached',
                       profile_evaluations=profile_evaluations,
                       profile_cache_hits=profile_cache_hits,
                       distinct_reference_classes=len(profile_cache) if cache else None,
                       distinct_edge_law_pairs=None,
                       pair_evaluations=0, pair_cache_hits=0,
                       projection_integrations=0,
                       vertex_intervals=0, vertex_breakpoint_events=0,
                       vertex_event_records=0,max_vertex_event_records=0,
                       alphabet_values_enumerated=0)
    means_at_vertex = [0]*n
    pair_cache = {} if cache else None
    mean_num = variance_num = 0
    q2 = q*q
    for i,j,weight in design.edges:
        key = (min(classes[i],classes[j]), max(classes[i],classes[j]))
        if cache and key in pair_cache:
            mu_num,residual_num = pair_cache[key]
            diagnostics['pair_cache_hits'] += 1
        else:
            pi, pj = profiles[i], profiles[j]
            first, second_i = _projection_sums(pi,pj,diagnostics)
            reverse, second_j = _projection_sums(pj,pi,diagnostics)
            if first != reverse:
                raise ArithmeticError('inconsistent pair mean')
            h2_num = q*(pi.second_sum+pj.second_sum)-2*pi.first_sum*pj.first_sum
            residual_num = q2*h2_num-q*(second_i+second_j)+first*first
            if residual_num < 0:
                raise ArithmeticError('negative exact edge residual variance')
            mu_num = first
            diagnostics['pair_evaluations'] += 1
            if cache:
                pair_cache[key] = (mu_num,residual_num)
        mean_num += weight*mu_num
        variance_num += weight*weight*residual_num
        means_at_vertex[i] += weight*mu_num
        means_at_vertex[j] += weight*mu_num
    if cache:
        diagnostics['distinct_edge_law_pairs'] = len(pair_cache)
    for i,neighbors in enumerate(design.adjacency):
        raw_second = _vertex_square_sum(profiles[i],neighbors,profiles,diagnostics)
        contribution = q*raw_second-means_at_vertex[i]*means_at_vertex[i]
        if contribution < 0:
            raise ArithmeticError('negative exact vertex variance')
        variance_num += contribution
    diagnostics['mean_numerator_bits'] = abs(mean_num).bit_length()
    diagnostics['variance_numerator_bits'] = variance_num.bit_length()
    return ExactMoments(q,mean_num,variance_num,diagnostics)


def moments_prepared(reference, q, design, *, cache=False):
    """Explicit prepared-topology entry point; output is the raw S moment."""
    if not isinstance(design,GraphDesign):
        raise TypeError('design must be a GraphDesign')
    return moments_graph(reference,q,design,cache=cache)


def _axis_multiplicity(lo, hi, length, width):
    return max(0,min(lo,length-width)-max(hi-width+1,0)+1)


def rectangular_edges(rows, columns, height, width):
    """Distinct pairs and multiplicities for all complete-pair h-by-w windows.

    Enumeration uses local displacements, costing O(rows*columns*height*width)
    arithmetic operations. All windows must fit fully inside the image.
    """
    for value,name in ((rows,'rows'),(columns,'columns'),
                       (height,'height'),(width,'width')):
        _integer(value,name,1)
    if height > rows or width > columns:
        raise ValueError('window must fit inside image')
    if height*width < 2:
        raise ValueError('window must contain at least two pixels')
    edges = []
    for row in range(rows):
        for col in range(columns):
            i = row*columns+col
            for dr in range(height):
                other_row = row+dr
                if other_row >= rows:
                    break
                start = 1 if dr == 0 else 1-width
                for dc in range(start,width):
                    other_col = col+dc
                    if not 0 <= other_col < columns:
                        continue
                    multiplicity = (
                        _axis_multiplicity(row,other_row,rows,height) *
                        _axis_multiplicity(min(col,other_col),max(col,other_col),columns,width))
                    if multiplicity:
                        edges.append((i,other_row*columns+other_col,multiplicity))
    return edges


def prepare_image_graph(rows, columns, height=2, width=2, *, normalizer=1):
    """Create reusable rectangular topology with explicit chosen normalizer."""
    edges = rectangular_edges(rows,columns,height,width)
    return prepare_graph(rows*columns,edges,normalizer=normalizer)


def image_normalizer(rows, columns, height, width, q):
    """K*binom(height*width,2)*(q-1), for fractional complete-pair score."""
    _integer(q,'q',2)
    for value,name in ((rows,'rows'),(columns,'columns'),
                       (height,'height'),(width,'width')):
        _integer(value,name,1)
    if height > rows or width > columns or height*width < 2:
        raise ValueError('invalid image/window geometry')
    area = height*width
    return (rows-height+1)*(columns-width+1)*(area*(area-1)//2)*(q-1)


def moments_image(reference, q, rows, columns, height=2, width=2, *, cache=False):
    """Compute raw complete-pair window-sum moments on an image grid."""
    edges = rectangular_edges(rows,columns,height,width)
    reference = tuple(reference)
    if len(reference) != rows*columns:
        raise ValueError('reference length does not match image dimensions')
    return moments_graph(reference,q,edges,cache=cache)


if __name__ == '__main__':
    result = moments_image([0,1,2,1],3,2,2)
    print('Raw mean:',result.mean)
    print('Raw variance:',result.variance)
    print('Diagnostics:',result.diagnostics)
