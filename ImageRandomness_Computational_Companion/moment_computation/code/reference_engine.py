"""Exact fixed-reference pair moments without alphabet enumeration.

Model: U_i are independent uniform integers in range(q), D_i=abs(v_i-U_i),
S=sum_{(i,j,c) in edges} c*abs(D_i-D_j). All graph weights are integers.
The algorithm returns raw S moments. No distribution or quantile is computed.

Only Python's standard library is required. Polynomial coefficients and all
accumulators are integers; Fraction is used only for the final output.
"""
from dataclasses import dataclass
from fractions import Fraction


def _integer(x, name, minimum=None):
    if not isinstance(x, int) or isinstance(x, bool):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and x < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return x


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
    cuts = tuple(sorted({0, 1, m+1, upper+1, q}))
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
    cuts = sorted(set(target.cuts) | set(neighbor.cuts))
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
    events = {}
    for j, weight in neighbors:
        profile = profiles[j]
        for k in range(3):
            active[k] += weight * profile.initial[k]
        for x, delta in profile.events:
            update = events.setdefault(x, [0, 0, 0])
            for k in range(3):
                update[k] += weight * delta[k]
            counts['vertex_breakpoint_events'] += 1
    cuts = sorted(set(events) | set(target.cuts))
    total = 0
    for lo, end in zip(cuts, cuts[1:]):
        if lo in events:
            change = events[lo]
            for k in range(3):
                active[k] += change[k]
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
        divisor = Fraction(divisor)
        if divisor <= 0:
            raise ValueError('normalization divisor must be positive')
        return self.mean/divisor, self.variance/(divisor*divisor)


def moments_graph(reference, q, edges):
    """Compute exact raw S moments for a finite simple integer-weighted graph.

    Input length is the vertex count; isolated vertices and empty graphs are
    allowed. Signed weights are allowed. Self-loops and repeated unordered
    edges are rejected. Zero-weight edges are checked and then discarded.
    Structural diagnostic counters are not CPU-instruction counts.
    """
    _integer(q, 'q', 2)
    reference = tuple(reference)
    n = len(reference)
    for v in reference:
        _integer(v, 'reference intensity', 0)
        if v >= q:
            raise ValueError('reference intensity must be smaller than q')
    cleaned, seen = [], set()
    for edge in edges:
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
        key = (min(i,j), max(i,j))
        if key in seen:
            raise ValueError('repeated unordered edge')
        seen.add(key)
        if c:
            cleaned.append((key[0],key[1],c))
    cache = {}
    profiles = []
    classes = []
    for v in reference:
        cls = min(v,q-1-v)
        if cls not in cache:
            cache[cls] = _profile(v,q)
        profiles.append(cache[cls])
        classes.append(cls)
    diagnostics = dict(vertices=n, edges=len(cleaned),
                       distinct_reference_classes=len(cache),
                       distinct_edge_law_pairs=0, projection_integrations=0,
                       vertex_intervals=0, vertex_breakpoint_events=0,
                       alphabet_values_enumerated=0)
    adjacent = [[] for _ in range(n)]
    means_at_vertex = [0]*n
    pair_cache = {}
    mean_num = variance_num = 0
    q2 = q*q
    for i,j,weight in cleaned:
        key = (min(classes[i],classes[j]), max(classes[i],classes[j]))
        if key not in pair_cache:
            pi, pj = profiles[i], profiles[j]
            first, second_i = _projection_sums(pi,pj,diagnostics)
            reverse, second_j = _projection_sums(pj,pi,diagnostics)
            if first != reverse:
                raise ArithmeticError('inconsistent pair mean')
            h2_num = q*(pi.second_sum+pj.second_sum)-2*pi.first_sum*pj.first_sum
            residual_num = q2*h2_num-q*(second_i+second_j)+first*first
            if residual_num < 0:
                raise ArithmeticError('negative exact edge residual variance')
            pair_cache[key] = (first,residual_num)
        mu_num,residual_num = pair_cache[key]
        mean_num += weight*mu_num
        variance_num += weight*weight*residual_num
        means_at_vertex[i] += weight*mu_num
        means_at_vertex[j] += weight*mu_num
        adjacent[i].append((j,weight))
        adjacent[j].append((i,weight))
    diagnostics['distinct_edge_law_pairs'] = len(pair_cache)
    for i,neighbors in enumerate(adjacent):
        raw_second = _vertex_square_sum(profiles[i],neighbors,profiles,diagnostics)
        contribution = q*raw_second-means_at_vertex[i]*means_at_vertex[i]
        if contribution < 0:
            raise ArithmeticError('negative exact vertex variance')
        variance_num += contribution
    diagnostics['mean_numerator_bits'] = abs(mean_num).bit_length()
    diagnostics['variance_numerator_bits'] = variance_num.bit_length()
    return ExactMoments(q,mean_num,variance_num,diagnostics)


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


def moments_image(reference, q, rows, columns, height=2, width=2):
    """Compute raw complete-pair window-sum moments on an image grid."""
    edges = rectangular_edges(rows,columns,height,width)
    reference = tuple(reference)
    if len(reference) != rows*columns:
        raise ValueError('reference length does not match image dimensions')
    return moments_graph(reference,q,edges)


if __name__ == '__main__':
    result = moments_image([0,1,2,1],3,2,2)
    print('Raw mean:',result.mean)
    print('Raw variance:',result.variance)
    print('Diagnostics:',result.diagnostics)
