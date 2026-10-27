"""Exact mean responses to disjoint copying of iid difference pixels.

The null difference law is that of abs(U-V), where U and V are independent
uniform integers on range(q). Copying acts on this difference field. It is
also induced by copying both input-image values with the same activation.
Copying only one input image is a different alternative.

All finite moment and mean-shift outputs are Fraction values. The sensitivity
index converts its exact numerator and variance to floating point only for
the final square root. No power or alternative-variance formula is asserted.
"""
from fractions import Fraction
from functools import lru_cache
from math import sqrt


def _integer(value, name, lower=1):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f'{name} must be a Python integer')
    if value < lower:
        raise ValueError(f'{name} must be at least {lower}')
    return value


def _probability(theta):
    if isinstance(theta, bool) or not isinstance(theta, (int, Fraction)):
        raise TypeError('theta must be an integer or Fraction; convert decimal inputs explicitly')
    theta = Fraction(theta)
    if not 0 <= theta <= 1:
        raise ValueError('theta must lie in [0,1]')
    return theta


def copying_pairs(rows, columns, lag):
    """Ordered donor-recipient pairs in complete horizontal blocks of 2*lag.

    In each row and block, the first lag positions are donors and the second
    lag positions their recipients. An incomplete terminal block is untouched.
    Every index occurs in at most one pair. Row-major indexing is used.
    """
    _integer(rows, 'rows')
    _integer(columns, 'columns')
    _integer(lag, 'lag')
    return tuple((r*columns + start + offset,
                  r*columns + start + offset + lag)
                 for r in range(rows)
                 for start in range(0, columns-2*lag+1, 2*lag)
                 for offset in range(lag))


@lru_cache(maxsize=None)
def iid_triangular_constants(q):
    """Exact (mu,A,C) for h(D,D')=abs(D-D'), with iid triangular D.

    mu=E[h], A=Var(h), C=Cov(h(D,D'),h(D,D'')); all D's are iid.
    Integer probability multiplicities over denominator q**2 and integer
    prefix sums give O(q) arithmetic work, with no q by q distance matrix.
    Only a constant number of integer accumulators is stored.
    """
    _integer(q, 'q', 2)
    denominator = q*q
    first = q*(q*q-1)//3
    second = q*q*(q*q-1)//6
    prefix_count = prefix_first = 0
    mean_numerator = projection_numerator = 0
    for x in range(q):
        multiplicity = q if x == 0 else 2*(q-x)
        prefix_count += multiplicity
        prefix_first += x*multiplicity
        projection = x*(2*prefix_count-denominator)+first-2*prefix_first
        mean_numerator += multiplicity*projection
        projection_numerator += multiplicity*projection*projection
    mu = Fraction(mean_numerator, denominator**2)
    second_edge = Fraction(2*denominator*second-2*first*first, denominator**2)
    A = second_edge-mu*mu
    C = Fraction(projection_numerator, denominator**3)-mu*mu
    if A <= 0 or C < 0 or A-2*C < 0:
        raise ArithmeticError('invalid exact triangular-law constants')
    return mu, A, C


def iid_triangular_moments(design, q):
    """Raw exact null mean and variance on a validated integer-weight graph."""
    mu, A, C = iid_triangular_constants(q)
    degree = [0]*design.n
    edge_square_sum = 0
    weight_sum = 0
    for i, j, c in design.edges:
        degree[i] += c
        degree[j] += c
        edge_square_sum += c*c
        weight_sum += c
    return mu*weight_sum, (A-2*C)*edge_square_sum+C*sum(x*x for x in degree)


def exact_mean_shift(design, q, lag, theta, rows, columns):
    """Return E_null[S]-E_copy[S] exactly, for the specified copying model.

    The design must have rows*columns vertices and validated integer weights.
    theta must be a Fraction or integer. Signed graph weights are allowed;
    positive mean decrease follows when weights are nonnegative.
    """
    theta = _probability(theta)
    pairs = copying_pairs(rows, columns, lag)
    if design.n != rows*columns:
        raise ValueError('graph size does not match the supplied image geometry')
    pair_set = set(pairs)
    copied_weight = sum(c for i, j, c in design.edges
                        if (min(i, j), max(i, j)) in pair_set)
    mu, _, _ = iid_triangular_constants(q)
    return theta*mu*copied_weight


def _axis_pair_count(length, window, first, second):
    return max(0, min(first, second, length-window)
               -max(0, max(first, second)-window+1)+1)


def _geometry(rows, columns, height, width):
    for value, name in ((rows, 'rows'), (columns, 'columns'),
                        (height, 'height'), (width, 'width')):
        _integer(value, name)
    if height > rows or width > columns or height*width < 2:
        raise ValueError('window must fit and contain at least two pixels')


def rectangular_copy_weight(rows, columns, height, width, lag):
    """Exact sum of window multiplicities on the designated copying pairs."""
    _geometry(rows, columns, height, width)
    _integer(lag, 'lag')
    column_total = sum(_axis_pair_count(columns, width, start+offset,
                                       start+offset+lag)
                       for start in range(0, columns-2*lag+1, 2*lag)
                       for offset in range(lag))
    return height*(rows-height+1)*column_total


def axis_intersection_sums(length, window):
    """Ordered interval-intersection sums and squared-intersection sums."""
    _integer(length, 'length')
    _integer(window, 'window')
    if window > length:
        raise ValueError('window must fit inside the axis')
    count = length-window+1
    maximum_offset = min(window-1, count-1)
    i1 = count*window + 2*sum((count-j)*(window-j)
                             for j in range(1, maximum_offset+1))
    i2 = count*window*window + 2*sum((count-j)*(window-j)**2
                                    for j in range(1, maximum_offset+1))
    return i1, i2


def rectangular_null_moments(rows, columns, height, width, q):
    """Exact iid triangular raw moments from finite boundary invariants."""
    _geometry(rows, columns, height, width)
    mu, A, C = iid_triangular_constants(q)
    i1r, i2r = axis_intersection_sums(rows, height)
    i1c, i2c = axis_intersection_sums(columns, width)
    gamma2 = (i2r*i2c-i1r*i1c)//2
    degree2 = (height*width-1)**2*i1r*i1c
    window_count = (rows-height+1)*(columns-width+1)
    edge_occurrences = window_count*(height*width)*(height*width-1)//2
    return mu*edge_occurrences, (A-2*C)*gamma2+C*degree2


def finite_grid_sensitivity(rows, columns, windows, q, lag,
                            theta=Fraction(1)):
    """Return exact finite-grid ingredients and a numerical mean-shift index.

    windows is a sequence of (height,width) pairs. Results are not predicted
    rejection probabilities and impose no assertion about optimal power.
    The field 'standardized_mean_decrease' uses the NULL standard deviation.
    """
    theta = _probability(theta)
    mu, _, _ = iid_triangular_constants(q)
    output = []
    for height, width in windows:
        mean, variance = rectangular_null_moments(rows, columns, height, width, q)
        copied_weight = rectangular_copy_weight(rows, columns, height, width, lag)
        decrease = theta*mu*copied_weight
        if variance <= 0:
            raise ArithmeticError('the null graph variance must be positive')
        output.append({'rows': rows, 'columns': columns, 'height': height,
                       'width': width, 'q': q, 'lag': lag, 'theta': theta,
                       'copy_pair_count': rows*lag*(columns//(2*lag)),
                       'copied_pair_weight': copied_weight, 'null_mean': mean,
                       'null_variance': variance, 'mean_decrease': decrease,
                       'alternative_mean': mean-decrease,
                       'standardized_mean_decrease': float(decrease)/sqrt(float(variance))})
    return output


def partial_identity_means(q, epsilon, n, weight_sum):
    """Exact NPCR count, UACI sum, and pair-statistic means after zero replacement."""
    _integer(q, 'q', 2)
    _integer(n, 'n', 0)
    if isinstance(weight_sum, bool) or not isinstance(weight_sum, int):
        raise TypeError('weight_sum must be an integer')
    epsilon = _probability(epsilon)
    mu, _, _ = iid_triangular_constants(q)
    first = Fraction(q*q-1, 3*q)
    retain = 1-epsilon
    return {'NPCR_count': retain*n*Fraction(q-1, q),
            'UACI_sum': retain*n*first,
            'S': weight_sum*(retain*retain*mu+2*epsilon*retain*first)}


def additive_copy_moments(q, theta, rows, columns, lag):
    """Exact additive null and copy moments, with unchanged means and altered variances."""
    _integer(q, 'q', 2)
    theta = _probability(theta)
    pairs = copying_pairs(rows, columns, lag)
    n = rows*columns
    factor = n+2*theta*len(pairs)
    first = Fraction(q*q-1, 3*q)
    variance = Fraction((q*q-1)*(q*q+2), 18*q*q)
    p = Fraction(q-1, q)
    return {'NPCR_count': {'mean': n*p, 'null_variance': n*p*(1-p),
                           'copy_variance': factor*p*(1-p)},
            'UACI_sum': {'mean': n*first, 'null_variance': n*variance,
                         'copy_variance': factor*variance}}


def _json_safe(value):
    if isinstance(value, Fraction):
        return {'numerator': str(value.numerator), 'denominator': str(value.denominator)}
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def run_sensitivity(protocol_path, output_directory, authorized_protocol_sha256):
    """Evaluate the frozen finite-grid analytic sensitivity matrix."""
    import csv
    import hashlib
    import json
    from pathlib import Path
    protocol_path = Path(protocol_path)
    digest = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    if digest != authorized_protocol_sha256:
        raise ValueError('protocol SHA256 differs from the authorized frozen design')
    protocol = json.loads(protocol_path.read_text())
    parameters = protocol['analysis']['sensitivity_parameters']
    theta = Fraction(parameters['theta'])
    rows = []
    maxima = []
    for M, N in parameters['grids']:
        for q in parameters['alphabets']:
            for lag in parameters['lags']:
                group = finite_grid_sensitivity(M, N,
                    [(w, w) for w in parameters['widths']], q, lag, theta)
                ratios = [x['mean_decrease']**2/x['null_variance'] for x in group]
                maximum = max(ratios)
                selected = [x['width'] for x, value in zip(group, ratios) if value == maximum]
                maxima.append({'rows': M, 'columns': N, 'q': q, 'lag': lag,
                               'theta': theta, 'maximizing_widths_in_declared_set': selected,
                               'maximum_squared_index': maximum,
                               'maximum_index': sqrt(float(maximum))})
                rows.extend(group)
    payload = {'status': 'completed', 'scope': 'Exact finite-grid mean sensitivity, not predicted power',
               'protocol_sha256': digest,
               'code_sha256': {'alternative_analysis.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
               'parameters': parameters, 'rows': rows, 'maxima': maxima,
               'row_count': len(rows), 'maximizer_comparison': 'Exact rational squared indices; all exact ties retained'}
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory/'sensitivity.json').write_text(json.dumps(_json_safe(payload), indent=2)+'\n')
    with (output_directory/'sensitivity.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return payload


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=root/'results/protocol.json')
    parser.add_argument('--output-directory', type=Path, default=root/'results')
    parser.add_argument('--authorized-protocol-sha256', required=True)
    args = parser.parse_args()
    result = run_sensitivity(args.protocol, args.output_directory,
                             args.authorized_protocol_sha256)
    print(f"Exact sensitivity completed with {result['row_count']} rows and {len(result['maxima'])} declared-width searches.")
