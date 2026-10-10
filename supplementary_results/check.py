"""Check saved supplementary exports against authoritative archived data.

Default operation reads files and prints JSON; --output regenerates a new folder.
It does not compile figures, regenerate simulations or modify archived inputs.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

import generate


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=2e-12, abs_tol=2e-14)


def digest(contents):
    return hashlib.sha256(contents).hexdigest()


def report_text(report):
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def check_rate(count, size, rate, lower, upper):
    count, size = int(count), int(size)
    require(size == 5000 and 0 <= count <= size, "Unexpected evaluation count")
    p = count / size
    require(close(p, rate), "Count/rate mismatch")
    z = NormalDist().inv_cdf(0.975)
    denominator = 1 + z*z/size
    center = (p + z*z/(2*size)) / denominator
    half = z * math.sqrt(p*(1-p)/size + z*z/(4*size*size)) / denominator
    require(close(lower, center-half) and close(upper, center+half), "Wilson interval mismatch")


def verify(archive_root, output):
    archive_root, output = Path(archive_root), Path(output)
    expected = generate.build_outputs(archive_root)
    for name, contents in expected.items():
        require((output/name).is_file(), "Missing export: " + name)
        require((output/name).read_bytes() == contents, "Export differs from archived data: " + name)

    repeated = generate.read_rows(output/"repeated_image_calibration.csv")
    fixed = generate.read_rows(output/"fixed_reference_calibration.csv")
    uaci = generate.read_rows(output/"uaci_calibration.csv")
    natural = generate.read_rows(output/"natural_reference_moments.csv")
    require(len(repeated) == 20 and len(fixed) == 20 and len(uaci) == 18 and len(natural) == 6, "Incomplete design coverage")
    require(len({(r['repeats'], r['alpha'], r['method']) for r in repeated}) == 20, "Repeated calibration duplicates")
    require({r['alpha'] for r in repeated} == {'0.01', '0.05'}, "Missing nominal level")
    require({r['design'] for r in fixed} == set(generate.FIXED), "Missing fixed design")
    require({r['alpha'] for r in fixed} == {'0.05'}, "Fixed figure level mismatch")
    for row in repeated + fixed:
        check_rate(row['rejection_count'], row['evaluation_size'], row['rejection_rate'], row['wilson95_lower'], row['wilson95_upper'])
        for a,b in [('rejection_rate','rejection_percent'),('wilson95_lower','wilson95_lower_percent'),('wilson95_upper','wilson95_upper_percent')]:
            require(close(100*float(row[a]), row[b]), "Percent conversion mismatch")
        condition = 'realized null bank' if row['method'] == 'independent_null_rank' else 'fixed decision rule'
        require(row['interval_conditioning'] == condition, "Conditioning label mismatch")
    for row in uaci:
        check_rate(row['rejections'], row['n'], row['rate'], row['wilson_low'], row['wilson_high'])
        require(close(100*float(row['rate']), row['rejection_percent']), "UACI percent conversion mismatch")
        require(float(row['error_minus_percentage_points']) >= 0 and float(row['error_plus_percentage_points']) >= 0, "Negative UACI error bar")
    # Validate exact natural moments independently of the decimal display source.
    exact = json.loads((archive_root/generate.INPUTS['natural_exact']).read_text())
    for row in natural:
        item = exact[row['image']+'_'+row['arrangement']]
        pairs = {
            'baci_mean': item['baci']['mean'], 'baci_variance': item['baci']['variance'],
            'baci_pair_occurrence_variance': item['baci']['pair_occurrence_variance'],
            'baci_independent_edge_variance': item['baci']['independent_edge_variance'],
            'uaci_mean': item['uaci']['mean'], 'uaci_fixed_variance': item['uaci']['variance'],
            'uaci_pooled_variance': item['uaci']['pooled_variance'], 'uaci_pooled_excess': item['uaci']['pooled_excess'],
            **item['ratios'],
        }
        for key, fraction in pairs.items():
            require(close(row[key], float(Fraction(fraction))), "Natural exact/decimal mismatch: " + key)
    for image in ('camera','clock','grass'):
        a,b = [r for r in natural if r['image'] == image]
        require({a['arrangement'],b['arrangement']} == {'original','permuted'}, "Missing natural arrangement")
        require(all(a[k] == b[k] for k in a if k.startswith('uaci_')), "Permutation changed UACI moments")
    geometry = generate.read_rows(output/'fixed_reference_geometry.csv')
    plot = generate.read_rows(output/'fixed_reference_geometry_plot.csv')
    require(len(geometry) == len(plot) == 2, "Incomplete geometry comparison")
    for a,b in zip(geometry,plot):
        require(close(float(b['sd_percent'])**2, a['variance_squared_percentage_points']), "Geometry variance unit mismatch")
        require(close(float(a['variance_squared_fraction'])*10000, a['variance_squared_percentage_points']), "Geometry squared-unit mismatch")

    return {
        'status': 'PASS', 'scope': 'Archived-output export and independent count, Wilson, exact-moment and unit checks; no simulation or figure compilation',
        'random_draws': 0,
        'counts': {'repeated_calibration_rows': 20, 'fixed_plot_points': 20, 'uaci_plot_points': 18, 'natural_moment_rows': 6, 'fixed_geometry_pairs': 2, 'recomputed_wilson_intervals': 58},
        'input_sha256': {name: digest((archive_root/name).read_bytes()) for name in generate.INPUTS.values()},
        'generated_sha256': {name: digest(contents) for name,contents in sorted(expected.items())},
        'qualification': 'Gaussian tail rules remain approximations; rank validity is marginal over bank and test draws; rank Wilson intervals condition on the reused bank.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root', type=Path, default=generate.ARCHIVE)
    parser.add_argument('--generated', type=Path, default=generate.MODULE/'generated')
    parser.add_argument('--output', type=Path, help='Optional absent directory outside the archive for regenerated exports and verification.json')
    args = parser.parse_args()
    report = verify(args.archive_root, args.generated)
    saved = args.generated/'verification.json'
    require(saved.is_file(), 'Missing saved verification.json')
    require(json.loads(saved.read_text()) == report, 'Saved verification record differs')
    if args.output:
        destination = args.output.resolve()
        require(not destination.exists(), 'Output directory already exists')
        require(not destination.is_relative_to(args.archive_root.resolve()), 'Check output must be outside the archive')
        destination.mkdir(parents=True)
        for name, contents in generate.build_outputs(args.archive_root).items():
            (destination/name).write_bytes(contents)
        regenerated_report = verify(args.archive_root, destination)
        require(regenerated_report == report, 'Regenerated results differ from supplied exports')
        (destination/'verification.json').write_text(report_text(regenerated_report), encoding='utf-8')
    print(report_text(report).strip())


if __name__ == '__main__':
    main()
