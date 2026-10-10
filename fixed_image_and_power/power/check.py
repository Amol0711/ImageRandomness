"""Check the archived numerical experiment without rerunning the full simulation."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
from pathlib import Path
import tempfile

import numpy as np

import run

HERE = Path(__file__).resolve().parent


def main():
    cfg = json.loads((HERE / "design.json").read_text())
    dest = HERE / "results"
    checks = 0
    hashes = json.loads((dest / "sha256.json").read_text())
    for name, expected in hashes.items():
        assert hashlib.sha256((dest / name).read_bytes()).hexdigest() == expected, name
        checks += 1
    # Independent direct enumeration of the six comparisons in every window.
    arrays = np.asarray(list(itertools.product(range(2), repeat=9)), dtype=np.int16).reshape(-1,3,3)
    computed = run.statistics(arrays)
    for i, d in enumerate(arrays):
        brute = sum(abs(int(x)-int(y)) for r in range(2) for c in range(2)
                    for x,y in itertools.combinations(d[r:r+2,c:c+2].ravel(), 2))
        assert computed[i].tolist() == [brute, int(d.sum()), int(np.count_nonzero(d))]
        checks += 1
    # Every held-out item in a fixed distinct lexicographic sample has a unique rank.
    # All numerical scores tie here, exercising the randomized boundary rule.
    testcfg = {"q": 2, "rows": 2, "columns": 2, "alpha": [1/3]}
    raw = np.ones((6,3), dtype=np.int64)
    tie = np.arange(18, dtype=float).reshape(6,3)/20
    rejected = np.zeros(3, dtype=int)
    for hold in range(6):
        keep = np.arange(6) != hold
        cutoff = run.thresholds(raw[keep], tie[keep], testcfg)
        rejected += run.decisions(raw[hold:hold+1], tie[hold:hold+1], testcfg, cutoff, 1/3)[0]
    assert rejected.tolist() == [2,2,2]
    checks += 1
    bank = np.load(dest / "raw/calibration.npz")
    cutoff = run.thresholds(bank["statistics"], bank["tie_uniforms"], cfg)
    assert cutoff == json.loads((dest / "thresholds.json").read_text())
    checks += 1
    bank_scores = run.scores(bank["statistics"], cfg)
    for j in range(3):
        pairs = np.rec.fromarrays([bank_scores[:,j], bank["tie_uniforms"][:,j]])
        assert len(np.unique(pairs)) == cfg["calibration_size"]
        checks += 1
    marginals, paired = [], []
    sample_list = [("null_validation", "null", 0.)]
    for family, key in (("identity", "identity_probabilities"),
                        ("horizontal_copy", "horizontal_copy_probabilities")):
        sample_list += [(f"{family}_{level:02d}", family, strength)
                       for level,strength in enumerate(cfg[key], 1)]
    for fname, family, strength in sample_list:
        z = np.load(dest / "raw" / (fname+".npz"))
        raw, ties = z["statistics"], z["tie_uniforms"]
        assert raw.dtype == np.int64 and ties.shape == raw.shape
        assert np.all((ties >= 0) & (ties < 1))
        mr, pr = run.records(raw, ties, cfg, cutoff, family, strength)
        marginals += mr
        paired += pr
        checks += 2
    with tempfile.TemporaryDirectory() as tmp:
        for name, rows in (("power_summary.csv",marginals), ("paired_differences.csv",paired)):
            path = Path(tmp)/name
            run.write_csv(path, rows)
            assert path.read_bytes() == (dest/name).read_bytes(), name
            checks += 1
    # Mean diagnostics check the construction against separate analytic expectations.
    q,m,n = cfg["q"],cfg["rows"],cfg["columns"]
    mu = 4*(q**4-1)/(15*q**3)
    ed = (q*q-1)/(3*q)
    diagnostics = []
    for fname, family, strength in sample_list:
        raw = np.load(dest / "raw" / (fname+".npz"))["statistics"]
        scale = np.asarray([6*(m-1)*(n-1)*(q-1), m*n*(q-1), m*n])
        values = raw/scale
        expected = np.asarray([mu/(q-1), ed/(q-1), (q-1)/q])
        if family == "identity":
            e = strength
            expected = np.asarray([((1-e)**2*mu+2*e*(1-e)*ed)/(q-1),
                                   (1-e)*ed/(q-1), (1-e)*(q-1)/q])
        elif family == "horizontal_copy":
            expected[0] *= 1 - n*strength/(6*(n-1))
        se = values.std(axis=0,ddof=1)/np.sqrt(len(values))
        zscore = (values.mean(axis=0)-expected)/se
        assert np.all(np.abs(zscore) < 5), (family,strength,zscore)
        diagnostics.append({"family":family,"strength":strength,
                            "mean_standardized_errors": zscore.tolist()})
        checks += 3
    print(json.dumps({"status":"passed", "checks":checks,
                      "largest_absolute_mean_standardized_error":max(
                          abs(v) for row in diagnostics for v in row["mean_standardized_errors"])}, indent=2))


if __name__ == "__main__":
    main()
