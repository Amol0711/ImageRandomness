"""Reproduce the registered paired image-statistic power experiment."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy.stats import beta, binom, norm

HERE = Path(__file__).resolve().parent
NAMES = ("BACI", "UACI", "NPCR")


def statistics(d):
    """Return integer BACI numerator, UACI numerator, and NPCR numerator."""
    a, b = d[:, :-1, :-1], d[:, :-1, 1:]
    c, e = d[:, 1:, :-1], d[:, 1:, 1:]
    s = sum(np.abs(x - y).sum(axis=(1, 2), dtype=np.int64)
            for x, y in ((a, b), (a, c), (a, e), (b, c), (b, e), (c, e)))
    return np.column_stack((s, d.sum(axis=(1, 2), dtype=np.int64),
                            np.count_nonzero(d, axis=(1, 2))))


def scores(raw, cfg):
    """Use integer-scaled exact centers, avoiding floating-point ties."""
    q, m, n = cfg["q"], cfg["rows"], cfg["columns"]
    pixels, windows = m * n, (m - 1) * (n - 1)
    return np.column_stack((
        np.abs((15 * q**3) * raw[:, 0] - 4 * (q**4 - 1) * 6 * windows),
        np.abs((3 * q) * raw[:, 1] - pixels * (q*q - 1)),
        np.abs(q * raw[:, 2] - pixels * (q - 1)),
    ))


def rng_for(cfg, stage, family, level):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        [cfg["root_seed"], stage, family, level])))


def sample(cfg, count, family, level, strength):
    image_rng = rng_for(cfg, 1, family, level)
    tie_rng = rng_for(cfg, 2, family, level)
    q, m, n = cfg["q"], cfg["rows"], cfg["columns"]
    out = np.empty((count, 3), dtype=np.int64)
    for lo in range(0, count, cfg["batch_size"]):
        hi = min(lo + cfg["batch_size"], count)
        u = image_rng.integers(q, size=(hi-lo, m, n), dtype=np.int16)
        v = image_rng.integers(q, size=(hi-lo, m, n), dtype=np.int16)
        if family == 2:
            mask = image_rng.random(u.shape) < strength
            v[mask] = u[mask]
        elif family == 3:
            # Independent Bernoulli selection for each disjoint horizontal pair.
            # Copy both image pixels so each coordinate retains its null law.
            mask = image_rng.random((hi-lo, m, n//2)) < strength
            np.copyto(u[:, :, 1::2], u[:, :, ::2], where=mask)
            np.copyto(v[:, :, 1::2], v[:, :, ::2], where=mask)
        out[lo:hi] = statistics(np.abs(u-v))
    return out, tie_rng.random((count, 3))


def thresholds(raw, ties, cfg):
    sc = scores(raw, cfg)
    result = {}
    for alpha in cfg["alpha"]:
        k = round(alpha * (len(raw) + 1))
        assert math.isclose(k, alpha * (len(raw) + 1), abs_tol=1e-10)
        result[str(alpha)] = []
        for j, name in enumerate(NAMES):
            order = np.lexsort((ties[:, j], sc[:, j]))
            idx = order[len(raw)-k]
            result[str(alpha)].append({"statistic": name,
                "score": int(sc[idx, j]), "tie_uniform": float(ties[idx, j]),
                "rank_from_largest": k})
    return result


def decisions(raw, ties, cfg, cutoff, alpha):
    sc = scores(raw, cfg)
    t = cutoff[str(alpha)]
    ss = np.asarray([r["score"] for r in t], dtype=np.int64)
    uu = np.asarray([r["tie_uniform"] for r in t])
    return (sc > ss) | ((sc == ss) & (ties > uu))


def wilson(k, n, confidence=.95):
    z = norm.ppf((1 + confidence) / 2)
    p = k/n
    den = 1 + z*z/n
    mid = (p + z*z/(2*n))/den
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return max(0., mid-half), min(1., mid+half)


def clopper_pearson(k, n, confidence=.975):
    a = (1-confidence)/2
    return (0. if k == 0 else float(beta.ppf(a, k, n-k+1)),
            1. if k == n else float(beta.ppf(1-a, k+1, n-k)))


def records(raw, ties, cfg, cutoff, family, strength):
    marg, paired = [], []
    count = len(raw)
    for alpha in cfg["alpha"]:
        reject = decisions(raw, ties, cfg, cutoff, alpha)
        for j, name in enumerate(NAMES):
            k = int(reject[:, j].sum())
            lo, hi = wilson(k, count, cfg["confidence_level"])
            marg.append(dict(family=family, strength=strength, alpha=alpha,
                statistic=name, replications=count, rejections=k,
                probability=k/count, ci_low=lo, ci_high=hi))
        for j, name in ((1, "UACI"), (2, "NPCR")):
            n10 = int((reject[:, 0] & ~reject[:, j]).sum())
            n01 = int((~reject[:, 0] & reject[:, j]).sum())
            lo10, hi10 = clopper_pearson(n10, count)
            lo01, hi01 = clopper_pearson(n01, count)
            paired.append(dict(family=family, strength=strength, alpha=alpha,
                comparison="BACI-minus-" + name, replications=count,
                baci_only=n10, comparator_only=n01,
                difference=(n10-n01)/count,
                ci_low=lo10-hi01, ci_high=hi10-lo01))
    return marg, paired


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--reuse-raw", action="store_true",
                        help="Rebuild summaries from existing complete raw archives")
    args = parser.parse_args()
    cfg = json.loads((HERE / "design.json").read_text())
    dest = args.output
    rawdir = dest / "raw"
    rawdir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    def obtain(name, count, family, level, strength):
        path = rawdir / (name + ".npz")
        if args.reuse_raw:
            z = np.load(path)
            raw, ties = z["statistics"], z["tie_uniforms"]
            assert raw.shape == ties.shape == (count, 3)
        else:
            raw, ties = sample(cfg, count, family, level, strength)
            np.savez_compressed(path, statistics=raw, tie_uniforms=ties)
        return raw, ties

    bank, btie = obtain("calibration", cfg["calibration_size"], 0, 0, 0.)
    cutoff = thresholds(bank, btie, cfg)
    (dest / "thresholds.json").write_text(json.dumps(cutoff, indent=2)+"\n")
    print("Calibration complete", flush=True)
    marginals, paired = [], []
    raw, tie = obtain("null_validation", cfg["null_validation_size"], 1, 0, 0.)
    mr, pr = records(raw, tie, cfg, cutoff, "null", 0.)
    marginals += mr
    paired += pr
    print("Independent null validation complete", flush=True)
    for family, index, key in (("identity", 2, "identity_probabilities"),
                                ("horizontal_copy", 3, "horizontal_copy_probabilities")):
        for level, strength in enumerate(cfg[key], 1):
            raw, tie = obtain(f"{family}_{level:02d}",
                cfg["alternative_replications"], index, level, strength)
            mr, pr = records(raw, tie, cfg, cutoff, family, strength)
            marginals += mr
            paired += pr
            print(f"{family} {strength:g} complete", flush=True)
    write_csv(dest / "power_summary.csv", marginals)
    write_csv(dest / "paired_differences.csv", paired)
    # Exact conditional NPCR size checks the discrete boundary convention.
    q, pixels = cfg["q"], cfg["rows"]*cfg["columns"]
    x = np.arange(pixels+1, dtype=np.int64)
    ps = binom.pmf(x, pixels, (q-1)/q)
    sc = np.abs(q*x-pixels*(q-1))
    exact_npcr = []
    for alpha in cfg["alpha"]:
        t = cutoff[str(alpha)][2]
        size = float(ps[sc > t["score"]].sum() +
                     (1-t["tie_uniform"]) * ps[sc == t["score"]].sum())
        exact_npcr.append({"alpha": alpha, "conditional_null_probability": size})
    (dest / "npcr_conditional_size.json").write_text(json.dumps(exact_npcr, indent=2)+"\n")
    metadata = dict(design_sha256=hashlib.sha256((HERE/"design.json").read_bytes()).hexdigest(),
                    python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                    raw_reused=args.reuse_raw, elapsed_seconds=time.perf_counter()-started)
    (dest / "execution.json").write_text(json.dumps(metadata, indent=2)+"\n")
    hashes = {str(p.relative_to(dest)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(dest.rglob("*")) if p.is_file() and p.name != "sha256.json"}
    (dest / "sha256.json").write_text(json.dumps(hashes, indent=2)+"\n")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
