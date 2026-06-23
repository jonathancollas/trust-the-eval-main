"""Statistics primitives (INFRA-4). Pure functions, unit-tested."""
from __future__ import annotations
import math
import random as _random
from collections import Counter
from typing import Sequence


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def proportion_halfwidth(p: float, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 1.0
    return z * math.sqrt(max(p * (1 - p), 1e-12) / n)


def min_significant_gap(p: float, n: int, z: float = 1.96) -> float:
    """Smallest accuracy gap between two equal-n models that is not noise."""
    return z * math.sqrt(2.0) * math.sqrt(max(p * (1 - p), 1e-12) / n)


def two_proportion_pvalue(k1: int, n1: int, k2: int, n2: int) -> float:
    if n1 == 0 or n2 == 0:
        return 1.0
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(max(p * (1 - p) * (1 / n1 + 1 / n2), 1e-12))
    if se == 0:
        return 1.0
    z = (p1 - p2) / se
    return math.erfc(abs(z) / math.sqrt(2.0))  # two-sided


def cohens_kappa(a: Sequence, b: Sequence) -> float:
    if len(a) != len(b) or not a:
        return float("nan")
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    labels = set(ca) | set(cb)
    pe = sum((ca.get(l, 0) / n) * (cb.get(l, 0) / n) for l in labels)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def bootstrap_mean_ci(values: Sequence[float], iters: int = 2000,
                      seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    vals = list(values)
    if not vals:
        return (0.0, 0.0)
    rng = _random.Random(seed)
    n = len(vals)
    means = []
    for _ in range(iters):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((alpha / 2) * iters)]
    hi = means[min(iters - 1, int((1 - alpha / 2) * iters))]
    return (lo, hi)


def expected_calibration_error(confidences: Sequence[float],
                               corrects: Sequence[bool], bins: int = 10) -> float:
    if not confidences:
        return float("nan")
    n = len(confidences)
    edges = [i / bins for i in range(bins + 1)]
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        idx = [j for j, c in enumerate(confidences)
               if (c > lo or (i == 0 and c >= lo)) and c <= hi]
        if not idx:
            continue
        conf = sum(confidences[j] for j in idx) / len(idx)
        acc = sum(1 for j in idx if corrects[j]) / len(idx)
        ece += (len(idx) / n) * abs(acc - conf)
    return ece


def shannon_entropy(counts: Sequence[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


def gini(values: Sequence[float]) -> float:
    vals = sorted(v for v in values if v >= 0)
    n = len(vals)
    if n == 0 or sum(vals) == 0:
        return 0.0
    cum = sum((i + 1) * v for i, v in enumerate(vals))
    return (2 * cum) / (n * sum(vals)) - (n + 1) / n


def point_biserial(continuous: Sequence[float], binary: Sequence[int]) -> float:
    n = len(continuous)
    if n < 2:
        return float("nan")
    mean = sum(continuous) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in continuous) / n)
    if sd == 0:
        return 0.0
    ones = [continuous[i] for i in range(n) if binary[i] == 1]
    zeros = [continuous[i] for i in range(n) if binary[i] == 0]
    if not ones or not zeros:
        return 0.0
    m1, m0 = sum(ones) / len(ones), sum(zeros) / len(zeros)
    p, q = len(ones) / n, len(zeros) / n
    return (m1 - m0) / sd * math.sqrt(p * q)


def _shingles(text: str, k: int = 5) -> set:
    toks = text.split()
    if len(toks) < k:
        return {text} if text.strip() else set()
    return {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)}


def near_duplicate_pairs(texts: Sequence[str], k: int = 5, num_hashes: int = 48,
                         bands: int = 12, threshold: float = 0.8, seed: int = 0) -> int:
    """Count near-duplicate text pairs via MinHash + LSH banding (Jaccard >= threshold).
    Zero-dependency; replaces crude 8-token-prefix matching with shingle similarity."""
    import hashlib
    rng = _random.Random(seed)
    MOD = (1 << 61) - 1
    coeffs = [(rng.randrange(1, MOD), rng.randrange(0, MOD)) for _ in range(num_hashes)]
    def base_hash(s: str) -> int:
        return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")
    shingle_sets = [_shingles(t, k) for t in texts]
    sigs = []
    for sh in shingle_sets:
        if not sh:
            sigs.append(None); continue
        hs = [base_hash(s) for s in sh]
        sigs.append([min(((a * h + b) % MOD) for h in hs) for (a, b) in coeffs])
    rows = max(1, num_hashes // bands)
    buckets: dict = {}
    for i, sig in enumerate(sigs):
        if sig is None:
            continue
        for b in range(bands):
            key = (b, tuple(sig[b * rows:(b + 1) * rows]))
            buckets.setdefault(key, []).append(i)
    cand = set()
    for members in buckets.values():
        if len(members) > 1:
            for a in range(len(members)):
                for c in range(a + 1, len(members)):
                    cand.add((members[a], members[c]))
    pairs = 0
    for i, j in cand:
        si, sj = shingle_sets[i], shingle_sets[j]
        if si and sj and len(si & sj) / len(si | sj) >= threshold:
            pairs += 1
    return pairs


def expected_max_gap(m: int, p: float, n: int) -> float:
    """Expected (best - mean) accuracy across m equal-n configs under pure noise,
    to judge whether a 'best of m' lead is just multiple-comparisons luck."""
    import statistics as _stats
    if m < 2 or n <= 0:
        return 0.0
    se = math.sqrt(max(p * (1 - p), 1e-12) / n)
    q = (m - math.pi / 8) / (m - math.pi / 4 + 1)          # Blom approx of E[max]
    return _stats.NormalDist().inv_cdf(min(0.9999, q)) * se
