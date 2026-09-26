"""Multi-criteria decision methods: TOPSIS and PROMETHEE II."""
from __future__ import annotations

import numpy as np
import pandas as pd

PREF_FUNCS = {
    "usual": "Usual (0/1)",
    "u-shape": "U-shape (q)",
    "v-shape": "V-shape (p)",
    "level": "Level (q, p)",
    "linear": "Linear / V-shape με αδιαφορία (q, p)",
    "gaussian": "Gaussian (s)",
}


def _weights(w) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    w = np.clip(w, 0, None)
    return w / w.sum() if w.sum() > 0 else np.full(len(w), 1 / len(w))


def topsis(X: pd.DataFrame, weights, benefit) -> pd.DataFrame:
    """Classic TOPSIS with vector normalisation.

    benefit: list[bool] (True = μεγαλύτερο καλύτερο).
    Returns D+, D-, closeness C* and rank.
    """
    M = X.to_numpy(dtype=float)
    w = _weights(weights)
    norm = np.sqrt((M ** 2).sum(axis=0))
    norm[norm == 0] = 1
    V = M / norm * w
    ben = np.asarray(benefit, dtype=bool)
    ideal = np.where(ben, V.max(axis=0), V.min(axis=0))
    anti = np.where(ben, V.min(axis=0), V.max(axis=0))
    dp = np.sqrt(((V - ideal) ** 2).sum(axis=1))
    dm = np.sqrt(((V - anti) ** 2).sum(axis=1))
    denom = dp + dm
    c = np.divide(dm, denom, out=np.full_like(dm, 0.5), where=denom > 0)
    out = pd.DataFrame({"D+": dp, "D-": dm, "TOPSIS C*": c}, index=X.index)
    out["Κατάταξη TOPSIS"] = out["TOPSIS C*"].rank(ascending=False, method="min").astype(int)
    return out


def _pref(d: np.ndarray, kind: str, q: float, p: float, s: float) -> np.ndarray:
    """Preference degree P(d) for difference d (already oriented so larger = better)."""
    P = np.zeros_like(d, dtype=float)
    pos = d > 0
    if kind == "usual":
        P[pos] = 1
    elif kind == "u-shape":
        P[d > q] = 1
    elif kind == "v-shape":
        p = max(p, 1e-12)
        P = np.clip(d / p, 0, 1)
    elif kind == "level":
        P[(d > q) & (d <= p)] = 0.5
        P[d > p] = 1
    elif kind == "linear":
        p = max(p, q + 1e-12)
        P = np.clip((d - q) / (p - q), 0, 1)
    elif kind == "gaussian":
        s = max(s, 1e-12)
        P[pos] = 1 - np.exp(-(d[pos] ** 2) / (2 * s ** 2))
    return P


def promethee(X: pd.DataFrame, weights, benefit, funcs=None) -> pd.DataFrame:
    """PROMETHEE II.

    funcs: list of dicts {kind, q, p, s}; q/p/s = None -> automatic
           (q = 0, p = std of criterion, s = std/2... see below).
    Returns Φ+, Φ-, Φ (net flow) and rank.
    """
    M = X.to_numpy(dtype=float)
    n, k = M.shape
    w = _weights(weights)
    ben = np.asarray(benefit, dtype=bool)
    funcs = funcs or [{"kind": "linear"}] * k
    pi = np.zeros((n, n))
    for j in range(k):
        col = M[:, j] if ben[j] else -M[:, j]
        f = funcs[j] or {}
        std = float(np.std(col)) or 1.0
        rng = float(col.max() - col.min()) or 1.0
        kind = f.get("kind") or "linear"
        q = f.get("q")
        p = f.get("p")
        s = f.get("s")
        q = 0.0 if q is None or (isinstance(q, float) and np.isnan(q)) else float(q)
        if p is None or (isinstance(p, float) and np.isnan(p)):
            p = max(std, q + 1e-9) if kind != "level" else max(rng / 2, q + 1e-9)
        s = std if s is None or (isinstance(s, float) and np.isnan(s)) else float(s)
        d = col[:, None] - col[None, :]
        pi += w[j] * _pref(d, kind, q, float(p), s)
    np.fill_diagonal(pi, 0)
    denom = max(n - 1, 1)
    phi_p = pi.sum(axis=1) / denom
    phi_m = pi.sum(axis=0) / denom
    out = pd.DataFrame({"Φ+": phi_p, "Φ-": phi_m, "Φ (net)": phi_p - phi_m}, index=X.index)
    out["Κατάταξη PROMETHEE"] = out["Φ (net)"].rank(ascending=False, method="min").astype(int)
    return out


def spearman(r1, r2) -> float:
    a = pd.Series(r1).rank()
    b = pd.Series(r2).rank()
    return float(a.corr(b, method="pearson"))
