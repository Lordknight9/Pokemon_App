"""Ansha's Donuts (Pokémon Legends: Z-A – Mega Dimension).

Data: berry flavours / level / calories from Serebii; star thresholds and multipliers from Game8;
flavour-budget model for Flavor Powers from RotomLabs' research (community data-mining).
The game rolls the actual powers at random among what the budget allows, so this module
tells you what is *possible* (and up to which level), not a guaranteed result.
"""
from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np

FLAVORS = ["Sweet", "Spicy", "Sour", "Bitter", "Fresh"]
FLAVOR_COLOR = {"Sweet": "#F28AB2", "Spicy": "#E4572E", "Sour": "#E8B923", "Bitter": "#4E9F50",
                "Fresh": "#3C9AD6"}

# name: sweet, spicy, sour, bitter, fresh, level boost, calories
_BERRIES = """Cheri,0,10,0,0,0,1,60
Chesto,0,0,0,0,10,1,60
Pecha,10,0,0,0,0,1,60
Rawst,0,0,0,10,0,1,60
Aspear,0,0,10,0,0,1,60
Oran,0,5,5,5,5,1,60
Persim,5,5,5,0,5,1,60
Lum,5,5,0,5,5,2,65
Sitrus,5,0,5,5,5,2,65
Pomeg,10,10,0,10,0,2,65
Kelpsy,0,0,10,10,10,2,65
Qualot,10,10,10,0,0,2,65
Hondew,0,10,0,10,10,2,65
Grepa,10,0,10,0,10,2,65
Tamato,0,15,0,0,10,2,65
Occa,10,15,0,0,0,3,70
Passho,0,0,0,10,15,3,70
Wacan,15,0,10,0,0,3,70
Rindo,0,10,0,15,0,3,70
Yache,0,0,15,0,10,3,70
Chople,0,15,0,10,0,3,70
Kebia,0,0,10,0,15,3,70
Shuca,15,10,0,0,0,3,70
Coba,0,0,0,15,10,3,70
Payapa,10,0,15,0,0,3,70
Tanga,0,20,10,0,0,3,70
Charti,0,10,0,0,20,3,70
Kasib,20,0,0,0,10,3,70
Haban,10,0,0,20,0,3,70
Colbur,0,0,20,10,0,3,70
Babiri,0,25,0,0,10,3,70
Chilan,10,0,0,0,25,3,70
Roseli,25,0,0,10,0,3,70
Hyper Cheri,0,40,0,0,0,5,80
Hyper Chesto,0,0,0,0,40,3,100
Hyper Pecha,40,0,0,0,0,2,100
Hyper Rawst,0,0,0,40,0,3,110
Hyper Aspear,0,0,40,0,0,4,90
Hyper Oran,10,20,15,15,0,6,90
Hyper Persim,0,15,15,10,20,4,110
Hyper Lum,20,15,10,0,15,3,110
Hyper Sitrus,15,10,0,20,15,4,120
Hyper Pomeg,30,35,0,0,5,7,140
Hyper Kelpsy,5,0,0,30,35,5,160
Hyper Qualot,35,0,30,5,0,4,160
Hyper Hondew,0,5,35,0,30,6,150
Hyper Grepa,0,60,25,0,5,8,140
Hyper Tamato,5,25,0,0,60,6,180
Hyper Occa,60,0,0,5,25,5,180
Hyper Passho,25,0,5,60,0,6,200
Hyper Wacan,0,5,60,25,0,7,160
Hyper Rindo,15,55,0,5,25,9,210
Hyper Yache,25,0,5,15,55,7,250
Hyper Chople,55,5,15,25,0,6,250
Hyper Kebia,0,15,25,55,5,7,270
Hyper Shuca,5,25,55,0,15,8,230
Hyper Coba,10,95,0,10,5,10,240
Hyper Payapa,5,0,10,10,95,8,300
Hyper Tanga,95,10,10,5,0,7,300
Hyper Charti,0,10,5,95,10,8,330
Hyper Kasib,10,5,95,0,10,9,270
Hyper Haban,85,0,0,0,65,8,370
Hyper Colbur,0,0,65,0,85,9,370
Hyper Babiri,0,0,65,85,0,9,400
Hyper Chilan,0,85,0,65,0,9,370
Hyper Roseli,0,65,85,0,0,10,340"""

BERRIES = {}
for _line in _BERRIES.splitlines():
    _n, *_v = _line.split(",")
    BERRIES[_n] = {"flavors": np.array(list(map(int, _v[:5]))), "level": int(_v[5]), "calories": int(_v[6]),
                   "hyper": _n.startswith("Hyper")}
NAMES = list(BERRIES)
FLAV_MAT = np.array([BERRIES[n]["flavors"] for n in NAMES])          # (B, 5)
LEVEL_VEC = np.array([BERRIES[n]["level"] for n in NAMES])
CAL_VEC = np.array([BERRIES[n]["calories"] for n in NAMES])
HYPER_VEC = np.array([BERRIES[n]["hyper"] for n in NAMES])

STAR_THRESHOLDS = [120, 240, 350, 700, 960]                         # flavour score for 1..5 stars
STAR_MULT = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]                          # by stars 0..5
FLAVOR_BUDGET = [(760, 9), (700, 8), (525, 7), (420, 6), (360, 5), (300, 4), (200, 3), (160, 2), (120, 1)]
TOTAL_BUDGET = [(960, 9), (800, 8), (700, 7), (600, 6), (500, 5), (420, 4), (350, 3), (240, 2), (120, 1)]

# flavour -> category -> powers
POWERS = {
    "Sweet": {"Sparkling": ["Sparkling Power (Shiny)"],
              "Size": ["Alpha Power", "Humungo Power", "Teensy Power"]},
    "Spicy": {"Move": ["Move Power (ένας τύπος)"],
              "Battle": ["Attack Power", "Sp. Atk Power", "Speed Power"]},
    "Sour": {"Item": ["Item Power: Berries", "Item Power: Candies", "Item Power: Treasure",
                      "Item Power: Poké Balls", "Item Power: Special", "Item Power: Coins"],
             "Distortion": ["Mega Power Charging", "Mega Power Conservation", "Big Haul Power"]},
    "Bitter": {"Resistance": ["Resistance Power (ένας τύπος)"],
               "Defense": ["Defense Power", "Sp. Def Power"]},
    "Fresh": {"Encounter": ["Encounter Power"], "Catch": ["Catching Power"]},
}
POWER_TO_CAT = {p: (f, c) for f, cats in POWERS.items() for c, ps in cats.items() for p in ps}
CAT_FLAVOR = {c: f for f, cats in POWERS.items() for c in cats}


def power_cost(category: str, level: int) -> int:
    return 3 if category == "Sparkling" else level


def _lookup(table, x):
    for th, v in table:
        if x >= th:
            return v
    return 0


def stars_of(score) -> np.ndarray:
    return np.searchsorted(STAR_THRESHOLDS, np.asarray(score), side="right")


def evaluate(recipe: dict) -> dict:
    """recipe {berry: count} -> flavours, stars, calories, level boost, budgets and possible powers."""
    fl = np.zeros(5, dtype=int)
    lvl = cal = 0
    for b, k in recipe.items():
        if k:
            fl += BERRIES[b]["flavors"] * k
            lvl += BERRIES[b]["level"] * k
            cal += BERRIES[b]["calories"] * k
    score = int(fl.sum())
    stars = int(stars_of(score))
    mult = STAR_MULT[stars]
    top = fl.max()
    rainbow = top > 0 and int((fl == top).sum()) > 1
    fbud = {f: _lookup(FLAVOR_BUDGET, v) + (1 if rainbow else 0) for f, v in zip(FLAVORS, fl)}
    fbud = {f: (b if fl[i] >= 120 or rainbow else 0) for i, (f, b) in enumerate(fbud.items())}
    total = _lookup(TOTAL_BUDGET, score)
    possible = {}
    for f, cats in POWERS.items():
        for c in cats:
            if c == "Sparkling":
                ok = fbud[f] >= 3 and total >= 3
                possible[c] = 3 if ok else 0
            else:
                possible[c] = int(min(3, fbud[f], total))
    order = [FLAVORS[i] for i in np.argsort(-fl, kind="stable") if fl[i] > 0]
    return {"flavors": dict(zip(FLAVORS, fl.tolist())), "score": score, "stars": stars, "mult": mult,
            "calories": int(cal * mult), "level": int(lvl * mult), "base_calories": cal, "base_level": lvl,
            "rainbow": bool(rainbow), "flavor_budget": fbud, "total_budget": total, "possible": possible,
            "order": order, "n": int(sum(recipe.values()))}


def feasible(ev: dict, targets: dict) -> bool:
    """targets {category: min_level}. Checks per-flavour and total budget for all targets together."""
    if len(targets) > 3:
        return False
    need_f, need_t = {}, 0
    for c, lv in targets.items():
        if ev["possible"].get(c, 0) < (1 if c == "Sparkling" else lv):
            return False
        cost = power_cost(c, lv)
        need_f[CAT_FLAVOR[c]] = need_f.get(CAT_FLAVOR[c], 0) + cost
        need_t += cost
    return need_t <= ev["total_budget"] and all(need_f[f] <= ev["flavor_budget"][f] for f in need_f)


# ------------------------------------------------------------------ search (vectorised)
def _combos(idx: list, n: int) -> np.ndarray:
    return np.array(list(combinations_with_replacement(idx, n)), dtype=np.int16)


def _vector_eval(C: np.ndarray):
    fl = FLAV_MAT[C].sum(axis=1)                       # (M, 5)
    score = fl.sum(axis=1)
    stars = stars_of(score)
    mult = np.array(STAR_MULT)[stars]
    top = fl.max(axis=1)
    rainbow = (fl == top[:, None]).sum(axis=1) > 1
    fb = np.zeros_like(fl)
    for th, v in reversed(FLAVOR_BUDGET):
        fb = np.where(fl >= th, v, fb)
    fb = fb + rainbow[:, None].astype(int) * (fl > 0)
    tb = np.zeros_like(score)
    for th, v in reversed(TOTAL_BUDGET):
        tb = np.where(score >= th, v, tb)
    return {"fl": fl, "score": score, "stars": stars, "mult": mult, "fb": fb, "tb": tb,
            "cal": (CAL_VEC[C].sum(axis=1) * mult).astype(int), "lvl": (LEVEL_VEC[C].sum(axis=1) * mult).astype(int),
            "hyper": HYPER_VEC[C].sum(axis=1)}


def _candidates(allowed: list, weights: np.ndarray, k: int) -> list:
    idx = [NAMES.index(b) for b in allowed]
    rel = FLAV_MAT[idx] @ weights + 0.15 * FLAV_MAT[idx].sum(axis=1)
    order = np.argsort(-rel)
    return [idx[i] for i in order[:k]]


def search(targets: dict, n: int = 8, min_stars: int = 0, exact_stars: int | None = None,
           allowed: list | None = None, dominant: str | None = None, prefer: str = "cheap",
           top: int = 10, k: int | None = None, main_flavor: bool = True) -> list[dict]:
    """Best recipes of n berries that make every target power possible.

    prefer: 'cheap' (fewest Hyper berries, then calories), 'calories', 'level' or 'stars'.
    """
    allowed = allowed or NAMES
    w = np.zeros(5)
    for c in targets:
        w[FLAVORS.index(CAT_FLAVOR[c])] += 1
    if dominant:
        w[FLAVORS.index(dominant)] += 2
    if not w.any():
        w[:] = 0.2
    k = k or {3: 30, 4: 24, 5: 18, 6: 15, 7: 13, 8: 11}.get(n, 11)
    cand = _candidates(allowed, w, k)
    if prefer == "cheap":  # also consider the cheapest regular berries for the target flavours
        reg = [b for b in allowed if not BERRIES[b]["hyper"]]
        if reg:
            cand = list(dict.fromkeys(cand + _candidates(reg, w, 4)))
    C = _combos(cand, n)
    V = _vector_eval(C)
    ok = V["stars"] >= min_stars
    if exact_stars is not None:
        ok &= V["stars"] == exact_stars
    if dominant:
        di = FLAVORS.index(dominant)
        ok &= V["fl"][:, di] == V["fl"].max(axis=1)
    tf = sorted({FLAVORS.index(CAT_FLAVOR[c]) for c in targets})
    if main_flavor and tf:  # the target flavours must be the donut's strongest flavours
        others = [i for i in range(5) if i not in tf]
        if others:
            ok &= V["fl"][:, tf].min(axis=1) >= V["fl"][:, others].max(axis=1)
    need_t = np.zeros(len(C), dtype=int)
    need_f = np.zeros((len(C), 5), dtype=int)
    for c, lv in targets.items():
        fi = FLAVORS.index(CAT_FLAVOR[c])
        cost = power_cost(c, lv)
        ok &= V["fb"][:, fi] >= cost
        need_f[:, fi] += cost
        need_t += cost
    ok &= need_t <= V["tb"]
    ok &= (need_f <= V["fb"]).all(axis=1)
    if len(targets) > 3:
        ok[:] = False
    idx = np.nonzero(ok)[0]
    if not len(idx):
        return []
    keys = {"cheap": (V["hyper"][idx], -V["cal"][idx]),
            "calories": (-V["cal"][idx], V["hyper"][idx]),
            "level": (-V["lvl"][idx], V["hyper"][idx]),
            "stars": (-V["stars"][idx], -V["score"][idx])}[prefer]
    order = idx[np.lexsort(keys[::-1])]
    out, seen = [], set()
    for i in order:
        rec = {}
        for b in C[i]:
            rec[NAMES[b]] = rec.get(NAMES[b], 0) + 1
        key = tuple(sorted(rec.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append({"recipe": rec, **evaluate(rec)})
        if len(out) >= top:
            break
    return out


def recipe_text(rec: dict) -> str:
    return " + ".join(f"{k}× {b}" for b, k in sorted(rec.items(), key=lambda kv: (-kv[1], kv[0])))
