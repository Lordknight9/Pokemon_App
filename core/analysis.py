"""Build criteria matrices for single-Pokémon ranking, pair synergy and 4-of-6 teams."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .damage import Battler, Field, Move, calc
from .stats import STATS, calc_all, spread_for
from .typechart import TYPES, defensive_profile, super_effective_targets


# ------------------------------------------------------------------ profiles
def build_profile(p: dict, level: int, spread: str, learnset: list, move_index: dict,
                  coverage_mode: str = "learnset", use_abilities: bool = True) -> dict:
    evs, nature, ivs = spread_for(p["base"], spread)
    stats = calc_all(p["base"], level, ivs=ivs, evs=evs, nature=nature)
    dmg_types = set(p["types"])
    if coverage_mode == "learnset":
        for m in learnset:
            info = move_index.get(m)
            if info and info.get("category") in ("physical", "special") and info.get("type"):
                dmg_types.add(info["type"])
    ability = p["abilities"][0] if (use_abilities and p.get("abilities")) else None
    return {
        "key": p["name"], "display": p["display"], "types": p["types"], "base": p["base"],
        "abilities": p.get("abilities", []) if use_abilities else [], "ability": ability,
        "stats": stats, "level": level, "bst": sum(p["base"].values()),
        "physical": stats["atk"] >= stats["spa"],
        "coverage_types": sorted(dmg_types),
        "def_profile": defensive_profile(p["types"], ability),
    }


def _battler(pr: dict) -> Battler:
    return Battler(pr["display"], pr["types"], pr["stats"], pr["level"], pr["ability"])


def best_stab_damage(att: dict, dfn: dict, power: int = 90, fld: Field | None = None) -> float:
    """Average % of defender HP dealt by attacker's best STAB (generic power, best category)."""
    cat = "physical" if att["physical"] else "special"
    A, D = _battler(att), _battler(dfn)
    best = 0.0
    for t in att["types"]:
        r = calc(A, D, Move(f"STAB {t}", t, power, cat), fld or Field(doubles=False))
        best = max(best, float(np.mean(r["rolls"])))
    return best / dfn["stats"]["hp"] * 100


def damage_matrix(profiles: list, power: int = 90, cap: float = 100.0) -> pd.DataFrame:
    names = [p["display"] for p in profiles]
    M = np.zeros((len(profiles), len(profiles)))
    for i, a in enumerate(profiles):
        for j, d in enumerate(profiles):
            if i != j:
                M[i, j] = min(best_stab_damage(a, d, power), cap)
    return pd.DataFrame(M, index=names, columns=names)


# ------------------------------------------------------------------ single ranking
CRITERIA = [  # key, label, benefit, default weight, default used
    ("hp", "HP", True, 1, True),
    ("atk", "Attack", True, 1, True),
    ("def", "Defense", True, 1, True),
    ("spa", "Sp. Atk", True, 1, True),
    ("spd", "Sp. Def", True, 1, True),
    ("spe", "Speed", True, 1.5, True),
    ("bst", "BST (base total)", True, 1, False),
    ("offense", "Επιθετική ισχύς (% ζημιάς)", True, 3, True),
    ("taken", "Δεχόμενη ζημιά (%)", False, 3, True),
    ("coverage", "Κάλυψη (τύποι που χτυπά SE)", True, 2, True),
    ("weak", "Αδυναμίες (πλήθος)", False, 1.5, True),
    ("resist", "Αντιστάσεις + ανοσίες", True, 1.5, True),
    ("pbulk", "Φυσική αντοχή HP×Def", True, 1, False),
    ("sbulk", "Ειδική αντοχή HP×SpD", True, 1, False),
]


def single_criteria(profiles: list, dmg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(profiles)
    for i, p in enumerate(profiles):
        s = p["stats"]
        off = dmg.iloc[i].sum() / max(n - 1, 1)
        taken = dmg.iloc[:, i].sum() / max(n - 1, 1)
        rows.append({
            "hp": s["hp"], "atk": s["atk"], "def": s["def"], "spa": s["spa"], "spd": s["spd"], "spe": s["spe"],
            "bst": p["bst"], "offense": off, "taken": taken,
            "coverage": len(super_effective_targets(p["coverage_types"])),
            "weak": sum(1 for m in p["def_profile"].values() if m > 1),
            "resist": sum(1 for m in p["def_profile"].values() if m < 1),
            "pbulk": s["hp"] * s["def"], "sbulk": s["hp"] * s["spd"],
        })
    return pd.DataFrame(rows, index=[p["display"] for p in profiles])


# ------------------------------------------------------------------ synergy
WEATHER_SET = {"drought": "sun", "orichalcum-pulse": "sun", "desolate-land": "sun", "drizzle": "rain",
               "primordial-sea": "rain", "sand-stream": "sand", "sand-spit": "sand", "snow-warning": "snow"}
WEATHER_ABUSE = {
    "sun": {"chlorophyll", "solar-power", "protosynthesis", "flower-gift", "harvest", "leaf-guard"},
    "rain": {"swift-swim", "rain-dish", "dry-skin", "hydration"},
    "sand": {"sand-rush", "sand-force", "sand-veil"},
    "snow": {"slush-rush", "ice-body", "snow-cloak", "ice-face"},
}
WEATHER_TYPE = {"sun": "fire", "rain": "water", "sand": "rock", "snow": "ice"}
TERRAIN_SET = {"grassy-surge": "grass", "electric-surge": "electric", "hadron-engine": "electric",
               "psychic-surge": "psychic", "misty-surge": "fairy"}
TERRAIN_ABUSE = {"electric": {"quark-drive", "surge-surfer"}, "grass": {"grass-pelt"}, "psychic": set(),
                 "fairy": set()}
SUPPORT = {"intimidate", "friend-guard", "prankster", "hospitality", "telepathy", "follow-me",
           "healer", "power-spot", "battery", "flower-veil", "commander", "curious-medicine"}

SYN_WEIGHTS = {"defense": 0.35, "offense": 0.25, "roles": 0.15, "type_div": 0.10, "abilities": 0.15}
SYN_LABEL = {"defense": "Αμυντική κάλυψη", "offense": "Επιθετική κάλυψη", "roles": "Ρόλοι (Φυσ./Ειδ.)",
             "type_div": "Διαφορετικότητα τύπων", "abilities": "Abilities / καιρός / terrain"}


def _ability_score(a: dict, b: dict) -> tuple[float, str]:
    score, why = 0.5, []
    for x, y in ((a, b), (b, a)):
        for ab in x["abilities"]:
            w = WEATHER_SET.get(ab)
            if w and (set(y["abilities"]) & WEATHER_ABUSE[w] or WEATHER_TYPE[w] in y["types"]):
                score += 0.5
                why.append(f"{x['display']} ({ab}) → {y['display']}")
                break
        for ab in x["abilities"]:
            t = TERRAIN_SET.get(ab)
            if t and (set(y["abilities"]) & TERRAIN_ABUSE[t] or t in y["types"]):
                score += 0.35
                why.append(f"{x['display']} ({ab}) → {y['display']}")
                break
        if set(x["abilities"]) & SUPPORT:
            score += 0.15
    wa = {WEATHER_SET[ab] for ab in a["abilities"] if ab in WEATHER_SET}
    wb = {WEATHER_SET[ab] for ab in b["abilities"] if ab in WEATHER_SET}
    only_a = a["abilities"] and all(ab in WEATHER_SET for ab in a["abilities"])
    only_b = b["abilities"] and all(ab in WEATHER_SET for ab in b["abilities"])
    if wa and wb and not (wa & wb) and only_a and only_b:
        score -= 0.4
        why.append("σύγκρουση καιρού")
    return float(np.clip(score, 0, 1)), "; ".join(why)


def pair_components(a: dict, b: dict) -> dict:
    da, db = a["def_profile"], b["def_profile"]
    cover = sum(1 for t in TYPES if (da[t] > 1 and db[t] < 1) or (db[t] > 1 and da[t] < 1))
    shared = sum(1 for t in TYPES if da[t] > 1 and db[t] > 1)
    defense = float(np.clip(0.5 + (cover - 1.5 * shared) / 10, 0, 1))

    sa = super_effective_targets(a["coverage_types"])
    sb = super_effective_targets(b["coverage_types"])
    union = sa | sb
    gain = len(union) - max(len(sa), len(sb))
    offense = 0.5 * len(union) / len(TYPES) + 0.5 * min(gain / 6, 1)

    ra = a["stats"]["atk"] / max(a["stats"]["spa"], 1)
    rb = b["stats"]["atk"] / max(b["stats"]["spa"], 1)
    lean = lambda r: "P" if r > 1.1 else "S" if r < 1 / 1.1 else "M"
    la, lb = lean(ra), lean(rb)
    roles = 1.0 if {la, lb} == {"P", "S"} else 0.6 if "M" in (la, lb) else 0.3

    type_div = 1 - len(set(a["types"]) & set(b["types"])) / 2
    abil, why = _ability_score(a, b)
    return {"defense": defense, "offense": offense, "roles": roles, "type_div": type_div,
            "abilities": abil, "cover_pairs": cover, "shared_weak": shared, "ability_notes": why}


def synergy_score(comp: dict, weights: dict | None = None) -> float:
    w = weights or SYN_WEIGHTS
    tot = sum(w.values()) or 1
    return 100 * sum(comp[k] * w[k] for k in SYN_WEIGHTS) / tot


def synergy_dataset(profiles: list, weights: dict | None = None) -> pd.DataFrame:
    rows = []
    for a, b in combinations(profiles, 2):
        c = pair_components(a, b)
        rows.append({"Pokémon A": a["display"], "Pokémon B": b["display"],
                     **{SYN_LABEL[k]: round(c[k], 3) for k in SYN_WEIGHTS},
                     "Τύποι που καλύπτει ο ένας τον άλλον": c["cover_pairs"],
                     "Κοινές αδυναμίες": c["shared_weak"],
                     "Σημειώσεις abilities": c["ability_notes"],
                     "Synergy (0-100)": round(synergy_score(c, weights), 1)})
    return pd.DataFrame(rows)


def synergy_matrix(ds: pd.DataFrame, names: list) -> pd.DataFrame:
    M = pd.DataFrame(np.nan, index=names, columns=names)
    for _, r in ds.iterrows():
        M.loc[r["Pokémon A"], r["Pokémon B"]] = r["Synergy (0-100)"]
        M.loc[r["Pokémon B"], r["Pokémon A"]] = r["Synergy (0-100)"]
    return M


# ------------------------------------------------------------------ quads
QUAD_CRITERIA = [  # key, label, benefit, weight
    ("syn_mean", "Μέση συνέργεια ζευγών", True, 3),
    ("syn_min", "Ελάχιστη συνέργεια ζεύγους", True, 1.5),
    ("coverage", "Επιθετική κάλυψη (τύποι SE)", True, 2),
    ("stack_weak", "Μέγιστη συσσώρευση αδυναμίας", False, 2),
    ("uncovered", "Ακάλυπτες αδυναμίες (τύποι)", False, 2),
    ("indiv", "Μέση ατομική αξία (TOPSIS)", True, 2),
    ("balance", "Ισορροπία Φυσ./Ειδ.", True, 1),
    ("speed", "Μέση Speed", True, 1),
]


def quad_criteria(team: list, syn: pd.DataFrame, indiv: dict, size: int = 4) -> pd.DataFrame:
    """team: 6 profiles; syn: symmetric synergy matrix; indiv: {display: score 0-1}."""
    rows, idx = [], []
    for quad in combinations(team, size):
        names = [p["display"] for p in quad]
        pairs = [syn.loc[a, b] for a, b in combinations(names, 2)]
        cov = set()
        for p in quad:
            cov |= super_effective_targets(p["coverage_types"])
        stack, uncovered = 0, 0
        for t in TYPES:
            w = sum(1 for p in quad if p["def_profile"][t] > 1)
            r = sum(1 for p in quad if p["def_profile"][t] < 1)
            stack = max(stack, w - r)
            if w >= 1 and r == 0:
                uncovered += 1
        nphys = sum(1 for p in quad if p["physical"])
        rows.append({
            "syn_mean": float(np.mean(pairs)), "syn_min": float(np.min(pairs)),
            "coverage": len(cov), "stack_weak": stack, "uncovered": uncovered,
            "indiv": float(np.mean([indiv.get(n, 0.5) for n in names])),
            "balance": 1 - abs(nphys - (size - nphys)) / size,
            "speed": float(np.mean([p["stats"]["spe"] for p in quad])),
        })
        idx.append(" + ".join(names))
    return pd.DataFrame(rows, index=idx)
