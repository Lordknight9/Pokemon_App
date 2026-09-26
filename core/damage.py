"""Damage calculator — Gen 5-9 main-series formula (Scarlet/Violet mechanics).

Note: Legends Z-A uses a real-time battle system and Champions may tweak details;
this uses the standard turn-based formula, which is the usual reference point.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .stats import stage_mult
from .typechart import effectiveness


def poke_round(x: float) -> int:
    """Game rounding: .5 rounds down."""
    return math.ceil(x - 0.5)


@dataclass
class Battler:
    name: str
    types: list
    stats: dict                    # final (level-adjusted) stats
    level: int = 50
    ability: str | None = None
    item: str | None = None
    boosts: dict = field(default_factory=dict)   # {'atk': +1, ...}
    tera: str | None = None
    burned: bool = False
    hp_pct: float = 100.0          # current HP %

    @property
    def def_types(self):
        return [self.tera] if self.tera else self.types


@dataclass
class Move:
    name: str
    type: str
    power: int
    category: str                  # physical / special
    spread: bool = False


@dataclass
class Field:
    weather: str | None = None     # sun, rain, sand, snow
    terrain: str | None = None     # electric, grassy, psychic, misty
    crit: bool = False
    doubles: bool = True
    reflect: bool = False
    light_screen: bool = False


ITEMS = ["(κανένα)", "choice-band", "choice-specs", "life-orb", "expert-belt", "assault-vest",
         "eviolite", "type-boost (1.2x)"]


def _grounded(b: Battler) -> bool:
    return "flying" not in b.def_types and b.ability != "levitate" and b.item != "air-balloon"


# ---------------------------------------------------------------- abilities the calculator models
ATTACKER_ABILITIES = {
    "huge-power": "Διπλασιάζει το Attack", "pure-power": "Διπλασιάζει το Attack",
    "hustle": "Attack ×1.5", "gorilla-tactics": "Attack ×1.5",
    "guts": "Attack ×1.5 όταν έχει status (και αγνοεί το burn)",
    "technician": "Κινήσεις ≤60 BP ×1.5", "adaptability": "STAB ×2",
    "aerilate": "Normal → Flying, ×1.2", "pixilate": "Normal → Fairy, ×1.2",
    "refrigerate": "Normal → Ice, ×1.2", "galvanize": "Normal → Electric, ×1.2",
    "normalize": "Όλες οι κινήσεις Normal, ×1.2",
    "blaze": "Fire ×1.5 με ≤1/3 HP", "torrent": "Water ×1.5 με ≤1/3 HP",
    "overgrow": "Grass ×1.5 με ≤1/3 HP", "swarm": "Bug ×1.5 με ≤1/3 HP",
    "transistor": "Electric ×1.3", "dragons-maw": "Dragon ×1.5", "steelworker": "Steel ×1.5",
    "steely-spirit": "Steel ×1.5", "rocky-payload": "Rock ×1.5", "water-bubble": "Water ×2 (και Fire ×0.5 που δέχεται)",
    "sand-force": "Rock/Ground/Steel ×1.3 σε sand", "solar-power": "Sp. Atk ×1.5 σε sun",
    "orichalcum-pulse": "Attack ×1.33 σε sun", "hadron-engine": "Sp. Atk ×1.33 σε Electric Terrain",
    "protosynthesis": "Υψηλότερο stat ×1.3 σε sun", "quark-drive": "Υψηλότερο stat ×1.3 σε Electric Terrain",
    "tinted-lens": "Not very effective ×2", "neuroforce": "Super effective ×1.25",
    "sniper": "Critical hit ×2.25", "scrappy": "Normal/Fighting χτυπούν Ghost",
    "minds-eye": "Normal/Fighting χτυπούν Ghost", "unaware": "Αγνοεί τα boosts άμυνας του στόχου",
    "mold-breaker": "Αγνοεί την ability του στόχου", "teravolt": "Αγνοεί την ability του στόχου",
    "turboblaze": "Αγνοεί την ability του στόχου",
}
DEFENDER_ABILITIES = {
    "multiscale": "Ζημιά ×0.5 με γεμάτο HP", "shadow-shield": "Ζημιά ×0.5 με γεμάτο HP",
    "filter": "Super effective ×0.75", "solid-rock": "Super effective ×0.75",
    "prism-armor": "Super effective ×0.75", "thick-fat": "Fire/Ice ×0.5", "heatproof": "Fire ×0.5",
    "water-bubble": "Fire ×0.5", "fluffy": "Fire ×2", "dry-skin": "Fire ×1.25, ανοσία Water",
    "ice-scales": "Special ×0.5", "fur-coat": "Defense ×2", "purifying-salt": "Ghost ×0.5",
    "levitate": "Ανοσία Ground", "flash-fire": "Ανοσία Fire", "well-baked-body": "Ανοσία Fire",
    "water-absorb": "Ανοσία Water", "storm-drain": "Ανοσία Water", "volt-absorb": "Ανοσία Electric",
    "lightning-rod": "Ανοσία Electric", "motor-drive": "Ανοσία Electric", "sap-sipper": "Ανοσία Grass",
    "earth-eater": "Ανοσία Ground", "wonder-guard": "Μόνο super effective κινήσεις το χτυπούν",
    "unaware": "Αγνοεί τα boosts επίθεσης του αντιπάλου",
    "protosynthesis": "Υψηλότερο stat ×1.3 σε sun", "quark-drive": "Υψηλότερο stat ×1.3 σε Electric Terrain",
    "intimidate": "Attack του αντιπάλου −1 (επιλογή στον calculator)",
}
CALC_ABILITIES = set(ATTACKER_ABILITIES) | set(DEFENDER_ABILITIES)
_ATE = {"aerilate": "flying", "pixilate": "fairy", "refrigerate": "ice", "galvanize": "electric"}
_PINCH = {"blaze": "fire", "torrent": "water", "overgrow": "grass", "swarm": "bug"}
_IMMUNE = {"flash-fire": "fire", "water-absorb": "water", "storm-drain": "water", "dry-skin": "water",
           "volt-absorb": "electric", "lightning-rod": "electric", "motor-drive": "electric",
           "sap-sipper": "grass", "earth-eater": "ground", "well-baked-body": "fire", "levitate": "ground"}
INTIMIDATE_IMMUNE = {"clear-body", "hyper-cutter", "white-smoke", "full-metal-body", "inner-focus",
                     "oblivious", "own-tempo", "scrappy"}


def intimidate_stage(ability: str | None) -> int:
    """Attack stage change an attacker with this ability gets from an opposing Intimidate."""
    if ability in INTIMIDATE_IMMUNE:
        return 0
    if ability == "guard-dog":
        return 1
    if ability == "defiant":
        return 1          # −1 then +2
    return -1


def _booster_stat(b: Battler) -> str:
    """Highest non-HP stat (for Protosynthesis / Quark Drive)."""
    return max(("atk", "def", "spa", "spd", "spe"), key=lambda s: b.stats[s])


def calc(att: Battler, dfn: Battler, mv: Move, fld: Field | None = None) -> dict:
    fld = fld or Field()
    if mv.category not in ("physical", "special") or not mv.power:
        return {"rolls": [0] * 16, "min": 0, "max": 0, "eff": 0, "note": "Status move"}

    phys = mv.category == "physical"
    a_key, d_key = ("atk", "def") if phys else ("spa", "spd")
    aab = att.ability
    dab = None if aab in ("mold-breaker", "teravolt", "turboblaze") else dfn.ability

    # ---- type-changing abilities ----
    mtype, power, ate = mv.type, mv.power, False
    if aab in _ATE and mtype == "normal":
        mtype, ate = _ATE[aab], True
    elif aab == "normalize":
        mtype, ate = "normal", True

    # ---- type effectiveness / immunities ----
    def_types = dfn.def_types
    if aab in ("scrappy", "minds-eye") and mtype in ("normal", "fighting"):
        def_types = [t for t in def_types if t != "ghost"] or ["normal"]
    eff = effectiveness(mtype, def_types)
    if _IMMUNE.get(dab) == mtype:
        eff = 0
    if dab == "wonder-guard" and eff <= 1:
        eff = 0
    if eff == 0:
        return {"rolls": [0] * 16, "min": 0, "max": 0, "eff": 0, "note": "Immune", "type": mtype}

    # ---- base power modifiers ----
    if aab == "technician" and power <= 60:
        power = math.floor(power * 1.5)
    if ate:
        power = poke_round(power * 4915 / 4096)
    if aab == "sand-force" and fld.weather == "sand" and mtype in ("rock", "ground", "steel"):
        power = poke_round(power * 5325 / 4096)
    if fld.terrain and _grounded(att) and (fld.terrain, mtype) in (
            ("electric", "electric"), ("grassy", "grass"), ("psychic", "psychic")):
        power = poke_round(power * 5325 / 4096)
    if fld.terrain == "misty" and mtype == "dragon" and _grounded(dfn):
        power = poke_round(power * 0.5)
    if att.item == "type-boost (1.2x)":
        power = poke_round(power * 4915 / 4096)

    # ---- attack stat ----
    a_stage = 0 if dab == "unaware" else att.boosts.get(a_key, 0)
    if fld.crit and a_stage < 0:
        a_stage = 0
    A = math.floor(att.stats[a_key] * stage_mult(a_stage))
    if phys and aab in ("huge-power", "pure-power"):
        A *= 2
    if phys and aab in ("hustle", "gorilla-tactics"):
        A = poke_round(A * 1.5)
    if phys and aab == "guts" and att.burned:
        A = math.floor(A * 1.5)
    if aab in _PINCH and _PINCH[aab] == mtype and att.hp_pct <= 100 / 3:
        A = poke_round(A * 1.5)
    boost_type = {"transistor": ("electric", 5325 / 4096), "dragons-maw": ("dragon", 1.5),
                  "steelworker": ("steel", 1.5), "steely-spirit": ("steel", 1.5),
                  "rocky-payload": ("rock", 1.5), "water-bubble": ("water", 2.0)}.get(aab)
    if boost_type and boost_type[0] == mtype:
        A = poke_round(A * boost_type[1])
    if aab == "solar-power" and not phys and fld.weather == "sun":
        A = poke_round(A * 1.5)
    if aab == "orichalcum-pulse" and phys and fld.weather == "sun":
        A = poke_round(A * 5461 / 4096)
    if aab == "hadron-engine" and not phys and fld.terrain == "electric":
        A = poke_round(A * 5461 / 4096)
    if ((aab == "protosynthesis" and fld.weather == "sun") or
            (aab == "quark-drive" and fld.terrain == "electric")) and _booster_stat(att) == a_key:
        A = poke_round(A * 5325 / 4096)
    if (phys and att.item == "choice-band") or (not phys and att.item == "choice-specs"):
        A = poke_round(A * 1.5)
    if dab == "thick-fat" and mtype in ("fire", "ice"):
        A = poke_round(A * 0.5)
    if dab in ("heatproof", "water-bubble") and mtype == "fire":
        A = poke_round(A * 0.5)

    # ---- defense stat ----
    d_stage = 0 if aab == "unaware" else dfn.boosts.get(d_key, 0)
    if fld.crit and d_stage > 0:
        d_stage = 0
    D = math.floor(dfn.stats[d_key] * stage_mult(d_stage))
    if fld.weather == "sand" and not phys and "rock" in dfn.def_types:
        D = math.floor(D * 1.5)
    if fld.weather == "snow" and phys and "ice" in dfn.def_types:
        D = math.floor(D * 1.5)
    if phys and dab == "fur-coat":
        D *= 2
    if ((dab == "protosynthesis" and fld.weather == "sun") or
            (dab == "quark-drive" and fld.terrain == "electric")) and _booster_stat(dfn) == d_key:
        D = poke_round(D * 5325 / 4096)
    if not phys and dfn.item == "assault-vest":
        D = poke_round(D * 1.5)
    if dfn.item == "eviolite":
        D = poke_round(D * 1.5)
    D = max(D, 1)

    # ---- base damage ----
    base = math.floor(math.floor(math.floor(2 * att.level / 5 + 2) * power * A / D) / 50) + 2

    # ---- modifiers in game order ----
    if mv.spread and fld.doubles:
        base = poke_round(base * 0.75)
    if fld.weather == "sun":
        base = poke_round(base * (1.5 if mtype == "fire" else 0.5 if mtype == "water" else 1))
    elif fld.weather == "rain":
        base = poke_round(base * (1.5 if mtype == "water" else 0.5 if mtype == "fire" else 1))
    if fld.crit:
        base = math.floor(base * (2.25 if aab == "sniper" else 1.5))

    stab = 1.0
    if mtype in att.types or (att.tera and mtype == att.tera):
        stab = 1.5
        if att.tera and mtype == att.tera and mtype in att.types:
            stab = 2.0
        if aab == "adaptability":
            stab = 2.25 if stab == 2.0 else 2.0

    final = 1.0
    if (fld.reflect and phys) or (fld.light_screen and not phys):
        if not fld.crit:
            final *= (2732 / 4096) if fld.doubles else 0.5
    if dab in ("multiscale", "shadow-shield") and dfn.hp_pct >= 100:
        final *= 0.5
    if dab in ("filter", "solid-rock", "prism-armor") and eff > 1:
        final *= 0.75
    if dab == "ice-scales" and not phys:
        final *= 0.5
    if dab == "fluffy" and mtype == "fire":
        final *= 2
    if dab == "dry-skin" and mtype == "fire":
        final *= 1.25
    if dab == "purifying-salt" and mtype == "ghost":
        final *= 0.5
    if aab == "tinted-lens" and eff < 1:
        final *= 2
    if aab == "neuroforce" and eff > 1:
        final *= 1.25
    if att.item == "expert-belt" and eff > 1:
        final *= 4915 / 4096
    if att.item == "life-orb":
        final *= 5324 / 4096

    rolls = []
    for r in range(85, 101):
        d = math.floor(base * r / 100)
        d = poke_round(d * stab)
        d = math.floor(d * eff)
        if att.burned and phys and aab != "guts":
            d = math.floor(d / 2)
        d = poke_round(d * final)
        rolls.append(max(1, d))
    return {"rolls": rolls, "min": rolls[0], "max": rolls[-1], "eff": eff, "stab": stab, "type": mtype,
            "A": A, "D": D, "power": power, "note": ""}


def ko_chances(rolls: list, hp: int, max_hits: int = 4) -> dict:
    """Probability of KO in exactly <=n hits (independent uniform rolls)."""
    if not rolls or max(rolls) == 0:
        return {}
    out = {}
    dist = {0: 1.0}
    for n in range(1, max_hits + 1):
        new = {}
        for dmg, p in dist.items():
            for r in rolls:
                new[min(dmg + r, hp)] = new.get(min(dmg + r, hp), 0) + p / len(rolls)
        dist = new
        out[n] = dist.get(hp, 0.0)
    return out


def ko_text(rolls, hp) -> str:
    ch = ko_chances(rolls, hp)
    if not ch:
        return "Καμία ζημιά"
    for n, p in ch.items():
        if p > 0:
            tag = f"{n}HKO" if n > 1 else "OHKO"
            return f"Εγγυημένο {tag}" if p >= 0.9999 else f"{p * 100:.1f}% πιθανότητα για {tag}"
    return f"Χρειάζονται >4 χτυπήματα (~{math.ceil(hp / max(1, sum(rolls) / len(rolls)))} κατά μέσο όρο)"
