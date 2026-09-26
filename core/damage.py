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


def calc(att: Battler, dfn: Battler, mv: Move, fld: Field | None = None) -> dict:
    fld = fld or Field()
    if mv.category not in ("physical", "special") or not mv.power:
        return {"rolls": [0] * 16, "min": 0, "max": 0, "eff": 0, "note": "Status move"}

    phys = mv.category == "physical"
    a_key, d_key = ("atk", "def") if phys else ("spa", "spd")

    # ---- type effectiveness / immunities ----
    eff = effectiveness(mv.type, dfn.def_types)
    if mv.type == "ground" and dfn.ability == "levitate":
        eff = 0
    immune_abil = {"flash-fire": "fire", "water-absorb": "water", "storm-drain": "water", "dry-skin": "water",
                   "volt-absorb": "electric", "lightning-rod": "electric", "motor-drive": "electric",
                   "sap-sipper": "grass", "earth-eater": "ground", "well-baked-body": "fire"}
    if immune_abil.get(dfn.ability) == mv.type:
        eff = 0
    if eff == 0:
        return {"rolls": [0] * 16, "min": 0, "max": 0, "eff": 0, "note": "Immune"}

    # ---- base power modifiers ----
    power = mv.power
    if att.ability == "technician" and power <= 60:
        power = math.floor(power * 1.5)
    if fld.terrain and _grounded(att) and (fld.terrain, mv.type) in (
            ("electric", "electric"), ("grassy", "grass"), ("psychic", "psychic")):
        power = poke_round(power * 5325 / 4096)
    if fld.terrain == "misty" and mv.type == "dragon" and _grounded(dfn):
        power = poke_round(power * 0.5)
    if att.item == "type-boost (1.2x)":
        power = poke_round(power * 4915 / 4096)

    # ---- attack stat ----
    a_stage = att.boosts.get(a_key, 0)
    if fld.crit and a_stage < 0:
        a_stage = 0
    A = math.floor(att.stats[a_key] * stage_mult(a_stage))
    if phys and att.ability in ("huge-power", "pure-power"):
        A *= 2
    if phys and att.ability == "guts" and att.burned:
        A = math.floor(A * 1.5)
    if (phys and att.item == "choice-band") or (not phys and att.item == "choice-specs"):
        A = poke_round(A * 1.5)
    if dfn.ability == "thick-fat" and mv.type in ("fire", "ice"):
        A = poke_round(A * 0.5)
    if att.ability == "solar-power" and not phys and fld.weather == "sun":
        A = poke_round(A * 1.5)

    # ---- defense stat ----
    d_stage = dfn.boosts.get(d_key, 0)
    if fld.crit and d_stage > 0:
        d_stage = 0
    D = math.floor(dfn.stats[d_key] * stage_mult(d_stage))
    if fld.weather == "sand" and not phys and "rock" in dfn.def_types:
        D = math.floor(D * 1.5)
    if fld.weather == "snow" and phys and "ice" in dfn.def_types:
        D = math.floor(D * 1.5)
    if phys and dfn.ability == "fur-coat":
        D *= 2
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
        base = poke_round(base * (1.5 if mv.type == "fire" else 0.5 if mv.type == "water" else 1))
    elif fld.weather == "rain":
        base = poke_round(base * (1.5 if mv.type == "water" else 0.5 if mv.type == "fire" else 1))
    if fld.crit:
        base = math.floor(base * 1.5)

    stab = 1.0
    if mv.type in att.types or (att.tera and mv.type == att.tera):
        stab = 1.5
        if att.tera and mv.type == att.tera and mv.type in att.types:
            stab = 2.0
        if att.ability == "adaptability":
            stab = 2.25 if stab == 2.0 else 2.0

    final = 1.0
    if (fld.reflect and phys) or (fld.light_screen and not phys):
        if not fld.crit:
            final *= (2732 / 4096) if fld.doubles else 0.5
    if dfn.ability == "multiscale" and dfn.hp_pct >= 100:
        final *= 0.5
    if dfn.ability in ("filter", "solid-rock", "prism-armor") and eff > 1:
        final *= 0.75
    if att.ability == "tinted-lens" and eff < 1:
        final *= 2
    if att.item == "expert-belt" and eff > 1:
        final *= 4915 / 4096
    if att.item == "life-orb":
        final *= 5324 / 4096

    rolls = []
    for r in range(85, 101):
        d = math.floor(base * r / 100)
        d = poke_round(d * stab)
        d = math.floor(d * eff)
        if att.burned and phys and att.ability != "guts":
            d = math.floor(d / 2)
        d = poke_round(d * final)
        rolls.append(max(1, d))
    return {"rolls": rolls, "min": rolls[0], "max": rolls[-1], "eff": eff, "stab": stab,
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
