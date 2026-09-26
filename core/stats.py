"""Stat formulas (Gen 3+ main-series)."""
import math

STATS = ["hp", "atk", "def", "spa", "spd", "spe"]
STAT_LABEL = {"hp": "HP", "atk": "Attack", "def": "Defense", "spa": "Sp. Atk", "spd": "Sp. Def", "spe": "Speed"}

# nature -> (increased, decreased); neutral natures have None
NATURES = {
    "Hardy": (None, None), "Lonely": ("atk", "def"), "Brave": ("atk", "spe"), "Adamant": ("atk", "spa"),
    "Naughty": ("atk", "spd"), "Bold": ("def", "atk"), "Docile": (None, None), "Relaxed": ("def", "spe"),
    "Impish": ("def", "spa"), "Lax": ("def", "spd"), "Timid": ("spe", "atk"), "Hasty": ("spe", "def"),
    "Serious": (None, None), "Jolly": ("spe", "spa"), "Naive": ("spe", "spd"), "Modest": ("spa", "atk"),
    "Mild": ("spa", "def"), "Quiet": ("spa", "spe"), "Bashful": (None, None), "Rash": ("spa", "spd"),
    "Calm": ("spd", "atk"), "Gentle": ("spd", "def"), "Sassy": ("spd", "spe"), "Careful": ("spd", "spa"),
    "Quirky": (None, None),
}


def nature_mult(nature: str, stat: str) -> float:
    up, down = NATURES.get(nature, (None, None))
    return 1.1 if stat == up else 0.9 if stat == down else 1.0


def calc_stat(stat: str, base: int, level: int, iv: int = 31, ev: int = 0, nature: str = "Hardy") -> int:
    core = math.floor((2 * base + iv + math.floor(ev / 4)) * level / 100)
    if stat == "hp":
        if base == 1:  # Shedinja
            return 1
        return core + level + 10
    return math.floor((core + 5) * nature_mult(nature, stat))


def calc_all(base: dict, level: int, ivs: dict | None = None, evs: dict | None = None, nature="Hardy") -> dict:
    ivs = ivs or {}
    evs = evs or {}
    return {s: calc_stat(s, base[s], level, ivs.get(s, 31), evs.get(s, 0), nature) for s in STATS}


def stage_mult(stage: int) -> float:
    stage = max(-6, min(6, int(stage)))
    return (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)


SHORT = {"hp": "HP", "atk": "Atk", "def": "Def", "spa": "SpA", "spd": "SpD", "spe": "Spe"}


def nature_label(nature: str) -> str:
    """'Adamant' -> 'Adamant (+Atk, −SpA)'."""
    up, down = NATURES.get(nature, (None, None))
    if not up:
        return f"{nature} (neutral)"
    return f"{nature} (+{SHORT[up]}, −{SHORT[down]})"


# ready-made spreads (used by Pokédex and the ranking pages)
SPREADS = {
    "recommended": "⭐ Recommended (auto, based on the Pokémon's stats)",
    "physical": "Physical attacker: 252 Atk / 252 Spe / 4 HP — Adamant",
    "special": "Special attacker: 252 SpA / 252 Spe / 4 HP — Modest",
    "offensive": "Best attacking stat (Atk or SpA): 252 / 252 Spe / 4 HP",
    "bulky": "Bulky: 252 HP / 128 Def / 128 SpD — neutral nature",
    "none": "No EVs (IV 31, neutral nature)",
}


def recommended_spread(base: dict) -> dict:
    """Heuristic 'standard' competitive spread from base stats.

    Returns {evs, nature, ivs, role}.
    """
    atk, spa, spe = base["atk"], base["spa"], base["spe"]
    bulk = base["hp"] + base["def"] + base["spd"]
    phys = atk >= spa
    best = max(atk, spa)
    mixed = abs(atk - spa) <= 10 and best >= 90
    A = "atk" if phys else "spa"
    ivs = {}
    if best < 90 and bulk >= 250:                       # support / wall
        if base["def"] <= base["spd"]:
            evs = {"hp": 252, "def": 252, "spd": 4}
            nature = "Bold" if atk <= spa or atk < 70 else "Impish"
        else:
            evs = {"hp": 252, "spd": 252, "def": 4}
            nature = "Calm" if atk <= spa or atk < 70 else "Careful"
        role = "Support / wall"
    elif spe < 50:                                      # Trick Room attacker
        evs = {"hp": 252, A: 252, "def": 4}
        nature = "Brave" if phys else "Quiet"
        ivs = {"spe": 0}
        role = ("Physical" if phys else "Special") + " Trick Room attacker (0 Spe IV)"
    elif spe >= 80:                                     # fast sweeper
        if mixed:
            evs = {"atk": 128, "spa": 128, "spe": 252}
            nature = "Naive"
            role = "Fast mixed attacker"
        else:
            evs = {A: 252, "spe": 252, "hp": 4}
            nature = "Jolly" if phys else "Timid"
            role = "Fast " + ("physical" if phys else "special") + " attacker"
    else:                                               # bulky attacker
        evs = {"hp": 252, A: 252, "def": 4}
        nature = "Adamant" if phys else "Modest"
        role = "Bulky " + ("physical" if phys else "special") + " attacker"
    return {"evs": evs, "nature": nature, "ivs": ivs, "role": role}


def spread_text(evs: dict, nature: str, ivs: dict | None = None) -> str:
    order = [s for s in STATS if evs.get(s)]
    txt = " / ".join(f"{evs[s]} {SHORT[s]}" for s in order) or "0 EVs"
    txt += f" — {nature_label(nature)}"
    if ivs:
        txt += " — IV " + ", ".join(f"{v} {SHORT[k]}" for k, v in ivs.items())
    return txt


def spread_for(base: dict, kind: str):
    """Return (evs, nature, ivs) for a named spread."""
    if kind == "recommended":
        r = recommended_spread(base)
        return r["evs"], r["nature"], r["ivs"]
    if kind == "physical":
        return {"atk": 252, "spe": 252, "hp": 4}, "Adamant", {}
    if kind == "special":
        return {"spa": 252, "spe": 252, "hp": 4}, "Modest", {}
    if kind == "offensive":
        phys = base["atk"] >= base["spa"]
        return {"atk" if phys else "spa": 252, "spe": 252, "hp": 4}, ("Adamant" if phys else "Modest"), {}
    if kind == "bulky":
        return {"hp": 252, "def": 128, "spd": 128}, "Hardy", {}
    return {}, "Hardy", {}
