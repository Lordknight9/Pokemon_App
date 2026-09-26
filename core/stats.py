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


# ready-made spreads used by the ranking pages
SPREADS = {
    "none": "Χωρίς EVs (IV 31, ουδέτερη φύση)",
    "offensive": "Επιθετικό: 252 επίθεση + 252 Speed, +φύση επίθεσης",
    "bulky": "Αμυντικό: 252 HP + 128 Def + 128 SpD, ουδέτερη φύση",
}


def spread_for(base: dict, kind: str):
    """Return (evs, nature) for a named spread."""
    if kind == "offensive":
        phys = base["atk"] >= base["spa"]
        return {"atk" if phys else "spa": 252, "spe": 252, "hp": 4}, ("Adamant" if phys else "Modest")
    if kind == "bulky":
        return {"hp": 252, "def": 128, "spd": 128}, "Hardy"
    return {}, "Hardy"
