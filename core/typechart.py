"""Gen 6+ type effectiveness chart (18 types).

Kept as a constant so the app works even when PokeAPI is slow/unreachable.
"""

TYPES = [
    "normal", "fire", "water", "electric", "grass", "ice", "fighting", "poison",
    "ground", "flying", "psychic", "bug", "rock", "ghost", "dragon", "dark",
    "steel", "fairy",
]

# attacking type -> {defending type: multiplier}; missing entries are 1.0
_CHART = {
    "normal": {"rock": .5, "ghost": 0, "steel": .5},
    "fire": {"fire": .5, "water": .5, "grass": 2, "ice": 2, "bug": 2, "rock": .5, "dragon": .5, "steel": 2},
    "water": {"fire": 2, "water": .5, "grass": .5, "ground": 2, "rock": 2, "dragon": .5},
    "electric": {"water": 2, "electric": .5, "grass": .5, "ground": 0, "flying": 2, "dragon": .5},
    "grass": {"fire": .5, "water": 2, "grass": .5, "poison": .5, "ground": 2, "flying": .5, "bug": .5,
              "rock": 2, "dragon": .5, "steel": .5},
    "ice": {"fire": .5, "water": .5, "grass": 2, "ice": .5, "ground": 2, "flying": 2, "dragon": 2, "steel": .5},
    "fighting": {"normal": 2, "ice": 2, "poison": .5, "flying": .5, "psychic": .5, "bug": .5, "rock": 2,
                 "ghost": 0, "dark": 2, "steel": 2, "fairy": .5},
    "poison": {"grass": 2, "poison": .5, "ground": .5, "rock": .5, "ghost": .5, "steel": 0, "fairy": 2},
    "ground": {"fire": 2, "electric": 2, "grass": .5, "poison": 2, "flying": 0, "bug": .5, "rock": 2, "steel": 2},
    "flying": {"electric": .5, "grass": 2, "fighting": 2, "bug": 2, "rock": .5, "steel": .5},
    "psychic": {"fighting": 2, "poison": 2, "psychic": .5, "dark": 0, "steel": .5},
    "bug": {"fire": .5, "grass": 2, "fighting": .5, "poison": .5, "flying": .5, "psychic": 2, "ghost": .5,
            "dark": 2, "steel": .5, "fairy": .5},
    "rock": {"fire": 2, "ice": 2, "fighting": .5, "ground": .5, "flying": 2, "bug": 2, "steel": .5},
    "ghost": {"normal": 0, "psychic": 2, "ghost": 2, "dark": .5},
    "dragon": {"dragon": 2, "steel": .5, "fairy": 0},
    "dark": {"fighting": .5, "psychic": 2, "ghost": 2, "dark": .5, "fairy": .5},
    "steel": {"fire": .5, "water": .5, "electric": .5, "ice": 2, "rock": 2, "steel": .5, "fairy": 2},
    "fairy": {"fire": .5, "fighting": 2, "poison": .5, "dragon": 2, "dark": 2, "steel": .5},
}

TYPE_GR = {
    "normal": "Normal", "fire": "Fire", "water": "Water", "electric": "Electric", "grass": "Grass",
    "ice": "Ice", "fighting": "Fighting", "poison": "Poison", "ground": "Ground", "flying": "Flying",
    "psychic": "Psychic", "bug": "Bug", "rock": "Rock", "ghost": "Ghost", "dragon": "Dragon",
    "dark": "Dark", "steel": "Steel", "fairy": "Fairy",
}


def effectiveness(atk_type: str, def_types) -> float:
    """Multiplier of an attacking type against a list of defending types."""
    m = 1.0
    row = _CHART.get(atk_type, {})
    for t in def_types:
        m *= row.get(t, 1.0)
    return m


def defensive_profile(def_types, ability: str | None = None) -> dict:
    """Multiplier taken from every attacking type (with a few ability immunities)."""
    prof = {t: effectiveness(t, def_types) for t in TYPES}
    immun = {
        "levitate": "ground", "flash-fire": "fire", "water-absorb": "water", "storm-drain": "water",
        "dry-skin": "water", "volt-absorb": "electric", "lightning-rod": "electric",
        "motor-drive": "electric", "sap-sipper": "grass", "earth-eater": "ground",
        "well-baked-body": "fire",
    }
    if ability in immun:
        prof[immun[ability]] = 0.0
    if ability == "thick-fat":
        prof["fire"] *= .5
        prof["ice"] *= .5
    if ability == "purifying-salt":
        prof["ghost"] *= .5
    return prof


def weaknesses(def_types, ability=None):
    return [t for t, m in defensive_profile(def_types, ability).items() if m > 1]


def resistances(def_types, ability=None):
    return [t for t, m in defensive_profile(def_types, ability).items() if m < 1]


def super_effective_targets(attack_types) -> set:
    """Mono-types hit super-effectively by at least one of the given attacking types."""
    out = set()
    for a in attack_types:
        for d in TYPES:
            if effectiveness(a, [d]) > 1:
                out.add(d)
    return out
