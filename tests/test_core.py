"""Offline tests: formulas, MCDA and the PokeAPI parser (with fake API responses).

Run:  python -m pytest -q   (or: python tests/test_core.py)
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.damage import Battler, Field, Move, calc, ko_chances  # noqa: E402
from core.data import LiveProvider, NotFound  # noqa: E402
from core.mcda import promethee, topsis  # noqa: E402
from core.rosters import parse_entry, static_roster  # noqa: E402
from core.stats import calc_stat  # noqa: E402
from core.typechart import effectiveness  # noqa: E402


def test_stats():
    assert calc_stat("hp", 108, 50) == 183
    assert calc_stat("hp", 108, 100, 31, 252) == 420
    assert calc_stat("atk", 130, 50, 31, 252, "Adamant") == 200
    assert calc_stat("spe", 102, 50, 31, 252, "Jolly") == 169


def test_types():
    assert effectiveness("ground", ["fire", "dark"]) == 2
    assert effectiveness("ice", ["dragon", "ground"]) == 4
    assert effectiveness("ghost", ["normal"]) == 0
    assert effectiveness("fairy", ["dragon", "dark"]) == 4


def test_damage_hand_calc():
    # 252+ Garchomp spread EQ vs 0/0 Incineroar, doubles: base 82 -> 61 after spread
    A = Battler("Garchomp", ["dragon", "ground"], {"hp": 184, "atk": 200, "def": 115, "spa": 90, "spd": 105,
                                                    "spe": 154}, 50)
    D = Battler("Incineroar", ["fire", "dark"], {"hp": 171, "atk": 183, "def": 110, "spa": 90, "spd": 110,
                                                 "spe": 112}, 50)
    r = calc(A, D, Move("earthquake", "ground", 100, "physical", spread=True), Field())
    assert (r["min"], r["max"]) == (152, 182)
    assert len(r["rolls"]) == 16
    # levitate immunity & tera STAB
    D.ability = "levitate"
    assert calc(A, D, Move("eq", "ground", 100, "physical"), Field())["max"] == 0
    D.ability = None
    A.tera = "ground"
    assert calc(A, D, Move("eq", "ground", 100, "physical"), Field())["stab"] == 2.0


def test_ko_chances():
    ch = ko_chances([50] * 8 + [60] * 8, 100)
    assert ch[1] == 0 and ch[2] == 1


def test_topsis_promethee_dominance():
    X = pd.DataFrame({"a": [3, 2, 1], "b": [30, 20, 10], "c": [1, 2, 3]}, index=["best", "mid", "worst"])
    ben = [True, True, False]
    t = topsis(X, [1, 1, 1], ben)
    p = promethee(X, [1, 1, 1], ben)
    assert list(t.sort_values("TOPSIS C*", ascending=False).index) == ["best", "mid", "worst"]
    assert list(p.sort_values("Φ (net)", ascending=False).index) == ["best", "mid", "worst"]
    assert abs(p["Φ (net)"].sum()) < 1e-9
    for kind in ["usual", "u-shape", "v-shape", "level", "linear", "gaussian"]:
        r = promethee(X, [1, 1, 1], ben, [{"kind": kind}] * 3)
        assert r.loc["best", "Φ (net)"] >= r.loc["worst", "Φ (net)"]


def test_rosters():
    assert parse_entry("Alolan Raichu")["pokemon"] == "raichu-alola"
    assert parse_entry("Farfetch'd")["pokemon"] == "farfetchd"
    assert parse_entry("Mr. Rime")["pokemon"] == "mr-rime"
    assert parse_entry("Mime Jr.")["pokemon"] == "mime-jr"
    assert parse_entry("Flabébé")["pokemon"] == "flabebe"
    assert parse_entry("Paldean Tauros")["pokemon"] == "tauros-paldea-combat-breed"
    assert len(static_roster("champions")) > 200
    assert len(static_roster("za")) > 350


# ---------- fake PokeAPI ----------
def _pk(name, species, types, moves):
    return {"name": name, "id": 1, "species": {"name": species},
            "types": [{"slot": i + 1, "type": {"name": t}} for i, t in enumerate(types)],
            "stats": [{"base_stat": v, "stat": {"name": n}} for n, v in
                      zip(["hp", "attack", "defense", "special-attack", "special-defense", "speed"],
                          [78, 84, 78, 109, 85, 100])],
            "abilities": [{"slot": 1, "ability": {"name": "blaze"}, "is_hidden": False}],
            "moves": [{"move": {"name": m}, "version_group_details": [{"version_group": {"name": vg}}]}
                      for m, vg in moves],
            "sprites": {"front_default": "x.png"}}


FAKE = {
    "pokemon/toxtricity": None,  # 404 -> species fallback
    "pokemon-species/toxtricity": {"name": "toxtricity", "varieties": [
        {"is_default": True, "pokemon": {"name": "toxtricity-amped"}},
        {"is_default": False, "pokemon": {"name": "toxtricity-low-key"}}]},
    "pokemon/toxtricity-amped": _pk("toxtricity-amped", "toxtricity", ["electric", "poison"],
                                    [("overdrive", "scarlet-violet"), ("tackle", "sword-shield")]),
    "pokemon/charizard-mega-x": _pk("charizard-mega-x", "charizard", ["fire", "dragon"], []),
    "pokemon-species/charizard": {"name": "charizard", "varieties": [
        {"is_default": True, "pokemon": {"name": "charizard"}},
        {"is_default": False, "pokemon": {"name": "charizard-mega-x"}}]},
    "pokemon/charizard": _pk("charizard", "charizard", ["fire", "flying"], [("flamethrower", "scarlet-violet")]),
    "pokedex/paldea": {"pokemon_entries": [{"pokemon_species": {"name": "sprigatito"}}]},
    "pokedex/kitakami": {"pokemon_entries": [{"pokemon_species": {"name": "sprigatito"}},
                                             {"pokemon_species": {"name": "ogerpon"}}]},
    "pokedex/blueberry": {"pokemon_entries": []},
    "move/overdrive": {"name": "overdrive", "type": {"name": "electric"}, "power": 80, "accuracy": 100,
                       "pp": 10, "priority": 0, "damage_class": {"name": "special"}},
}


class FakeProvider(LiveProvider):
    def _get(self, path):
        if path not in FAKE or FAKE[path] is None:
            raise NotFound(path)
        return FAKE[path]


def test_live_provider_parsing():
    with tempfile.TemporaryDirectory() as d:
        fp = FakeProvider(cache_dir=d)
        t = fp.pokemon(parse_entry("Toxtricity"))
        assert t["name"] == "toxtricity-amped" and t["types"] == ["electric", "poison"]
        assert t["base"]["spa"] == 109
        assert fp.learnset(t, "sv") == ["overdrive"]
        assert fp.learnset(t, "za") == ["overdrive"]          # falls back to SV learnset
        m = fp.pokemon({"display": "Mega Charizard X", "species": "charizard",
                        "pokemon": "charizard-mega-x", "form": "mega"})
        assert m["types"] == ["fire", "dragon"] and "flamethrower" in m["moves"]
        assert [e["species"] for e in fp.roster("sv")] == ["sprigatito", "ogerpon"]
        assert fp.move("overdrive")["category"] == "special"
        # cache hit works without the fake API
        FakeProvider._get = lambda self, p: (_ for _ in ()).throw(RuntimeError("no net"))
        assert fp.pokemon(parse_entry("Toxtricity"))["name"] == "toxtricity-amped"


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("✓", k)
