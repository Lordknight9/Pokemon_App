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


def test_recommended_spreads():
    from core.stats import recommended_spread, spread_text
    r = recommended_spread({"hp": 108, "atk": 130, "def": 95, "spa": 80, "spd": 85, "spe": 102})
    assert r["nature"] == "Jolly" and r["evs"]["atk"] == 252
    r = recommended_spread({"hp": 130, "atk": 140, "def": 105, "spa": 45, "spd": 80, "spe": 40})
    assert r["ivs"] == {"spe": 0}
    r = recommended_spread({"hp": 114, "atk": 85, "def": 70, "spa": 85, "spd": 80, "spe": 30})
    assert "wall" in r["role"] or "Trick Room" in r["role"]
    assert "Adamant (+Atk, −SpA)" in spread_text({"atk": 252}, "Adamant")


def test_evo_condition():
    from core.data import evo_condition
    assert evo_condition({"trigger": {"name": "level-up"}, "min_level": 36}) == "Lv. 36"
    assert evo_condition({"trigger": {"name": "use-item"}, "item": {"name": "thunder-stone"}}) == "Use Thunder Stone"
    assert evo_condition({"trigger": {"name": "level-up"}, "min_happiness": 160, "time_of_day": "day"}) == \
        "Level up high Friendship (Day)"
    assert evo_condition({"trigger": {"name": "trade"}, "held_item": {"name": "metal-coat"}}) == \
        "Trade holding Metal Coat"


def test_ability_calcs():
    from core.damage import intimidate_stage
    st = {"hp": 150, "atk": 150, "def": 100, "spa": 100, "spd": 100, "spe": 100}
    A = Battler("A", ["normal"], dict(st), 50)
    D = Battler("D", ["ghost"], dict(st), 50)
    mv = Move("tackle", "normal", 80, "physical")
    assert calc(A, D, mv)["max"] == 0
    A.ability = "scrappy"
    assert calc(A, D, mv)["max"] > 0
    A.ability = "pixilate"
    r = calc(A, D, mv)
    assert r["type"] == "fairy" and r["stab"] == 1.0
    D2 = Battler("D2", ["water"], dict(st), 50, "multiscale")
    A.ability = None
    full = calc(A, D2, mv)["max"]
    D2.hp_pct = 50
    assert calc(A, D2, mv)["max"] > full
    A.ability = "mold-breaker"
    D2.hp_pct = 100
    assert calc(A, D2, mv)["max"] > full
    assert intimidate_stage("clear-body") == 0 and intimidate_stage("defiant") == 1
    assert intimidate_stage(None) == -1


def test_move_flags_and_abilities():
    from core.move_flags import fallback_table, parse_showdown, sd_id
    raw = {"closecombat": {"flags": {"contact": 1, "protect": 1}, "secondary": None, "self": {"boosts": {}}},
           "crunch": {"flags": {"contact": 1, "bite": 1}, "secondary": {"chance": 20}},
           "flareblitz": {"flags": {"contact": 1}, "recoil": [33, 100], "secondary": {"chance": 10}}}
    t = parse_showdown(raw)
    assert t["closecombat"] == ["contact"] and set(t["crunch"]) == {"contact", "bite", "secondary"}
    assert "recoil" in t["flareblitz"] and sd_id("close-combat") == "closecombat"
    assert "punch" in fallback_table()["drainpunch"]
    st = {"hp": 150, "atk": 150, "def": 100, "spa": 100, "spd": 100, "spe": 100}
    D = Battler("D", ["water"], dict(st), 50)
    bite = Move("crunch", "dark", 80, "physical", flags=frozenset({"contact", "bite", "secondary"}))
    plain = calc(Battler("A", ["normal"], dict(st), 50), D, bite)["max"]
    for ab in ("strong-jaw", "tough-claws", "sheer-force"):
        assert calc(Battler("A", ["normal"], dict(st), 50, ab), D, bite)["max"] > plain, ab
    D.ability = "fluffy"
    assert calc(Battler("A", ["normal"], dict(st), 50), D, bite)["max"] < plain
    D.ability = "bulletproof"
    ball = Move("shadow-ball", "ghost", 80, "special", flags=frozenset({"bullet"}))
    assert calc(Battler("A", ["normal"], dict(st), 50), D, ball)["max"] == 0


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
    "pokemon-species?limit=3000": {"results": [
        {"name": "charizard", "url": "https://pokeapi.co/api/v2/pokemon-species/6/"},
        {"name": "sprigatito", "url": "https://pokeapi.co/api/v2/pokemon-species/906/"},
        {"name": "ogerpon", "url": "https://pokeapi.co/api/v2/pokemon-species/1017/"},
        {"name": "floette", "url": "https://pokeapi.co/api/v2/pokemon-species/670/"}]},
    "pokemon?limit=5000": {"results": [{"name": n, "url": f"https://pokeapi.co/api/v2/pokemon/{i}/"} for i, n in
                                       enumerate(["charizard", "charizard-mega-x", "charizard-mega-y",
                                                  "venusaur-mega", "floette-eternal", "floette-eternal-mega"],
                                                 start=10000)]},
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
        assert [(e["dex"], e["species"]) for e in fp.roster("sv")] == [(906, "sprigatito"), (1017, "ogerpon")]
        ch = fp.roster("champions", include_megas=True)
        assert [e["display"] for e in ch[:3]] == ["Charizard", "Mega Charizard X", "Mega Charizard Y"]
        fl = [e for e in ch if e["species"] == "floette"]
        assert [e["pokemon"] for e in fl] == ["floette-eternal", "floette-eternal-mega"]
        assert not any("venusaur" in e["pokemon"] and e["form"] == "mega" for e in ch if e["species"] != "venusaur")
        assert fp.move("overdrive")["category"] == "special"
        assert fp.sprite_url({"pokemon": "charizard-mega-x", "species": "charizard"}).endswith("/10001.png")
        assert fp.sprite_url({"pokemon": "sprigatito", "species": "sprigatito"}).endswith("/906.png")
        # cache hit works without the fake API
        FakeProvider._get = lambda self, p: (_ for _ in ()).throw(RuntimeError("no net"))
        assert fp.pokemon(parse_entry("Toxtricity"))["name"] == "toxtricity-amped"


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("✓", k)
