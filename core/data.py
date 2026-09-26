"""Data layer: PokeAPI (live, cached on disk) and an offline demo provider.

All providers return *normalized* dicts so the rest of the app never touches raw
PokeAPI JSON:

pokemon = {name, display, id, species, types[], base{hp,atk,def,spa,spd,spe},
           abilities[], moves{move_name: [version_groups]}, sprite}
move    = {name, display, type, power, accuracy, pp, priority, category}
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from .rosters import GAMES, parse_entry, static_roster
from .typechart import TYPES

API = "https://pokeapi.co/api/v2"
STAT_KEYS = {"hp": "hp", "attack": "atk", "defense": "def", "special-attack": "spa",
             "special-defense": "spd", "speed": "spe"}


def sort_roster(entries: list) -> list:
    """National dex order; Mega forms right after their base form."""
    return sorted(entries, key=lambda e: (e.get("dex") or 99999, e.get("form") == "mega", e["display"]))


class NotFound(Exception):
    pass


def pretty(slug: str) -> str:
    return " ".join(p.capitalize() for p in slug.split("-"))


class LiveProvider:
    """PokeAPI with a JSON disk cache (normalized data only, so it stays small)."""

    name = "PokeAPI"

    def __init__(self, cache_dir: str | os.PathLike = ".cache/pokeapi", timeout: int = 20):
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.s = requests.Session()
        self.s.headers["User-Agent"] = "pokemon-mcda-streamlit/1.0 (educational)"

    # ---------- low level ----------
    def _cpath(self, key: str) -> Path:
        return self.cache / (key.replace("/", "__") + ".json")

    def _cached(self, key: str, builder):
        p = self._cpath(key)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
        val = builder()
        p.write_text(json.dumps(val, ensure_ascii=False), encoding="utf-8")
        return val

    def _get(self, path: str) -> dict:
        url = f"{API}/{path.strip('/')}" + ("" if "?" in path else "/")
        last = None
        for attempt in range(3):
            try:
                r = self.s.get(url, timeout=self.timeout)
                if r.status_code == 404:
                    raise NotFound(path)
                r.raise_for_status()
                return r.json()
            except NotFound:
                raise
            except Exception as e:  # network hiccup -> retry
                last = e
                time.sleep(1 + attempt)
        raise ConnectionError(f"PokeAPI unreachable for {path}: {last}")

    def ping(self) -> bool:
        try:
            self._get("type/1")
            return True
        except Exception:
            return False

    # ---------- species / pokemon ----------
    def species(self, sp: str) -> dict:
        def build():
            j = self._get(f"pokemon-species/{sp}")
            return {"name": j["name"],
                    "varieties": [v["pokemon"]["name"] for v in j["varieties"]],
                    "default": next(v["pokemon"]["name"] for v in j["varieties"] if v["is_default"])}
        return self._cached(f"species/{sp}", build)

    def _normalize_pokemon(self, j: dict, display: str | None = None) -> dict:
        moves = {}
        for m in j["moves"]:
            moves[m["move"]["name"]] = sorted({d["version_group"]["name"] for d in m["version_group_details"]})
        spr = j.get("sprites") or {}
        art = ((spr.get("other") or {}).get("official-artwork") or {}).get("front_default")
        return {
            "name": j["name"], "display": display or pretty(j["name"]), "id": j["id"],
            "species": j["species"]["name"],
            "types": [t["type"]["name"] for t in sorted(j["types"], key=lambda t: t["slot"])],
            "base": {STAT_KEYS[s["stat"]["name"]]: s["base_stat"] for s in j["stats"]},
            "abilities": [a["ability"]["name"] for a in sorted(j["abilities"], key=lambda a: a["slot"])],
            "moves": moves,
            "sprite": art or spr.get("front_default"),
        }

    def pokemon(self, entry) -> dict:
        """entry: roster dict (from parse_entry) or a plain name/slug."""
        if isinstance(entry, str):
            entry = parse_entry(entry) if " " in entry or entry[:1].isupper() else \
                {"display": pretty(entry), "species": entry, "pokemon": entry, "form": None}

        def build():
            try:
                j = self._get(f"pokemon/{entry['pokemon']}")
            except NotFound:
                sp = self.species(entry["species"])
                target = sp["default"]
                if entry.get("form"):
                    cand = [v for v in sp["varieties"] if entry["form"] in v]
                    if cand:
                        target = cand[0]
                j = self._get(f"pokemon/{target}")
            p = self._normalize_pokemon(j, entry["display"])
            if not p["moves"]:  # Mega / battle-only forms have no learnset -> borrow base form's
                try:
                    base = self.species(p["species"])["default"]
                    if base != p["name"]:
                        p["moves"] = self._normalize_pokemon(self._get(f"pokemon/{base}"))["moves"]
                except Exception:
                    pass
            return p
        return self._cached(f"pokemon/{entry['pokemon']}", build)

    def pokemon_many(self, entries, workers: int = 8) -> list[dict]:
        with ThreadPoolExecutor(workers) as ex:
            return list(ex.map(self._safe_pokemon, entries))

    def _safe_pokemon(self, e):
        try:
            return self.pokemon(e)
        except Exception as ex:  # keep going, report later
            return {"error": str(ex), "display": e["display"] if isinstance(e, dict) else e}

    # ---------- rosters ----------
    def species_ids(self) -> dict:
        """{species_slug: national dex number} — one API call, cached."""
        def build():
            j = self._get("pokemon-species?limit=3000")
            return {r["name"]: int(r["url"].rstrip("/").split("/")[-1]) for r in j["results"]}
        return self._cached("species_ids", build)

    def all_pokemon_names(self) -> list:
        """Every pokemon/form slug known to PokeAPI (incl. megas) — one call, cached."""
        return self._cached("pokemon_names",
                            lambda: [r["name"] for r in self._get("pokemon?limit=5000")["results"]])

    def roster(self, game: str, include_megas: bool = False) -> list[dict]:
        g = GAMES[game]
        if g["pokedexes"]:
            def build():
                seen, out = set(), []
                for dex in g["pokedexes"]:
                    j = self._get(f"pokedex/{dex}")
                    for pe in j["pokemon_entries"]:
                        sp = pe["pokemon_species"]["name"]
                        if sp not in seen:
                            seen.add(sp)
                            out.append({"display": pretty(sp), "species": sp, "pokemon": sp, "form": None})
                return out
            entries = self._cached(f"roster/{game}", build)
        else:
            entries = static_roster(game)
        entries = [dict(e) for e in entries]
        if include_megas:
            entries += self.mega_entries(entries)
        try:
            ids = self.species_ids()
        except Exception:
            ids = {}
        for e in entries:
            e["dex"] = ids.get(e["species"])
        return sort_roster(entries)

    def mega_entries(self, entries) -> list[dict]:
        species = {e["species"] for e in entries}
        try:
            names = self.all_pokemon_names()
        except Exception:
            return []
        out = []
        for n in names:
            if "-mega" not in n:
                continue
            base, _, suffix = n.partition("-mega")
            parts = base.split("-")
            sp = next(("-".join(parts[:k]) for k in range(len(parts), 0, -1)
                       if "-".join(parts[:k]) in species), None)
            if not sp:
                continue
            disp = "Mega " + pretty(base) + ((" " + pretty(suffix.strip("-"))) if suffix.strip("-") else "")
            out.append({"display": disp, "species": sp, "pokemon": n, "form": "mega"})
        return out

    # ---------- moves ----------
    def move(self, name: str) -> dict:
        def build():
            j = self._get(f"move/{name}")
            return {"name": j["name"], "display": pretty(j["name"]), "type": j["type"]["name"],
                    "power": j.get("power"), "accuracy": j.get("accuracy"), "pp": j.get("pp"),
                    "priority": j.get("priority", 0), "category": j["damage_class"]["name"]}
        return self._cached(f"move/{name}", build)

    def move_index(self) -> dict:
        """{move: {'type','category'}} from 18 type + 3 damage-class calls (cheap)."""
        def build():
            idx = {}
            for t in TYPES:
                for m in self._get(f"type/{t}")["moves"]:
                    idx.setdefault(m["name"], {})["type"] = t
            for c in ("physical", "special", "status"):
                for m in self._get(f"move-damage-class/{c}")["moves"]:
                    idx.setdefault(m["name"], {})["category"] = c
            return idx
        return self._cached("move_index", build)

    # ---------- helpers ----------
    def learnset(self, pkmn: dict, game: str) -> list[str]:
        """Moves for this game; falls back to SV learnset, then to every move."""
        for vgs in (GAMES[game]["version_groups"], GAMES["sv"]["version_groups"]):
            ms = [m for m, g in pkmn["moves"].items() if set(g) & set(vgs)]
            if ms:
                return sorted(ms)
        return sorted(pkmn["moves"])


class DemoProvider(LiveProvider):
    """Offline provider with a small built-in dataset (for demos / no internet)."""

    name = "Demo (offline)"

    def __init__(self):
        from . import demo_data
        self.d = demo_data

    def ping(self):
        return True

    def roster(self, game, include_megas=False):
        return sort_roster([{"display": v["display"], "species": k, "pokemon": k, "form": None,
                             "dex": v.get("dex")} for k, v in self.d.POKEMON.items()])

    def pokemon(self, entry):
        key = entry["pokemon"] if isinstance(entry, dict) else entry
        if key not in self.d.POKEMON:
            raise NotFound(key)
        p = dict(self.d.POKEMON[key])
        p.update({"name": key, "species": key, "id": 0, "sprite": None,
                  "moves": {m: ["demo"] for m in p["moves"]}})
        return p

    def pokemon_many(self, entries, workers=1):
        return [self._safe_pokemon(e) for e in entries]

    def move(self, name):
        t, c, pw = self.d.MOVES[name]
        return {"name": name, "display": pretty(name), "type": t, "power": pw, "accuracy": 100,
                "pp": 10, "priority": 1 if name in ("extreme-speed", "sucker-punch", "fake-out") else 0,
                "category": c}

    def move_index(self):
        return {k: {"type": t, "category": c} for k, (t, c, _) in self.d.MOVES.items()}

    def learnset(self, pkmn, game):
        return sorted(pkmn["moves"])
