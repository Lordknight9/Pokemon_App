"""Real VGC usage data from Smogon / Pokémon Showdown ladder statistics.

https://www.smogon.com/stats/<YYYY-MM>/chaos/<format>-<cutoff>.json
Each Pokémon has weighted counts of abilities, items, moves, spreads and *teammates*
(how often two Pokémon appear on the same team). That co-occurrence is the ground truth
we use for "synergy" and as the training target of the ML model.
"""
from __future__ import annotations

import math
import re

STATS_ROOT = "https://www.smogon.com/stats"

# game -> regex of the Showdown VGC formats to look for (newest first)
FORMAT_PATTERNS = {
    "champions": r"gen9championsvgc\d{4}reg[a-z0-9]+?",
    "sv": r"gen9vgc\d{4}reg[a-z0-9]+?",
    "za": r"gen9championsvgc\d{4}reg[a-z0-9]+?",   # no Z-A ladder: use Champions as proxy
}
CUTOFFS = ["1760", "1630", "1500", "0"]


def sd_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def latest_months(index_html: str, n: int = 6) -> list[str]:
    return sorted(set(re.findall(r"(\d{4}-\d{2})/", index_html)), reverse=True)[:n]


def formats_in(listing_html: str, pattern: str) -> list[str]:
    """Format names (without cutoff) matching pattern, excluding best-of-3 variants, newest first."""
    names = set(re.findall(rf"({pattern})-\d+\.json", listing_html))
    names = {n for n in names if not n.endswith("bo3")}
    return sorted(names, reverse=True)


def compact_chaos(raw: dict, top_moves: int = 12) -> dict:
    """Keep only what the app needs from a (large) chaos JSON."""
    data = raw.get("data", {})
    out = {}
    for name, d in data.items():
        w = sum((d.get("Abilities") or {}).values()) or sum((d.get("Items") or {}).values())
        if w <= 0:
            continue
        mv = sorted((d.get("Moves") or {}).items(), key=lambda kv: -kv[1])[:top_moves]
        it = sorted((d.get("Items") or {}).items(), key=lambda kv: -kv[1])[:6]
        ab = sorted((d.get("Abilities") or {}).items(), key=lambda kv: -kv[1])[:3]
        sp = sorted((d.get("Spreads") or {}).items(), key=lambda kv: -kv[1])[:3]
        out[name] = {
            "w": w,
            "moves": {k: v / w for k, v in mv if k},
            "items": {k: v / w for k, v in it if k},
            "abilities": {k: v / w for k, v in ab if k},
            "spreads": {k: v / w for k, v in sp},
            "teammates": {k: v for k, v in (d.get("Teammates") or {}).items() if v},
        }
    return {"info": raw.get("info", {}), "mons": out}


class Usage:
    """Co-occurrence statistics on names resolved to the app's roster display names."""

    def __init__(self, compact: dict, resolve):
        """resolve(showdown_name) -> roster display name or None."""
        self.info = compact.get("info", {})
        mons = compact.get("mons", {})
        self.name_of = {}
        for sd in mons:
            r = resolve(sd)
            if r:
                self.name_of[sd] = r
        self.w, self.detail = {}, {}
        for sd, d in mons.items():
            r = self.name_of.get(sd)
            if not r:
                continue
            self.w[r] = self.w.get(r, 0) + d["w"]
            if r not in self.detail or d["w"] > self.detail[r]["w"]:
                self.detail[r] = d
        self.total_teams = sum(v["w"] for v in mons.values()) / 6 or 1.0
        # joint weighted counts; chaos 'Teammates' may be raw counts or (older) deltas vs expectation
        delta = any(v < 0 for d in mons.values() for v in d["teammates"].values())
        self.joint = {}
        for sd, d in mons.items():
            a = self.name_of.get(sd)
            if not a:
                continue
            for sd2, v in d["teammates"].items():
                b = self.name_of.get(sd2)
                if not b or b == a:
                    continue
                if delta:
                    v = v + d["w"] * mons.get(sd2, {}).get("w", 0) / self.total_teams
                key = frozenset((a, b))
                # each pair is listed from both sides; keep the max (they should match)
                self.joint[key] = max(self.joint.get(key, 0.0), max(v, 0.0))

    def usage(self, a: str) -> float:
        return self.w.get(a, 0.0) / self.total_teams

    def teammate_pct(self, a: str, b: str) -> float:
        """P(b on team | a on team)."""
        wa = self.w.get(a, 0)
        return self.joint.get(frozenset((a, b)), 0.0) / wa if wa else 0.0

    def log_lift(self, a: str, b: str, m: float | None = None) -> float | None:
        """Smoothed log2( observed / expected co-occurrence ); 0 = independent, >0 = paired more often.

        m = pseudo-count (default 0.2% of teams) pulls rare pairs towards 0.
        """
        if a not in self.w or b not in self.w:
            return None
        m = m if m is not None else max(1e-9, 0.002 * self.total_teams)
        pa, pb = self.usage(a), self.usage(b)
        expected = pa * pb * self.total_teams
        obs = self.joint.get(frozenset((a, b)), 0.0)
        return math.log2((obs + m) / (expected + m))

    def top(self, n: int | None = None) -> list[str]:
        return sorted(self.w, key=lambda k: -self.w[k])[:n]
