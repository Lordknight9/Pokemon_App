"""Move flags (contact, punch, bite, …) that PokeAPI does not provide.

Primary source: Pokémon Showdown's public move data
(https://play.pokemonshowdown.com/data/moves.json, MIT licensed), downloaded once and
cached on disk by the data provider. If it cannot be reached, the curated fallback
lists below are used (no contact / Sheer Force information in that case).
"""
import re

SHOWDOWN_MOVES_URL = "https://play.pokemonshowdown.com/data/moves.json"
FLAG_KEYS = ("contact", "punch", "bite", "pulse", "slicing", "sound", "bullet", "wind")

FLAG_LABEL = {"contact": "Contact", "punch": "Punch", "bite": "Bite", "pulse": "Pulse", "slicing": "Slicing",
              "sound": "Sound", "bullet": "Ball/Bomb", "wind": "Wind", "secondary": "Secondary effect",
              "recoil": "Recoil"}


def sd_id(slug: str) -> str:
    """PokeAPI slug -> Showdown id ('close-combat' -> 'closecombat')."""
    return re.sub(r"[^a-z0-9]", "", slug.lower())


def parse_showdown(raw: dict) -> dict:
    """Compact {showdown_id: [flags...]} including 'secondary' and 'recoil' pseudo-flags."""
    out = {}
    for mid, m in raw.items():
        if not isinstance(m, dict):
            continue
        fl = [k for k in FLAG_KEYS if (m.get("flags") or {}).get(k)]
        if m.get("secondary") or m.get("secondaries"):
            fl.append("secondary")
        if m.get("recoil") or m.get("hasCrashDamage"):
            fl.append("recoil")
        out[mid] = fl
    return out


_FALLBACK_LISTS = {
    "punch": """bullet-punch comet-punch dizzy-punch double-iron-bash drain-punch dynamic-punch fire-punch focus-punch
        hammer-arm headlong-rush ice-hammer ice-punch jet-punch mach-punch mega-punch meteor-mash plasma-fists
        power-up-punch rage-fist shadow-punch sky-uppercut surging-strikes thunder-punch wicked-blow""",
    "bite": """bite crunch fire-fang fishious-rend hyper-fang ice-fang jaw-lock poison-fang psychic-fangs
        thunder-fang""",
    "pulse": "aura-sphere dark-pulse dragon-pulse heal-pulse origin-pulse terrain-pulse water-pulse",
    "slicing": """aerial-ace air-cutter air-slash aqua-cutter behemoth-blade bitter-blade ceaseless-edge
        cross-poison cut fury-cutter kowtow-cleave leaf-blade mighty-cleave night-slash population-bomb psyblade
        psycho-cut razor-leaf razor-shell sacred-sword secret-sword slash solar-blade stone-axe tachyon-cutter
        x-scissor""",
    "sound": """alluring-voice boomburst bug-buzz chatter clanging-scales clangorous-soul disarming-voice
        echoed-voice eerie-spell grass-whistle growl heal-bell howl hyper-voice metal-sound noble-roar overdrive
        parting-shot perish-song psychic-noise relic-song roar round screech sing snarl snore sparkling-aria
        supersonic torch-song uproar""",
    "bullet": """acid-spray aura-sphere barrage beak-blast bullet-seed egg-bomb electro-ball energy-ball
        focus-blast gyro-ball ice-ball magnet-bomb mist-ball mud-bomb octazooka pollen-puff pyro-ball rock-blast
        rock-wrecker searing-shot seed-bomb shadow-ball sludge-bomb weather-ball zap-cannon""",
    "wind": """air-cutter bleakwind-storm blizzard fairy-wind gust heat-wave hurricane icy-wind petal-blizzard
        sandsear-storm sandstorm springtide-storm tailwind twister whirlwind wildbolt-storm""",
    "recoil": """brave-bird double-edge flare-blitz head-charge head-smash high-jump-kick jump-kick light-of-ruin
        submission supercell-slam take-down volt-tackle wave-crash wild-charge wood-hammer""",
}


def fallback_table() -> dict:
    out = {}
    for flag, txt in _FALLBACK_LISTS.items():
        for slug in txt.split():
            out.setdefault(sd_id(slug), []).append(flag)
    return out
