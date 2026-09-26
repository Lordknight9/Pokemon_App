"""Pokémon MCDA Lab — Streamlit app.

Σελίδες:
  1. Pokédex & Stats         (PokeAPI: base stats, stats ανά level, τύποι, κινήσεις)
  2. Damage Calculator       (Gen 9 formula)
  3. Κατάταξη Pokémon        (TOPSIS / PROMETHEE II)
  4. Synergy & τετράδες      (dataset συνέργειας + κατάταξη 4άδων από 6)
  5. Μεθοδολογία

Τρέξιμο:  streamlit run app.py
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import streamlit as st

from core.analysis import (CRITERIA, QUAD_CRITERIA, SYN_LABEL, SYN_WEIGHTS, build_profile, damage_matrix,
                           quad_criteria, single_criteria, synergy_dataset, synergy_matrix)
from core.damage import ITEMS, Battler, Field, Move, calc, ko_text
from core.data import DemoProvider, LiveProvider, pretty
from core.mcda import PREF_FUNCS, promethee, spearman, topsis
from core.rosters import GAMES
from core.stats import NATURES, SPREADS, STAT_LABEL, STATS, calc_all, spread_for
from core.typechart import TYPES, defensive_profile

st.set_page_config(page_title="Pokémon MCDA Lab", page_icon="⚔️", layout="wide")

TYPE_COLOR = {
    "normal": "#A8A77A", "fire": "#EE8130", "water": "#6390F0", "electric": "#F7D02C", "grass": "#7AC74C",
    "ice": "#96D9D6", "fighting": "#C22E28", "poison": "#A33EA1", "ground": "#E2BF65", "flying": "#A98FF3",
    "psychic": "#F95587", "bug": "#A6B91A", "rock": "#B6A136", "ghost": "#735797", "dragon": "#6F35FC",
    "dark": "#705746", "steel": "#B7B7CE", "fairy": "#D685AD",
}
POPULAR = ["Garchomp", "Incineroar", "Rillaboom", "Amoonguss", "Dragonite", "Gholdengo", "Kingambit",
           "Tyranitar", "Excadrill", "Pelipper", "Torkoal", "Gardevoir", "Corviknight", "Sneasler",
           "Dragapult", "Whimsicott", "Sinistcha", "Archaludon", "Milotic", "Charizard", "Venusaur",
           "Baxcalibur", "Kommo-o", "Abomasnow"]


def badge(t: str) -> str:
    return (f"<span style='background:{TYPE_COLOR.get(t, '#888')};color:white;padding:2px 10px;"
            f"border-radius:10px;margin-right:4px;font-size:0.85em;font-weight:600'>{t.capitalize()}</span>")


# ============================================================ data access (cached)
@st.cache_resource
def get_provider(source: str):
    return DemoProvider() if source == "demo" else LiveProvider()


@st.cache_data(show_spinner=False)
def api_ok(source: str) -> bool:
    return get_provider(source).ping()


@st.cache_data(show_spinner="Φόρτωση λίστας Pokémon…")
def get_roster(source: str, game: str, megas: bool) -> list:
    return get_provider(source).roster(game, include_megas=megas)


@st.cache_data(show_spinner=False)
def get_pokemon(source: str, entry: dict) -> dict:
    return get_provider(source).pokemon(entry)


@st.cache_data(show_spinner="Λήψη δεδομένων από PokeAPI…")
def get_many(source: str, entries: tuple) -> list:
    return get_provider(source).pokemon_many([dict(e) for e in entries])


@st.cache_data(show_spinner="Λήψη πίνακα κινήσεων (μία φορά)…")
def get_move_index(source: str) -> dict:
    return get_provider(source).move_index()


@st.cache_data(show_spinner=False)
def get_move(source: str, name: str) -> dict:
    return get_provider(source).move(name)


def as_key(e: dict) -> tuple:
    return tuple(sorted(e.items()))


def load_entries(entries: list) -> list:
    res = get_many(SRC, tuple(as_key(e) for e in entries))
    bad = [r["display"] for r in res if "error" in r]
    if bad:
        st.warning("Δεν βρέθηκαν στο PokeAPI (παραλείπονται): " + ", ".join(bad))
    return [r for r in res if "error" not in r]


# ============================================================ sidebar
st.sidebar.title("⚔️ Pokémon MCDA Lab")
src_label = st.sidebar.radio("Πηγή δεδομένων", ["PokeAPI (online)", "Demo (offline, 24 Pokémon)"],
                             help="Το Demo λειτουργεί χωρίς internet με μικρό ενσωματωμένο dataset.")
SRC = "demo" if src_label.startswith("Demo") else "live"
if SRC == "live" and not api_ok("live"):
    st.sidebar.error("Το PokeAPI δεν απαντά. Έλεγξε τη σύνδεση ή διάλεξε Demo.")

GAME = st.sidebar.selectbox("Παιχνίδι", list(GAMES), format_func=lambda g: GAMES[g]["label"])
G = GAMES[GAME]
LEVEL = st.sidebar.slider("Level", 1, 100, G["default_level"])
MEGAS = st.sidebar.checkbox("Συμπερίληψη Mega μορφών", value=False, disabled=(GAME == "sv" or SRC == "demo"),
                            help="Προσθέτει τις Mega μορφές που υπάρχουν στο PokeAPI (Z-A / Champions).")
PAGE = st.sidebar.radio("Σελίδα", ["📘 Pokédex & Stats", "💥 Damage Calculator", "🏆 Κατάταξη Pokémon",
                                   "🤝 Synergy & τετράδες", "ℹ️ Μεθοδολογία"])
st.sidebar.caption("Δεδομένα: PokeAPI (pokeapi.co). Rosters Z-A/Champions: ενσωματωμένες λίστες, Σεπτ. 2026.")

try:
    ROSTER = get_roster(SRC, GAME, MEGAS)
except Exception as ex:
    st.error(f"Αποτυχία φόρτωσης roster: {ex}")
    st.stop()
BY_NAME = {e["display"]: e for e in ROSTER}
NAMES = list(BY_NAME)


def default_pick(n: int) -> list:
    pop = [x for x in POPULAR if x in BY_NAME]
    return (pop + [x for x in NAMES if x not in pop])[:n]


def synced_multiselect(label: str, store_key: str, default: list, **kw) -> list:
    """Multiselect whose value survives page switches and game changes."""
    wkey = "w_" + store_key
    if wkey in st.session_state:
        st.session_state[wkey] = [x for x in st.session_state[wkey] if x in BY_NAME]
    else:
        st.session_state[wkey] = [x for x in (st.session_state.get(store_key) or default) if x in BY_NAME]
    val = st.multiselect(label, NAMES, key=wkey, **kw)
    st.session_state[store_key] = val
    return val


def damaging_learnset(p: dict) -> list:
    mi = get_move_index(SRC)
    ls = get_provider(SRC).learnset(p, GAME)
    return [m for m in ls if mi.get(m, {}).get("category") in ("physical", "special")]


# ============================================================ criteria editor (shared)
def criteria_editor(key: str, spec: list) -> tuple:
    """spec: list of (key, label, benefit, weight, used). Returns (keys, weights, benefit, funcs)."""
    df = pd.DataFrame({
        "Χρήση": [s[4] if len(s) > 4 else True for s in spec],
        "Κριτήριο": [s[1] for s in spec],
        "Βάρος": [float(s[3]) for s in spec],
        "Κατεύθυνση": ["max" if s[2] else "min" for s in spec],
        "Συνάρτηση PROMETHEE": ["linear"] * len(spec),
        "q": [np.nan] * len(spec),
        "p": [np.nan] * len(spec),
    })
    ed = st.data_editor(
        df, key=key, hide_index=True, width="stretch",
        column_config={
            "Κριτήριο": st.column_config.TextColumn(disabled=True),
            "Βάρος": st.column_config.NumberColumn(min_value=0.0, max_value=10.0, step=0.5),
            "Κατεύθυνση": st.column_config.SelectboxColumn(options=["max", "min"]),
            "Συνάρτηση PROMETHEE": st.column_config.SelectboxColumn(options=list(PREF_FUNCS)),
            "q": st.column_config.NumberColumn(help="Κατώφλι αδιαφορίας (κενό = 0)"),
            "p": st.column_config.NumberColumn(help="Κατώφλι αυστηρής προτίμησης (κενό = τυπ. απόκλιση)"),
        })
    use = ed["Χρήση"].to_numpy(dtype=bool)
    keys = [s[0] for s, u in zip(spec, use) if u]
    w = ed.loc[use, "Βάρος"].astype(float).tolist()
    ben = (ed.loc[use, "Κατεύθυνση"] == "max").tolist()
    funcs = [{"kind": r["Συνάρτηση PROMETHEE"], "q": r["q"], "p": r["p"], "s": None}
             for _, r in ed.loc[use].iterrows()]
    return keys, w, ben, funcs


def run_mcda(X: pd.DataFrame, w, ben, funcs, method: str) -> pd.DataFrame:
    out = pd.DataFrame(index=X.index)
    if method in ("TOPSIS", "Και τα δύο"):
        out = out.join(topsis(X, w, ben))
    if method in ("PROMETHEE II", "Και τα δύο"):
        out = out.join(promethee(X, w, ben, funcs))
    sort_col = "Κατάταξη TOPSIS" if "Κατάταξη TOPSIS" in out else "Κατάταξη PROMETHEE"
    return out.sort_values(sort_col)


def show_ranking(res: pd.DataFrame, method: str, label: str):
    score_col = "TOPSIS C*" if "TOPSIS C*" in res else "Φ (net)"
    c1, c2 = st.columns([3, 2])
    with c1:
        st.dataframe(res.round(4), width="stretch")
    with c2:
        st.bar_chart(res[score_col].sort_values(ascending=True), horizontal=True)
    if method == "Και τα δύο":
        rho = spearman(res["Κατάταξη TOPSIS"], res["Κατάταξη PROMETHEE"])
        st.info(f"Συσχέτιση Spearman ανάμεσα σε TOPSIS και PROMETHEE: **ρ = {rho:.3f}**")
    st.download_button(f"⬇️ Αποτελέσματα {label} (CSV)", res.to_csv().encode("utf-8-sig"),
                       file_name=f"ranking_{label}.csv", mime="text/csv")


# ============================================================ page 1: Pokédex
def page_pokedex():
    st.header("📘 Pokédex & Stats")
    st.caption(f"{G['label']} — {len(NAMES)} Pokémon στη λίστα")
    name = st.selectbox("Pokémon", NAMES, index=NAMES.index(default_pick(1)[0]))
    try:
        p = get_pokemon(SRC, BY_NAME[name])
    except Exception as ex:
        st.error(f"Σφάλμα: {ex}")
        return
    c1, c2 = st.columns([1, 2])
    with c1:
        if p.get("sprite"):
            st.image(p["sprite"], width=220)
        st.markdown(" ".join(badge(t) for t in p["types"]), unsafe_allow_html=True)
        if G["has_abilities"]:
            st.write("**Abilities:** " + ", ".join(pretty(a) for a in p["abilities"]))
        else:
            st.caption("Στο Legends Z-A δεν υπάρχουν abilities στη μάχη.")
        st.metric("Base Stat Total", sum(p["base"].values()))
    with c2:
        st.subheader(f"Stats στο Level {LEVEL}")
        a, b = st.columns(2)
        nature = a.selectbox("Nature", list(NATURES), index=list(NATURES).index("Hardy"))
        spread = b.selectbox("Έτοιμο spread", ["custom"] + list(SPREADS),
                             format_func=lambda k: "Custom EVs" if k == "custom" else SPREADS[k])
        evs, ivs = {}, {}
        with st.expander("EVs / IVs"):
            cols = st.columns(6)
            for i, s in enumerate(STATS):
                evs[s] = cols[i].number_input(f"EV {STAT_LABEL[s]}", 0, 252, 0, 4, key=f"ev_{s}")
                ivs[s] = cols[i].number_input(f"IV {STAT_LABEL[s]}", 0, 31, 31, key=f"iv_{s}")
        if spread != "custom":
            evs, nature = spread_for(p["base"], spread)
        final = calc_all(p["base"], LEVEL, ivs, evs, nature)
        tbl = pd.DataFrame({"Base": p["base"], f"Lv {LEVEL}": final,
                            "Lv 100 (max)": calc_all(p["base"], 100, None, {s: 252 for s in STATS})})
        tbl.index = [STAT_LABEL[s] for s in tbl.index]
        st.dataframe(tbl, width="stretch")
        st.bar_chart(tbl[[f"Lv {LEVEL}"]], horizontal=True)

    st.subheader("Αμυντικό προφίλ τύπων")
    prof = defensive_profile(p["types"], p["abilities"][0] if (p["abilities"] and G["has_abilities"]) else None)
    groups = {"×4": [], "×2": [], "×½": [], "×¼": [], "×0": []}
    for t, m in prof.items():
        k = {4: "×4", 2: "×2", .5: "×½", .25: "×¼", 0: "×0"}.get(m)
        if k:
            groups[k].append(t)
    for k, ts in groups.items():
        if ts:
            st.markdown(f"**{k}** " + " ".join(badge(t) for t in ts), unsafe_allow_html=True)

    st.subheader("Κινήσεις (learnset για το παιχνίδι)")
    mi = get_move_index(SRC)
    ls = get_provider(SRC).learnset(p, GAME)
    mv = pd.DataFrame([{"Κίνηση": pretty(m), "Τύπος": mi.get(m, {}).get("type", "?"),
                        "Κατηγορία": mi.get(m, {}).get("category", "?")} for m in ls])
    if st.checkbox("Φόρτωση power/accuracy για όλες τις κινήσεις (περισσότερα requests)"):
        details = [get_move(SRC, m) for m in ls]
        mv["Power"] = [d["power"] for d in details]
        mv["Accuracy"] = [d["accuracy"] for d in details]
        mv["Priority"] = [d["priority"] for d in details]
    st.dataframe(mv, width="stretch", hide_index=True, height=320)
    if GAME != "sv":
        st.caption("Αν το PokeAPI δεν έχει ακόμα learnset για αυτό το παιχνίδι, εμφανίζεται το learnset του "
                   "Scarlet/Violet (ή όλες οι κινήσεις).")


# ============================================================ page 2: damage calc
def battler_panel(col, side: str, default_name: str):
    with col:
        st.subheader("Επιτιθέμενος" if side == "a" else "Αμυνόμενος")
        name = st.selectbox("Pokémon", NAMES, index=NAMES.index(default_name), key=f"{side}_name")
        p = get_pokemon(SRC, BY_NAME[name])
        st.markdown(" ".join(badge(t) for t in p["types"]), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        lvl = c1.number_input("Level", 1, 100, LEVEL, key=f"{side}_lvl")
        nature = c2.selectbox("Nature", list(NATURES), key=f"{side}_nat",
                              index=list(NATURES).index("Adamant" if side == "a" else "Hardy"))
        ability = None
        if G["has_abilities"] and p["abilities"]:
            ability = c1.selectbox("Ability", p["abilities"], format_func=pretty, key=f"{side}_ab")
        item = c2.selectbox("Item", ITEMS, key=f"{side}_item")
        item = None if item.startswith("(") else item
        tera = None
        if G["has_tera"]:
            t = c1.selectbox("Tera type", ["—"] + TYPES, key=f"{side}_tera")
            tera = None if t == "—" else t
        with st.expander("EVs / IVs / Boosts"):
            evs, ivs, boosts = {}, {}, {}
            cc = st.columns(6)
            for i, s in enumerate(STATS):
                d_ev = 252 if (side == "a" and s in ("atk", "spe")) or (side == "d" and s == "hp") else 0
                evs[s] = cc[i].number_input(f"EV {STAT_LABEL[s]}", 0, 252, d_ev, 4, key=f"{side}_ev_{s}")
                ivs[s] = cc[i].number_input(f"IV {STAT_LABEL[s]}", 0, 31, 31, key=f"{side}_iv_{s}")
                if s != "hp":
                    boosts[s] = cc[i].number_input(f"Boost {STAT_LABEL[s]}", -6, 6, 0, key=f"{side}_b_{s}")
        stats = calc_all(p["base"], lvl, ivs, evs, nature)
        burned, hp_pct = False, 100.0
        if side == "a":
            burned = st.checkbox("Burned", key="a_burn")
        else:
            hp_pct = st.slider("Τρέχον HP %", 1, 100, 100, key="d_hp")
        st.caption(" · ".join(f"{STAT_LABEL[s]} {stats[s]}" for s in STATS))
        b = Battler(p["display"], p["types"], stats, lvl, ability, item, boosts, tera, burned, hp_pct)
        return p, b


def page_damage():
    st.header("💥 Damage Calculator")
    st.caption("Τύπος ζημιάς Gen 9 (όπως Showdown / SV). 16 random rolls 85%–100%.")
    d1, d2 = default_pick(2)
    ca, cd = st.columns(2)
    pa, A = battler_panel(ca, "a", d1)
    pd_, D = battler_panel(cd, "d", d2)

    st.divider()
    st.subheader("Κίνηση & πεδίο μάχης")
    moves = damaging_learnset(pa)
    if not moves:
        st.warning("Δεν βρέθηκαν επιθετικές κινήσεις.")
        return
    c1, c2, c3, c4 = st.columns(4)
    mname = c1.selectbox("Κίνηση", moves, format_func=pretty)
    md = get_move(SRC, mname)
    power = c2.number_input("Power", 1, 300, int(md["power"] or 60),
                            help="Αλλάξτε για κινήσεις με μεταβλητή δύναμη.")
    fmt = c3.radio("Format", ["Doubles", "Singles"], horizontal=True)
    spread = c4.checkbox("Spread move (χτυπά 2 στόχους)", value=False, disabled=(fmt == "Singles"))
    f1, f2, f3, f4, f5 = st.columns(5)
    weather = f1.selectbox("Καιρός", ["—", "sun", "rain", "sand", "snow"])
    terrain = f2.selectbox("Terrain", ["—", "electric", "grassy", "psychic", "misty"])
    crit = f3.checkbox("Critical hit")
    reflect = f4.checkbox("Reflect")
    ls = f5.checkbox("Light Screen")
    fld = Field(None if weather == "—" else weather, None if terrain == "—" else terrain, crit,
                fmt == "Doubles", reflect, ls)
    mv = Move(mname, md["type"], power, md["category"], spread)
    r = calc(A, D, mv, fld)
    hp = D.stats["hp"]
    cur = max(1, round(hp * D.hp_pct / 100))

    st.markdown(f"### {A.name} — **{pretty(mname)}** {badge(md['type'])} ({md['category']}, {power} BP) "
                f"→ {D.name}", unsafe_allow_html=True)
    if r["max"] == 0:
        st.error(r["note"] or "Καμία ζημιά")
        return
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Ζημιά", f"{r['min']} – {r['max']}")
    m2.metric("% του HP", f"{r['min'] / hp * 100:.1f}% – {r['max'] / hp * 100:.1f}%")
    m3.metric("Αποτελεσματικότητα", f"×{r['eff']:g}")
    m4.metric("STAB", f"×{r['stab']:g}")
    st.success(ko_text(r["rolls"], cur) + (f" (από {D.hp_pct:.0f}% HP)" if D.hp_pct < 100 else ""))
    st.progress(min(1.0, r["max"] / cur), text=f"Μέγιστη ζημιά σε σχέση με το τρέχον HP ({cur}/{hp})")
    st.caption("Rolls: " + ", ".join(map(str, r["rolls"])))

    if st.checkbox("Σύγκριση όλων των επιθετικών κινήσεων του επιτιθέμενου"):
        rows = []
        for m in moves:
            d = get_move(SRC, m)
            if not d["power"]:
                continue
            rr = calc(A, D, Move(m, d["type"], d["power"], d["category"], spread), fld)
            rows.append({"Κίνηση": pretty(m), "Τύπος": d["type"], "Κατ.": d["category"], "Power": d["power"],
                         "Min %": rr["min"] / hp * 100, "Max %": rr["max"] / hp * 100,
                         "KO": ko_text(rr["rolls"], cur) if rr["max"] else "—"})
        st.dataframe(pd.DataFrame(rows).sort_values("Max %", ascending=False).round(1),
                     width="stretch", hide_index=True)


# ============================================================ page 3: ranking
def profiles_for(names: list, spread: str, cov: str) -> list:
    ps = load_entries([BY_NAME[n] for n in names])
    mi = get_move_index(SRC)
    prov = get_provider(SRC)
    return [build_profile(p, LEVEL, spread, prov.learnset(p, GAME), mi, cov, G["has_abilities"]) for p in ps]


def page_ranking():
    st.header("🏆 Κατάταξη Pokémon με TOPSIS / PROMETHEE II")
    names = synced_multiselect("Υποψήφια Pokémon (εναλλακτικές)", "rank_names", default_pick(12))
    if len(names) < 3:
        st.info("Διάλεξε τουλάχιστον 3 Pokémon.")
        return
    if len(names) > 60:
        st.warning("Πολλά Pokémon → η πρώτη λήψη από το PokeAPI θα αργήσει (μετά μένουν στην cache).")
    c1, c2, c3, c4 = st.columns(4)
    spread = c1.selectbox("Spread", list(SPREADS), format_func=lambda k: SPREADS[k])
    power = c2.number_input("Power γενικής STAB κίνησης", 40, 150, 90,
                            help="Για τον πίνακα ζημιάς κάθε Pokémon χτυπά με την καλύτερη STAB του.")
    cov = c3.selectbox("Κάλυψη βάσει", ["learnset", "stab"],
                       format_func=lambda k: "όλων των επιθετικών κινήσεων" if k == "learnset" else "μόνο STAB")
    method = c4.radio("Μέθοδος", ["TOPSIS", "PROMETHEE II", "Και τα δύο"], index=2)

    profiles = profiles_for(names, spread, cov)
    dmg = damage_matrix(profiles, power)
    X = single_criteria(profiles, dmg)

    st.subheader("Κριτήρια & βάρη")
    keys, w, ben, funcs = criteria_editor("crit_single", CRITERIA)
    if not keys:
        st.warning("Επίλεξε τουλάχιστον ένα κριτήριο.")
        return
    labels = {c[0]: c[1] for c in CRITERIA}
    Xs = X[keys].rename(columns=labels)
    with st.expander("Πίνακας αποφάσεων (decision matrix)"):
        st.dataframe(Xs.round(2), width="stretch")
        st.download_button("⬇️ Decision matrix (CSV)", Xs.to_csv().encode("utf-8-sig"), "decision_matrix.csv")
    with st.expander("Πίνακας ζημιάς: γραμμή = επιτιθέμενος, στήλη = αμυνόμενος (% HP, μέσος όρος rolls)"):
        st.dataframe(dmg.round(1).style.background_gradient(cmap="Reds", axis=None).format("{:.1f}"),
                     width="stretch")

    st.subheader("Αποτελέσματα")
    res = run_mcda(Xs, w, ben, funcs, method)
    show_ranking(res, method, "pokemon")
    score = res["TOPSIS C*"] if "TOPSIS C*" in res else \
        (res["Φ (net)"] - res["Φ (net)"].min()) / max(res["Φ (net)"].max() - res["Φ (net)"].min(), 1e-9)
    st.session_state["indiv_scores"] = score.to_dict()
    st.session_state["team_suggest"] = list(res.index[:6])
    st.caption("Τα top-6 προτείνονται ως ομάδα στη σελίδα 🤝 Synergy & τετράδες.")


# ============================================================ page 4: synergy & quads
def page_synergy():
    st.header("🤝 Synergy & κατάταξη τετράδων (4 από 6)")
    sugg = st.session_state.get("team_suggest")
    if sugg and st.button(f"Χρήση των top-6 της κατάταξης: {', '.join(sugg)}"):
        st.session_state["w_team_names"] = [n for n in sugg if n in BY_NAME]
    team_names = synced_multiselect("Ομάδα 6 Pokémon", "team_names", (sugg or default_pick(6))[:6],
                                    max_selections=6)
    if len(team_names) != 6:
        st.info("Διάλεξε ακριβώς 6 Pokémon (ή τρέξε πρώτα την Κατάταξη — τα top-6 μπαίνουν αυτόματα).")
        return
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("**Βάρη συνιστωσών συνέργειας ζεύγους**")
        wc = st.columns(5)
        sw = {k: wc[i].slider(SYN_LABEL[k], 0.0, 1.0, float(SYN_WEIGHTS[k] if (k != "abilities" or
                              G["has_abilities"]) else 0.0), 0.05, key=f"sw_{k}")
              for i, k in enumerate(SYN_WEIGHTS)}
    with c2:
        spread = st.selectbox("Spread", list(SPREADS), format_func=lambda k: SPREADS[k], key="syn_spread")
        method = st.radio("Μέθοδος", ["TOPSIS", "PROMETHEE II", "Και τα δύο"], index=2, key="syn_method")

    team = profiles_for(team_names, spread, "learnset")
    ds = synergy_dataset(team, sw)

    st.subheader("1. Dataset συνέργειας ζευγών")
    up = st.file_uploader("Προαιρετικά: ανέβασε δικό σου CSV συνέργειας "
                          "(στήλες: Pokémon A, Pokémon B, Synergy (0-100))", type="csv")
    if up is not None:
        user = pd.read_csv(up)
        need = {"Pokémon A", "Pokémon B", "Synergy (0-100)"}
        if need <= set(user.columns):
            m = {frozenset((r["Pokémon A"], r["Pokémon B"])): r["Synergy (0-100)"] for _, r in user.iterrows()}
            ds["Synergy (0-100)"] = [m.get(frozenset((a, b)), s) for a, b, s in
                                     zip(ds["Pokémon A"], ds["Pokémon B"], ds["Synergy (0-100)"])]
            st.success("Χρησιμοποιούνται οι τιμές από το αρχείο σου όπου υπάρχουν.")
        else:
            st.error(f"Το CSV πρέπει να έχει τις στήλες: {', '.join(need)}")
    st.caption("Μπορείς να διορθώσεις χειροκίνητα τη στήλη «Synergy (0-100)» (π.χ. με γνώση του meta).")
    ds = st.data_editor(ds, hide_index=True, width="stretch", key="syn_editor",
                        disabled=[c for c in ds.columns if c != "Synergy (0-100)"])
    st.download_button("⬇️ Synergy dataset (CSV)", ds.to_csv(index=False).encode("utf-8-sig"),
                       "synergy_pairs.csv", "text/csv")
    M = synergy_matrix(ds, team_names)
    st.dataframe(M.style.background_gradient(cmap="RdYlGn", axis=None, vmin=0, vmax=100).format("{:.0f}",
                 na_rep="—"), width="stretch")

    with st.expander("Δημιουργία synergy dataset για μεγαλύτερη λίστα (π.χ. όλα τα υποψήφια της κατάταξης)"):
        pool = st.multiselect("Pokémon", NAMES, key="pool",
                              default=[n for n in (st.session_state.get("rank_names") or team_names)
                                       if n in BY_NAME])
        if st.button("Δημιουργία dataset") and len(pool) >= 2:
            big = synergy_dataset(profiles_for(pool, spread, "learnset"), sw)
            st.write(f"{len(big)} ζεύγη")
            st.dataframe(big, width="stretch", hide_index=True, height=300)
            st.download_button("⬇️ Πλήρες dataset (CSV)", big.to_csv(index=False).encode("utf-8-sig"),
                               "synergy_dataset_full.csv", "text/csv")

    st.subheader("2. Κατάταξη των 15 τετράδων")
    indiv = st.session_state.get("indiv_scores", {})
    missing = [n for n in team_names if n not in indiv]
    if missing:
        # individual value inside the team: TOPSIS on base criteria, team as benchmark
        X = single_criteria(team, damage_matrix(team))
        keys = [c[0] for c in CRITERIA if c[4]]
        t = topsis(X[keys], [c[3] for c in CRITERIA if c[4]], [c[2] for c in CRITERIA if c[4]])
        indiv = {**t["TOPSIS C*"].to_dict(), **{k: v for k, v in indiv.items() if k in team_names}}
        st.caption("Η «ατομική αξία» υπολογίστηκε με TOPSIS μέσα στην 6άδα (για χρήση της κατάταξης της "
                   "σελίδας 🏆, τρέξε την πρώτα με αυτά τα Pokémon).")
    Q = quad_criteria(team, M, indiv)
    keys, w, ben, funcs = criteria_editor("crit_quad", [(*c, True) for c in QUAD_CRITERIA])
    if not keys:
        st.warning("Επίλεξε τουλάχιστον ένα κριτήριο.")
        return
    labels = {c[0]: c[1] for c in QUAD_CRITERIA}
    Qs = Q[keys].rename(columns=labels)
    with st.expander("Πίνακας αποφάσεων τετράδων"):
        st.dataframe(Qs.round(3), width="stretch")
    res = run_mcda(Qs, w, ben, funcs, method)
    best = res.index[0]
    bench = [n for n in team_names if n not in best.split(" + ")]
    st.success(f"🥇 Καλύτερη τετράδα: **{best}**  ·  Πάγκος: {', '.join(bench)}")
    show_ranking(res, method, "quads")

    st.markdown("**Πόσο συχνά εμφανίζεται κάθε Pokémon στις top-5 τετράδες**")
    top5 = res.index[:5]
    freq = pd.Series({n: sum(n in q.split(" + ") for q in top5) for n in team_names}).sort_values()
    st.bar_chart(freq, horizontal=True)


# ============================================================ page 5: methodology
def page_method():
    st.header("ℹ️ Μεθοδολογία")
    st.markdown(r"""
**Stats ανά level** (Gen 3+):
$HP = \lfloor (2B + IV + \lfloor EV/4 \rfloor)\cdot L/100 \rfloor + L + 10$,
$Stat = \lfloor(\lfloor (2B + IV + \lfloor EV/4 \rfloor)\cdot L/100 \rfloor + 5)\cdot Nature\rfloor$

**Ζημιά** (Gen 5–9): $Base = \lfloor \lfloor \lfloor 2L/5 + 2 \rfloor \cdot P \cdot A/D \rfloor / 50 \rfloor + 2$,
έπειτα × spread (0.75) × καιρός × crit (1.5) × random (0.85–1.00) × STAB (1.5 / 2.0 με Tera) ×
αποτελεσματικότητα τύπου × burn × λοιποί (items, abilities, screens).

**Κριτήρια κατάταξης Pokémon**: τα stats στο επιλεγμένο level, *επιθετική ισχύς* = μέσο % HP που αφαιρεί η
καλύτερη STAB κίνηση (γενικής δύναμης) σε κάθε άλλο υποψήφιο (round-robin, cap 100%), *δεχόμενη ζημιά* =
το αντίστροφο, *κάλυψη* = πόσους τύπους χτυπά super-effective, αδυναμίες/αντιστάσεις από τον πίνακα τύπων.

**TOPSIS**: διανυσματική κανονικοποίηση $r_{ij}=x_{ij}/\sqrt{\sum_i x_{ij}^2}$, σταθμισμένος πίνακας
$v_{ij}=w_j r_{ij}$, ιδανική $A^+$ / αρνητικά ιδανική $A^-$ λύση, αποστάσεις $D^\pm$ και
$C_i^*=D_i^-/(D_i^+ + D_i^-)$.

**PROMETHEE II**: για κάθε ζεύγος $(a,b)$ και κριτήριο $j$, $d_j=g_j(a)-g_j(b)$, συνάρτηση προτίμησης
$P_j(d)$ (usual, U-shape, V-shape, level, linear, Gaussian), $\pi(a,b)=\sum_j w_j P_j(d_j)$,
$\Phi^+(a)=\frac{1}{n-1}\sum_b \pi(a,b)$, $\Phi^-(a)=\frac{1}{n-1}\sum_b \pi(b,a)$,
$\Phi(a)=\Phi^+-\Phi^-$. Default: linear με $q=0$ και $p$ = τυπική απόκλιση του κριτηρίου.

**Συνέργεια ζεύγους (0–100)** = σταθμισμένο άθροισμα:
- *Αμυντική κάλυψη*: τύποι όπου ο ένας είναι αδύναμος και ο άλλος αντέχει, μείον 1.5× οι κοινές αδυναμίες
- *Επιθετική κάλυψη*: τύποι SE που καλύπτει το ζεύγος + το κέρδος έναντι του καλύτερου μεμονωμένου
- *Ρόλοι*: φυσικός + ειδικός επιθετικός = 1, ίδιος ρόλος = 0.3
- *Διαφορετικότητα τύπων*: ποινή για κοινούς τύπους
- *Abilities*: καιρός/terrain setter + abuser (π.χ. Drought + Chlorophyll), υποστηρικτικές abilities (Intimidate…)

**Τετράδες**: όλοι οι $\binom{6}{4}=15$ συνδυασμοί αξιολογούνται με μέση/ελάχιστη συνέργεια, κάλυψη,
συσσώρευση αδυναμιών, ακάλυπτες αδυναμίες, μέση ατομική αξία, ισορροπία φυσ./ειδ., μέση Speed.

**Σημειώσεις**: Το Legends Z-A έχει real-time μάχες (ο τύπος ζημιάς είναι προσέγγιση) και δεν έχει abilities.
Αν το PokeAPI δεν έχει ακόμα learnset για Z-A/Champions, χρησιμοποιείται του Scarlet/Violet.
""")


{"📘 Pokédex & Stats": page_pokedex, "💥 Damage Calculator": page_damage, "🏆 Κατάταξη Pokémon": page_ranking,
 "🤝 Synergy & τετράδες": page_synergy, "ℹ️ Μεθοδολογία": page_method}[PAGE]()
