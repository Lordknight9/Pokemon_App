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
from core.damage import (ATTACKER_ABILITIES, CALC_ABILITIES, DEFENDER_ABILITIES, ITEMS, Battler, Field, Move, calc,
                         intimidate_stage, ko_text)
from core.data import DemoProvider, LiveProvider, pretty
from core import ml
from core.move_flags import FLAG_LABEL, sd_id
from core.usage import Usage
from core.mcda import PREF_FUNCS, promethee, spearman, topsis
from core.rosters import GAMES
from core.stats import (NATURES, SHORT, SPREADS, STAT_LABEL, STATS, calc_all, nature_label, recommended_spread,
                        spread_for, spread_text)
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


CAT_COLOR = {"physical": "#E4572E", "special": "#2F6FDB", "status": "#8C8C8C"}
CAT_ICON = {"physical": "💥", "special": "🌀", "status": "✨"}


def cat_badge(c: str) -> str:
    return (f"<span style='background:{CAT_COLOR.get(c, '#888')};color:white;padding:2px 10px;"
            f"border-radius:10px;margin-right:4px;font-size:0.85em;font-weight:600'>"
            f"{CAT_ICON.get(c, '')} {c.capitalize()}</span>")


def _text_on(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#222" if (0.299 * r + 0.587 * g + 0.114 * b) > 170 else "white"


def style_moves(df: pd.DataFrame, move_col: str, cat_col: str, types: list):
    """Colour move names by type (types = list aligned with df rows) and category cells."""
    types = [str(t).lower() for t in types]
    mi_, ci_ = df.columns.get_loc(move_col), df.columns.get_loc(cat_col)
    ti_ = df.columns.get_loc("Τύπος") if "Τύπος" in df.columns else None

    def styles(_):
        out = pd.DataFrame("", index=df.index, columns=df.columns)
        for pos, (idx, t) in enumerate(zip(df.index, types)):
            tc = TYPE_COLOR.get(t)
            if tc:
                css = f"background-color:{tc};color:{_text_on(tc)};font-weight:600"
                out.iat[pos, mi_] = css
                if ti_ is not None:
                    out.iat[pos, ti_] = css
            cc = CAT_COLOR.get(str(df.iat[pos, ci_]).lower())
            if cc:
                out.iat[pos, ci_] = f"background-color:{cc};color:white;font-weight:600"
        return out
    floats = [c for c in df.columns if df[c].dtype.kind == "f"]
    return (df.style.apply(styles, axis=None).format(na_rep="—")
            .format("{:.1f}", subset=floats, na_rep="—"))


def badge(t: str) -> str:
    return (f"<span style='background:{TYPE_COLOR.get(t, '#888')};color:white;padding:2px 10px;"
            f"border-radius:10px;margin-right:4px;font-size:0.85em;font-weight:600'>{t.capitalize()}</span>")


# ============================================================ data access (cached)
@st.cache_resource
def _provider(source: str, class_id: int):
    return DemoProvider() if source == "demo" else LiveProvider()


def get_provider(source: str):
    """Provider instance; the class id in the cache key makes a code update (redeploy / hot reload)
    create a fresh instance instead of reusing an object built from the old class."""
    cls = DemoProvider if source == "demo" else LiveProvider
    return _provider(source, id(cls))


@st.cache_data(show_spinner=False)
def api_ok(source: str) -> bool:
    return get_provider(source).ping()


@st.cache_data(show_spinner="Φόρτωση λίστας Pokémon…")
def get_roster(source: str, game: str, megas: bool) -> list:
    return get_provider(source).roster(game, include_megas=megas)


@st.cache_data(show_spinner=False)
def get_pokemon(source: str, entry: dict) -> dict:
    return {**get_provider(source).pokemon(entry), "display": entry["display"]}


@st.cache_data(show_spinner="Λήψη δεδομένων από PokeAPI…")
def get_many(source: str, entries: tuple) -> list:
    ents = [dict(e) for e in entries]
    res = get_provider(source).pokemon_many(ents)
    return [{**r, "display": e["display"]} for r, e in zip(res, ents)]


@st.cache_data(show_spinner="Λήψη πίνακα κινήσεων (μία φορά)…")
def get_move_index(source: str) -> dict:
    return get_provider(source).move_index()


@st.cache_data(show_spinner=False)
def get_move(source: str, name: str) -> dict:
    return get_provider(source).move(name)


@st.cache_data(show_spinner="Λήψη λεπτομερειών κινήσεων…")
def get_moves_many(source: str, names: tuple) -> dict:
    return get_provider(source).moves_many(list(names))


@st.cache_data(show_spinner="Λήψη move flags (Pokémon Showdown data)…")
def get_flags_table(source: str) -> tuple:
    return get_provider(source).move_flags_table()


def flags_for(slug: str) -> frozenset:
    return frozenset(get_flags_table(SRC)[0].get(sd_id(slug), []))


def flags_text(fl) -> str:
    return " · ".join(FLAG_LABEL[f] for f in ("contact", "punch", "bite", "pulse", "slicing", "sound", "bullet",
                                               "wind", "secondary", "recoil") if f in fl)


@st.cache_data(show_spinner=False)
def get_ability(source: str, name: str) -> dict:
    try:
        return get_provider(source).ability(name)
    except Exception:
        return {"name": name, "display": pretty(name), "desc": ""}


@st.cache_data(show_spinner=False)
def get_evolution(source: str, species: str):
    try:
        return get_provider(source).evolution_chain(species)
    except Exception:
        return None


def as_key(e: dict) -> tuple:
    return tuple(sorted(e.items()))


def load_entries(entries: list) -> list:
    res = get_many(SRC, tuple(as_key(e) for e in entries))
    bad = [r["display"] for r in res if "error" in r]
    if bad:
        st.warning("Δεν βρέθηκαν στο PokeAPI (παραλείπονται): " + ", ".join(bad))
    return [r for r in res if "error" not in r]


# ============================================================ sidebar (+ URL query params for links)
PAGES = {"dex": "📘 Pokédex & Stats", "dmg": "💥 Damage Calculator", "rank": "🏆 Κατάταξη Pokémon",
         "mate": "🎯 VGC συμπαίκτες (ML)", "syn": "🤝 Synergy & τετράδες", "info": "ℹ️ Μεθοδολογία"}
QP = dict(st.query_params)


def _qp_index(options: list, key: str, default: int = 0) -> int:
    v = QP.get(key)
    return options.index(v) if v in options else default


st.sidebar.title("⚔️ Pokémon MCDA Lab")
_srcs = ["live", "demo"]
SRC = st.sidebar.radio("Πηγή δεδομένων", _srcs, index=_qp_index(_srcs, "src"),
                       format_func=lambda k: "PokeAPI (online)" if k == "live" else "Demo (offline, 24 Pokémon)",
                       help="Το Demo λειτουργεί χωρίς internet με μικρό ενσωματωμένο dataset.")
if SRC == "live" and not api_ok("live"):
    st.sidebar.error("Το PokeAPI δεν απαντά. Έλεγξε τη σύνδεση ή διάλεξε Demo.")

GAME = st.sidebar.selectbox("Παιχνίδι", list(GAMES), index=_qp_index(list(GAMES), "game"),
                            format_func=lambda g: GAMES[g]["label"])
G = GAMES[GAME]
try:
    _lvl = min(100, max(1, int(QP.get("lvl", G["default_level"]))))
except ValueError:
    _lvl = G["default_level"]
LEVEL = st.sidebar.slider("Level", 1, 100, _lvl)
MEGAS = st.sidebar.checkbox("✦ Mega Evolutions", value=QP.get("mega", "1" if GAME != "sv" else "0") == "1",
                            disabled=(GAME == "sv" or SRC == "demo"), key=f"megas_{GAME}",
                            help="Προσθέτει τις Mega μορφές που υπάρχουν στο PokeAPI (Z-A / Champions). "
                                 "Εμφανίζονται με ✦ αμέσως μετά τη βασική μορφή.")
PAGE_KEY = st.sidebar.radio("Σελίδα", list(PAGES), index=_qp_index(list(PAGES), "page"),
                            format_func=PAGES.get)
st.sidebar.caption("Δεδομένα: PokeAPI (pokeapi.co). Rosters Z-A/Champions: ενσωματωμένες λίστες, Σεπτ. 2026.")

try:
    ROSTER = get_roster(SRC, GAME, MEGAS)
except Exception as ex:
    st.error(f"Αποτυχία φόρτωσης roster: {ex}")
    st.stop()
BY_NAME = {e["display"]: e for e in ROSTER}
NAMES = list(BY_NAME)
IN_ROSTER = list(NAMES)
N_MEGA = sum(1 for e in ROSTER if e.get("form") == "mega")
if MEGAS and N_MEGA:
    st.sidebar.caption(f"✦ {N_MEGA} Mega μορφές στη λίστα")


def find_name(slug: str | None) -> str | None:
    """Roster display name for a pokemon or species slug (adds out-of-roster Pokémon on demand)."""
    if not slug:
        return None
    for e in ROSTER:
        if e["pokemon"] == slug:
            return e["display"]
    for e in ROSTER:
        if e["species"] == slug and not e.get("form"):
            return e["display"]
    if SRC == "demo":
        return None
    try:
        dex = get_provider(SRC).species_ids().get(slug)
    except Exception:
        dex = None
    e = {"display": pretty(slug), "species": slug, "pokemon": slug, "form": None, "dex": dex, "extra": True}
    if e["display"] not in BY_NAME:
        BY_NAME[e["display"]] = e
        NAMES.append(e["display"])
        NAMES.sort(key=lambda n: (BY_NAME[n].get("dex") or 99999, BY_NAME[n].get("form") == "mega", n))
    return e["display"]


def page_link(slug: str) -> str:
    """URL (query string) that opens the Pokédex page of a Pokémon, keeping the sidebar settings."""
    return f"?src={SRC}&game={GAME}&lvl={LEVEL}&mega={int(MEGAS)}&page=dex&p={slug}"


def sync_url(**extra):
    q = {"src": SRC, "game": GAME, "lvl": str(LEVEL), "mega": str(int(MEGAS)), "page": PAGE_KEY, **extra}
    if dict(st.query_params) != q:
        st.query_params.clear()
        st.query_params.update(q)


def label(name: str) -> str:
    """'#0006 Charizard' / '#0006 ✦ Mega Charizard X' — searchable by number or name."""
    e = BY_NAME.get(name, {})
    num = f"#{e['dex']:04d} " if e.get("dex") else ""
    tail = " (εκτός λίστας παιχνιδιού)" if e.get("extra") else ""
    return f"{num}{'✦ ' if e.get('form') == 'mega' else ''}{name}{tail}"


def sprite(entry: dict, small: bool = False) -> str | None:
    try:
        return get_provider(SRC).sprite_url(entry, small)
    except Exception:
        return None


def default_pick(n: int) -> list:
    pop = [x for x in POPULAR if x in BY_NAME]
    return (pop + [x for x in NAMES if x not in pop])[:n]


def synced_multiselect(title: str, store_key: str, default: list, **kw) -> list:
    """Multiselect whose value survives page switches and game changes."""
    wkey = "w_" + store_key
    if wkey in st.session_state:
        st.session_state[wkey] = [x for x in st.session_state[wkey] if x in BY_NAME]
    else:
        st.session_state[wkey] = [x for x in (st.session_state.get(store_key) or default) if x in BY_NAME]
    val = st.multiselect(title, NAMES, key=wkey, format_func=label, **kw)
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
        cfg = {"Εικόνα": st.column_config.ImageColumn("", width="small")} if "Εικόνα" in res else None
        st.dataframe(res.round(4), width="stretch", column_config=cfg, height=min(38 + 35 * len(res), 600))
    with c2:
        top = res[score_col].sort_values(ascending=False).head(30).sort_values()
        if len(res) > 30:
            st.caption("Top 30")
        st.bar_chart(top, horizontal=True)
    if method == "Και τα δύο":
        rho = spearman(res["Κατάταξη TOPSIS"], res["Κατάταξη PROMETHEE"])
        st.info(f"Συσχέτιση Spearman ανάμεσα σε TOPSIS και PROMETHEE: **ρ = {rho:.3f}**")
    st.download_button(f"⬇️ Αποτελέσματα {label} (CSV)",
                       res.drop(columns=["Εικόνα"], errors="ignore").to_csv().encode("utf-8-sig"),
                       file_name=f"ranking_{label}.csv", mime="text/csv")


# ============================================================ page 1: Pokédex
BOX_CSS = """
<style>
.pk-row{display:flex;flex-wrap:wrap;gap:8px;align-items:stretch;justify-content:center;margin:6px 0 14px}
.pk-box{display:flex;flex-direction:column;align-items:center;justify-content:center;width:92px;
  padding:6px 4px;border:2px solid rgba(128,128,128,.35);border-radius:12px;background:rgba(120,140,200,.08);
  text-decoration:none!important;color:inherit!important;font-size:11px;line-height:1.2;text-align:center}
.pk-box:hover{border-color:#E3350D;background:rgba(227,53,13,.08)}
.pk-box img{width:64px;height:64px;object-fit:contain}
.pk-box.cur{border-color:#E3350D;box-shadow:0 0 0 2px rgba(227,53,13,.25)}
.pk-box.nav{width:130px;font-size:13px;font-weight:600}
.pk-box .num{opacity:.65}
.pk-evo{display:flex;align-items:center;flex-wrap:nowrap;overflow-x:auto;padding:4px 0 10px}
.pk-evo .col{display:flex;flex-direction:column;gap:10px}
.pk-evo .step{display:flex;align-items:center}
.pk-arrow{min-width:110px;max-width:150px;text-align:center;font-size:12px;opacity:.85;padding:0 6px}
.pk-arrow b{font-size:20px;display:block;line-height:1}
.pk-tag{display:inline-block;font-size:11px;padding:1px 7px;border-radius:8px;margin-left:6px;
  background:rgba(128,128,128,.18)}
.pk-abil{margin:4px 0 10px}
</style>
"""


def box_html(entry: dict | None, cur: bool = False, nav: str = "", big: bool = False) -> str:
    if not entry:
        return ""
    img = sprite(entry)
    num = f"#{entry['dex']:04d}" if entry.get("dex") else ""
    mega = "✦ " if entry.get("form") == "mega" else ""
    pic = f"<img src='{img}' loading='lazy'/>" if img else "<div style='height:64px;line-height:64px'>?</div>"
    arrow_l = "◀ " if nav == "prev" else ""
    arrow_r = " ▶" if nav == "next" else ""
    cls = "pk-box" + (" cur" if cur else "") + (" nav" if nav or big else "")
    return (f"<a class='{cls}' href='{page_link(entry['pokemon'])}' target='_self'>{pic}"
            f"<span class='num'>{arrow_l}{num}{arrow_r}</span><span>{mega}{entry['display']}</span></a>")


def evo_html(node: dict, current_species: str) -> str:
    e = {"display": pretty(node["species"]), "species": node["species"], "pokemon": node["species"],
         "dex": node["id"]}
    nm = find_name(node["species"])
    if nm:
        e = dict(BY_NAME[nm])
    out = box_html(e, cur=(node["species"] == current_species))
    kids = node.get("evolves_to") or []
    if kids:
        steps = []
        for k in kids:
            cond = " / ".join(dict.fromkeys(k["details"])) or "—"
            steps.append(f"<div class='step'><div class='pk-arrow'><b>➜</b>{cond}</div>"
                         f"{evo_html(k, current_species)}</div>")
        out += f"<div class='col'>{''.join(steps)}</div>"
    return f"<div class='step'>{out}</div>"


def page_pokedex():
    st.markdown(BOX_CSS, unsafe_allow_html=True)
    st.header("📘 Pokédex & Stats")
    st.caption(f"{G['label']} — {len(IN_ROSTER)} Pokémon στη λίστα · πάτα σε εικόνα για να ανοίξει η σελίδα του")
    start = find_name(QP.get("p")) or default_pick(1)[0]
    name = st.selectbox("Pokémon (γράψε όνομα ή αριθμό Pokédex)", NAMES, format_func=label,
                        key="dex_pokemon", index=NAMES.index(start))
    entry = BY_NAME[name]
    sync_url(p=entry["pokemon"])
    try:
        p = get_pokemon(SRC, entry)
    except Exception as ex:
        st.error(f"Σφάλμα: {ex}")
        return
    c1, c2 = st.columns([1, 2])
    with c1:
        if p.get("sprite"):
            st.image(p["sprite"], width=220)
        st.markdown(f"**{label(name)}**")
        st.markdown(" ".join(badge(t) for t in p["types"]), unsafe_allow_html=True)
        st.metric("Base Stat Total", sum(p["base"].values()))
    with c2:
        st.subheader(f"Stats στο Level {LEVEL}")
        spread = st.selectbox("Spread", list(SPREADS) + ["custom"],
                              format_func=lambda k: "✏️ Custom (δικά μου EVs / IVs / Nature)" if k == "custom"
                              else SPREADS[k], key="dex_spread")
        if spread == "custom":
            nature = st.selectbox("Nature", list(NATURES), format_func=nature_label,
                                  index=list(NATURES).index("Hardy"), key="dex_nat")
            evs, ivs = {}, {}
            cols = st.columns(6)
            for i, s in enumerate(STATS):
                evs[s] = cols[i].number_input(f"EV {SHORT[s]}", 0, 252, 0, 4, key=f"ev_{s}")
                ivs[s] = cols[i].number_input(f"IV {SHORT[s]}", 0, 31, 31, key=f"iv_{s}")
            if sum(evs.values()) > 510:
                st.warning(f"Σύνολο EVs {sum(evs.values())} > 510 (όριο παιχνιδιού).")
        else:
            evs, nature, ivs = spread_for(p["base"], spread)
            if spread == "recommended":
                st.info(f"**{recommended_spread(p['base'])['role']}** → {spread_text(evs, nature, ivs)}")
            else:
                st.caption(spread_text(evs, nature, ivs))
        final = calc_all(p["base"], LEVEL, ivs, evs, nature)
        tbl = pd.DataFrame({"Base": p["base"], "EVs": {s: evs.get(s, 0) for s in STATS},
                            f"Lv {LEVEL}": final,
                            "Lv 100 (max)": calc_all(p["base"], 100, None, {s: 252 for s in STATS})})
        tbl.index = [STAT_LABEL[s] for s in tbl.index]
        st.dataframe(tbl, width="stretch")
        st.bar_chart(tbl[[f"Lv {LEVEL}"]], horizontal=True)

    abilities_section(p)

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

    evolution_section(p)
    mega_section(p, entry)
    moves_section(p)
    box_navigation(name)


def abilities_section(p: dict):
    st.subheader("Abilities")
    if not G["has_abilities"]:
        st.caption("Στο Legends Z-A οι abilities δεν λειτουργούν στη μάχη — εμφανίζονται για αναφορά.")
    hidden = set(p.get("hidden_abilities", []))
    rows = []
    for a in p["abilities"]:
        info = get_ability(SRC, a)
        tags = ""
        if a in hidden:
            tags += "<span class='pk-tag'>Hidden</span>"
        if a in CALC_ABILITIES:
            eff = ATTACKER_ABILITIES.get(a) or DEFENDER_ABILITIES.get(a)
            tags += f"<span class='pk-tag' title='{eff}'>⚙️ calculator: {eff}</span>"
        rows.append(f"<div class='pk-abil'><b>{pretty(a)}</b>{tags}<br>"
                    f"<span style='opacity:.8'>{info.get('desc') or '—'}</span></div>")
    st.markdown("".join(rows), unsafe_allow_html=True)


def evolution_section(p: dict):
    chain = get_evolution(SRC, p["species"])
    if not chain:
        return
    st.subheader("Evolution line")
    if not chain.get("evolves_to"):
        st.caption("Δεν εξελίσσεται.")
        return
    st.markdown(f"<div class='pk-evo'>{evo_html(chain, p['species'])}</div>", unsafe_allow_html=True)


def mega_section(p: dict, entry: dict):
    """Clickable cards for the Mega forms (or the base form, if a Mega is selected) + stat deltas."""
    same = [e for e in ROSTER if e["species"] == entry["species"] and e["display"] != entry["display"]]
    if entry.get("form") == "mega":
        related = sorted(same, key=lambda e: e.get("form") == "mega")
        title = "✦ Βασική μορφή & άλλες Mega"
    else:
        related = [e for e in same if e.get("form") == "mega"]
        title = "✦ Mega Evolutions"
    if not related:
        if GAME != "sv" and not MEGAS:
            st.caption("✦ Ενεργοποίησε «Mega Evolutions» στο sidebar για να δεις τις Mega μορφές.")
        return
    st.subheader(title)
    st.markdown("<div class='pk-row' style='justify-content:flex-start'>" +
                box_html(entry, cur=True, big=True) + "".join(box_html(e, big=True) for e in related) +
                "</div>", unsafe_allow_html=True)
    forms = [p] + [get_pokemon(SRC, e) for e in related]
    comp = pd.DataFrame({f["display"]: f["base"] for f in forms})
    comp.index = [STAT_LABEL[s] for s in comp.index]
    comp.loc["BST"] = comp.sum()
    first = comp.columns[0]
    for c in comp.columns[1:]:
        comp[f"Δ {c}"] = comp[c] - comp[first]
    delta_cols = [c for c in comp.columns if c.startswith("Δ ")]
    info = pd.DataFrame({f["display"]: [" / ".join(t.capitalize() for t in f["types"]),
                                        ", ".join(pretty(a) for a in f["abilities"])] for f in forms},
                        index=["Τύποι", "Ability"])
    st.dataframe(info, width="stretch")
    st.dataframe(comp.style.map(lambda v: "color:#1a7f37;font-weight:600" if v > 0 else
                                "color:#c62828;font-weight:600" if v < 0 else "", subset=delta_cols)
                 .format("{:+d}", subset=delta_cols), width="stretch")


def moves_section(p: dict):
    st.subheader("Κινήσεις (learnset για το παιχνίδι)")
    mi = get_move_index(SRC)
    ls = get_provider(SRC).learnset(p, GAME)
    details = get_moves_many(SRC, tuple(ls))
    rows = []
    for m in ls:
        d = details.get(m) or {}
        rows.append({"Κίνηση": pretty(m),
                     "Τύπος": (d.get("type") or mi.get(m, {}).get("type", "?")).capitalize(),
                     "Κατ.": (d.get("category") or mi.get(m, {}).get("category", "?")).capitalize(),
                     "Power": d.get("power"), "Acc.": d.get("accuracy"), "PP": d.get("pp"),
                     "Prio": d.get("priority"), "Flags": flags_text(flags_for(m)),
                     "Περιγραφή": d.get("desc", "")})
    mv = pd.DataFrame(rows)
    if mv.empty:
        st.info("Δεν βρέθηκαν κινήσεις.")
        return
    for c in ("Power", "Acc.", "PP", "Prio"):
        mv[c] = pd.to_numeric(mv[c], errors="coerce").astype("Int64")
    f1, f2, f3 = st.columns([2, 2, 1])
    ft = f1.multiselect("Φίλτρο τύπου", sorted(mv["Τύπος"].unique()), key="dex_ft")
    fc = f2.multiselect("Φίλτρο κατηγορίας", ["Physical", "Special", "Status"], key="dex_fc")
    show_type = f3.checkbox("Στήλη τύπου", value=False, key="dex_showtype")
    if ft:
        mv = mv[mv["Τύπος"].isin(ft)]
    if fc:
        mv = mv[mv["Κατ."].isin(fc)]
    st.markdown(" ".join(cat_badge(c) for c in ("physical", "special", "status")) +
                "<span style='opacity:.7;font-size:.85em'> · το χρώμα του ονόματος = τύπος κίνησης</span>",
                unsafe_allow_html=True)
    mv = mv.reset_index(drop=True)
    types = mv["Τύπος"].tolist()
    if not show_type:
        mv = mv.drop(columns=["Τύπος"])
    st.dataframe(style_moves(mv, "Κίνηση", "Κατ.", types), hide_index=True, width="stretch",
                 height=min(38 + 35 * len(mv), 460),
                 column_config={
                     "Κίνηση": st.column_config.TextColumn(width="medium"),
                     "Τύπος": st.column_config.TextColumn(width="small"),
                     "Κατ.": st.column_config.TextColumn(width="small"),
                     "Power": st.column_config.NumberColumn(width="small"),
                     "Acc.": st.column_config.NumberColumn(width="small"),
                     "PP": st.column_config.NumberColumn(width="small"),
                     "Prio": st.column_config.NumberColumn(width="small"),
                     "Flags": st.column_config.TextColumn(width="small", help="Από τα δεδομένα του Pokémon Showdown"),
                     "Περιγραφή": st.column_config.TextColumn(width="large"),
                 })
    if GAME != "sv":
        st.caption("Αν το PokeAPI δεν έχει ακόμα learnset για αυτό το παιχνίδι, εμφανίζεται το learnset του "
                   "Scarlet/Violet (ή όλες οι κινήσεις). Οι Mega μορφές έχουν το learnset της βασικής μορφής.")


def box_navigation(name: str):
    """Previous / Next like PC-box slots, with the neighbouring Pokémon around the current one."""
    if name not in IN_ROSTER:
        return
    i = IN_ROSTER.index(name)
    n = len(IN_ROSTER)
    prev_e = BY_NAME[IN_ROSTER[(i - 1) % n]]
    next_e = BY_NAME[IN_ROSTER[(i + 1) % n]]
    window = [IN_ROSTER[(i + k) % n] for k in range(-3, 4)] if n > 7 else IN_ROSTER
    st.divider()
    html = ("<div class='pk-row'>" + box_html(prev_e, nav="prev") +
            "".join(box_html(BY_NAME[x], cur=(x == name)) for x in window) +
            box_html(next_e, nav="next") + "</div>")
    st.markdown(html, unsafe_allow_html=True)


# ============================================================ page 2: damage calc
def battler_panel(col, side: str, default_name: str):
    with col:
        st.subheader("Επιτιθέμενος" if side == "a" else "Αμυνόμενος")
        name = st.selectbox("Pokémon", NAMES, index=NAMES.index(default_name), key=f"{side}_name",
                            format_func=label)
        p = get_pokemon(SRC, BY_NAME[name])
        st.markdown(" ".join(badge(t) for t in p["types"]), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        lvl = c1.number_input("Level", 1, 100, LEVEL, key=f"{side}_lvl")
        spread = c2.selectbox("Spread", list(SPREADS) + ["custom"], key=f"{side}_spread",
                              format_func=lambda k: "✏️ Custom" if k == "custom" else SPREADS[k])
        ability = None
        if G["has_abilities"] and p["abilities"]:
            hidden = set(p.get("hidden_abilities", []))
            ability = c1.selectbox(
                "Ability", p["abilities"], key=f"{side}_ab",
                format_func=lambda a: pretty(a) + (" (H)" if a in hidden else "") +
                (" ⚙️" if a in CALC_ABILITIES else ""),
                help="⚙️ = επηρεάζει τον υπολογισμό ζημιάς · (H) = Hidden ability")
        item = c2.selectbox("Item", ITEMS, key=f"{side}_item")
        item = None if item.startswith("(") else item
        tera = None
        if G["has_tera"]:
            t = c1.selectbox("Tera type", ["—"] + TYPES, key=f"{side}_tera", format_func=str.capitalize)
            tera = None if t == "—" else t
        if spread == "custom":
            nature = c2.selectbox("Nature", list(NATURES), key=f"{side}_nat", format_func=nature_label,
                                  index=list(NATURES).index("Adamant" if side == "a" else "Hardy"))
        else:
            evs, nature, ivs = spread_for(p["base"], spread)
            st.caption(("⭐ " + recommended_spread(p["base"])["role"] + ": " if spread == "recommended" else "")
                       + spread_text(evs, nature, ivs))
        with st.expander("EVs / IVs / Boosts" if spread == "custom" else "Boosts"):
            boosts = {}
            if spread == "custom":
                evs, ivs = {}, {}
            cc = st.columns(6)
            for i, s in enumerate(STATS):
                if spread == "custom":
                    d_ev = 252 if (side == "a" and s in ("atk", "spe")) or (side == "d" and s == "hp") else 0
                    evs[s] = cc[i].number_input(f"EV {SHORT[s]}", 0, 252, d_ev, 4, key=f"{side}_ev_{s}")
                    ivs[s] = cc[i].number_input(f"IV {SHORT[s]}", 0, 31, 31, key=f"{side}_iv_{s}")
                if s != "hp":
                    boosts[s] = cc[i].number_input(f"Boost {SHORT[s]}", -6, 6, 0, key=f"{side}_b_{s}")
        if ability:
            eff = (ATTACKER_ABILITIES if side == "a" else DEFENDER_ABILITIES).get(ability)
            desc = get_ability(SRC, ability).get("desc", "")
            role = "επιτιθέμενος" if side == "a" else "αμυνόμενος"
            head = (f"⚙️ **{pretty(ability)}**: {eff}" if eff else
                    f"**{pretty(ability)}** (δεν επηρεάζει τον υπολογισμό ως {role})")
            st.caption(head + (f" — {desc}" if desc else ""))
        stats = calc_all(p["base"], lvl, ivs, evs, nature)
        burned = False
        if side == "a":
            h1, h2 = st.columns([1, 2])
            burned = h1.checkbox("Burned", key="a_burn")
            hp_pct = h2.slider("Τρέχον HP % (για Blaze/Torrent κ.λπ.)", 1, 100, 100, key="a_hp")
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
    if D.ability == "intimidate" and st.checkbox(
            f"Intimidate του {D.name} ενεργό (Attack επιτιθέμενου {intimidate_stage(A.ability):+d})", value=True):
        A.boosts = dict(A.boosts)
        A.boosts["atk"] = max(-6, min(6, A.boosts.get("atk", 0) + intimidate_stage(A.ability)))

    st.divider()
    st.subheader("Κίνηση & πεδίο μάχης")
    moves = damaging_learnset(pa)
    if not moves:
        st.warning("Δεν βρέθηκαν επιθετικές κινήσεις.")
        return
    c1, c2, c3, c4 = st.columns(4)
    mi = get_move_index(SRC)
    mname = c1.selectbox("Κίνηση", moves, format_func=lambda m: f"{pretty(m)} · "
                         f"{mi.get(m, {}).get('type', '?').capitalize()} {CAT_ICON.get(mi.get(m, {}).get('category'), '')}")
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
    mflags = flags_for(mname)
    mv = Move(mname, md["type"], power, md["category"], spread, mflags)
    r = calc(A, D, mv, fld)
    hp = D.stats["hp"]
    cur = max(1, round(hp * D.hp_pct / 100))

    mtype = r.get("type", md["type"])
    tc = TYPE_COLOR.get(mtype, "#888")
    st.markdown(f"### {A.name} — <span style='color:{tc}'>**{pretty(mname)}**</span> {badge(mtype)}"
                f"{cat_badge(md['category'])} {power} BP → {D.name}", unsafe_allow_html=True)
    if md.get("desc"):
        st.caption(md["desc"])
    if mflags:
        st.caption("Flags: " + flags_text(mflags))
    if get_flags_table(SRC)[1] == "fallback" and (A.ability in ("tough-claws", "sheer-force") or
                                                   D.ability == "fluffy"):
        st.warning("Δεν ήταν διαθέσιμα τα δεδομένα του Pokémon Showdown, οπότε δεν είναι γνωστό αν η κίνηση "
                   "είναι contact ή έχει secondary effect — η ability δεν εφαρμόζεται.")
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
            rr = calc(A, D, Move(m, d["type"], d["power"], d["category"], spread, flags_for(m)), fld)
            rows.append({"Κίνηση": pretty(m), "Τύπος": d["type"].capitalize(), "Κατ.": d["category"].capitalize(),
                         "Power": d["power"],
                         "Min %": rr["min"] / hp * 100, "Max %": rr["max"] / hp * 100,
                         "KO": ko_text(rr["rolls"], cur) if rr["max"] else "—"})
        tab = pd.DataFrame(rows).sort_values("Max %", ascending=False).reset_index(drop=True)
        st.dataframe(style_moves(tab, "Κίνηση", "Κατ.", tab["Τύπος"].tolist()), width="stretch", hide_index=True)


# ============================================================ page 3: ranking
def profiles_for(names: list, spread: str, cov: str) -> list:
    ps = load_entries([BY_NAME[n] for n in names])
    mi = get_move_index(SRC)
    prov = get_provider(SRC)
    return [build_profile(p, LEVEL, spread, prov.learnset(p, GAME), mi, cov, G["has_abilities"]) for p in ps]


@st.cache_data(show_spinner="Υπολογισμός κριτηρίων & πίνακα ζημιάς…", max_entries=20)
def ranking_data(src: str, game: str, level: int, names: tuple, spread: str, power: int, cov: str):
    profiles = profiles_for(list(names), spread, cov)
    dmg = damage_matrix(profiles, power)
    return single_criteria(profiles, dmg), dmg


def page_ranking():
    st.header("🏆 Κατάταξη Pokémon με TOPSIS / PROMETHEE II")
    mode = st.radio("Υποψήφια Pokémon", ["all", "manual"], horizontal=True, key="rank_mode",
                    format_func=lambda k: f"Όλα τα Pokémon του παιχνιδιού ({len(IN_ROSTER)})" if k == "all"
                    else "Επιλογή με το χέρι")
    if mode == "all":
        f1, f2, f3 = st.columns([1, 2, 2])
        with_megas = f1.checkbox("Με Megas", value=True, key="rank_megas", disabled=not N_MEGA)
        types_f = f2.multiselect("Μόνο τύποι", TYPES, format_func=str.capitalize, key="rank_types",
                                 help="Κενό = όλοι οι τύποι")
        exclude = f3.multiselect("Εξαίρεση", IN_ROSTER, format_func=label, key="rank_excl")
        names = [n for n in IN_ROSTER if n not in exclude and (with_megas or BY_NAME[n].get("form") != "mega")]
        st.caption(f"**{len(names)}** Pokémon στην κατάταξη. Η πρώτη λήψη όλων από το PokeAPI παίρνει "
                   "1–2 λεπτά· μετά μένουν αποθηκευμένα.")
    else:
        names = synced_multiselect("Επιλογή Pokémon", "rank_names", default_pick(12))
        types_f = []
    if len(names) < 3:
        st.info("Χρειάζονται τουλάχιστον 3 Pokémon.")
        return
    c1, c2, c3, c4 = st.columns(4)
    spread = c1.selectbox("Spread", list(SPREADS), format_func=lambda k: SPREADS[k])
    power = c2.number_input("Power γενικής STAB κίνησης", 40, 150, 90,
                            help="Για τον πίνακα ζημιάς κάθε Pokémon χτυπά με την καλύτερη STAB του.")
    cov = c3.selectbox("Κάλυψη βάσει", ["learnset", "stab"],
                       format_func=lambda k: "όλων των επιθετικών κινήσεων" if k == "learnset" else "μόνο STAB")
    method = c4.radio("Μέθοδος", ["TOPSIS", "PROMETHEE II", "Και τα δύο"], index=2)

    X, dmg = ranking_data(SRC, GAME, LEVEL, tuple(names), spread, int(power), cov)
    if types_f:  # type filter applied after loading (types are known only then)
        keep = [n for n in X.index if n in BY_NAME and set(get_pokemon(SRC, BY_NAME[n])["types"]) & set(types_f)]
        X, dmg = X.loc[keep], dmg.loc[keep, keep]
        st.caption(f"Φίλτρο τύπου: {len(keep)} Pokémon (η επιθετική/αμυντική ισχύς μετράει απέναντι σε όλα).")
        if len(keep) < 3:
            st.info("Λιγότερα από 3 Pokémon με αυτούς τους τύπους.")
            return
    names = list(X.index)

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
        if len(dmg) <= 40:
            st.dataframe(dmg.round(1).style.background_gradient(cmap="Reds", axis=None).format("{:.1f}"),
                         width="stretch")
        else:
            st.dataframe(dmg.round(1), width="stretch", height=420)
        st.download_button("⬇️ Πίνακας ζημιάς (CSV)", dmg.round(2).to_csv().encode("utf-8-sig"),
                           "damage_matrix.csv")

    st.subheader("Αποτελέσματα")
    res = run_mcda(Xs, w, ben, funcs, method)
    disp = res.copy()
    disp.insert(0, "Εικόνα", [sprite(BY_NAME[n], small=True) if n in BY_NAME else None for n in res.index])
    disp.index = [label(n) if n in BY_NAME else n for n in res.index]
    show_ranking(disp, method, "pokemon")
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
        pool = st.multiselect("Pokémon", NAMES, key="pool", format_func=label,
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


# ============================================================ page: VGC teammates (usage data + ML)
@st.cache_data(show_spinner="Αναζήτηση VGC στατιστικών στο Smogon…", ttl=6 * 3600)
def get_smogon_formats(source: str, game: str) -> list:
    try:
        return get_provider(source).smogon_formats(game)
    except Exception:
        return []


@st.cache_data(show_spinner="Λήψη VGC στατιστικών (Smogon, μία φορά)…")
def get_smogon_usage(source: str, month: str, fmt: str, cutoff: str) -> dict:
    return get_provider(source).smogon_usage(month, fmt, cutoff)


@st.cache_data(show_spinner="Φόρτωση όλων των Pokémon του παιχνιδιού…", max_entries=6)
def roster_profiles(src: str, game: str, level: int, names: tuple) -> tuple:
    ps = load_entries([BY_NAME[n] for n in names])
    mi = get_move_index(src)
    prov = get_provider(src)
    profiles, learn = {}, {}
    for p in ps:
        ls = prov.learnset(p, game)
        profiles[p["display"]] = build_profile(p, level, "recommended", ls, mi, "learnset", G["has_abilities"])
        learn[p["display"]] = ls
    return profiles, learn


def resolver():
    """Showdown name ('Ninetales-Alola', 'Floette-Mega') -> roster display name."""
    idx = {}
    for e in ROSTER:
        slug = e["pokemon"]
        for k in {slug, slug.replace("-eternal", ""), slug.replace("-breed", "")}:
            idx.setdefault(sd_id(k), e["display"])
        if not e.get("form"):
            idx.setdefault(sd_id(e["species"]), e["display"])

    def resolve(name: str):
        k = sd_id(name)
        if k in idx:
            return idx[k]
        if "-mega" in name.lower():  # Mega not in PokeAPI yet -> count it as its base species
            return resolve(name[: name.lower().index("-mega")])
        return None
    return resolve


@st.cache_resource(show_spinner="Εκπαίδευση μοντέλου ML…", max_entries=8)
def trained_model(src, game, level, month, fmt, cutoff, kind, n_train, _usage, _profiles, _feats):
    names = [n for n in _usage.top() if n in _profiles][:n_train]
    X, y, w, idx = ml.training_set(names, _profiles, _feats, _usage)
    if len(X) < 50:
        return None
    res = ml.train_and_evaluate(X, y, w, kind)
    res.update({"n_pairs": len(X), "n_mons": len(names), "y": y, "idx": idx})
    return res


def reasons(pa, pb, fa, fb, x) -> str:
    out = []
    for w, lab in (("rain", "Rain"), ("sun", "Sun"), ("sand", "Sand"), ("snow", "Snow")):
        if fa[f"set_{w}"] * fb[f"abuse_{w}"] > 0.3:
            out.append(f"{lab}: {pa['display']} → {pb['display']}")
        if fb[f"set_{w}"] * fa[f"abuse_{w}"] > 0.3:
            out.append(f"{lab}: {pb['display']} → {pa['display']}")
    if x["trickroom_combo"] > 0.3:
        out.append("Trick Room")
    if x["tailwind_combo"] > 0.4:
        out.append("Tailwind")
    if x["redirect_combo"] > 0.3:
        out.append("Follow Me / Rage Powder")
    if x["eq_immune"] > 0.2:
        out.append("Earthquake + ανοσία Ground")
    if x["fakeout_sum"] > 0.8:
        out.append("Fake Out")
    if x["intimidate_sum"] > 0.5:
        out.append("Intimidate")
    if x["terrain_combo"] > 0.3:
        out.append("Terrain")
    if x["cover_pairs"] >= 4:
        out.append(f"καλύπτουν αδυναμίες ({x['cover_pairs']} τύποι)")
    if x["shared_weak"] >= 3:
        out.append(f"⚠️ {x['shared_weak']} κοινές αδυναμίες")
    return " · ".join(out)


def page_mates():
    st.header("🎯 VGC συμπαίκτες με πραγματικά δεδομένα + ML")
    st.caption("Διάλεξε ένα Pokémon και η εφαρμογή προτείνει με ποια ταιριάζει σε ομάδα VGC. Η «αλήθεια» "
               "είναι πόσο συχνά εμφανίζονται μαζί στις ομάδες του Showdown ladder (Smogon stats). Ένα μοντέλο "
               "ML μαθαίνει γιατί (καιρός, Trick Room, Fake Out, τύποι…) και προβλέπει και για ζευγάρια χωρίς δεδομένα.")
    if SRC == "demo":
        st.info("Χρειάζεται η πηγή «PokeAPI (online)».")
        return
    fmts = get_smogon_formats(SRC, GAME)
    if not fmts:
        st.warning("Δεν βρέθηκαν VGC στατιστικά για αυτό το παιχνίδι στο smogon.com/stats.")
        return
    c1, c2, c3 = st.columns([2, 1, 2])
    mf = c1.selectbox("Στατιστικά (μήνας · format)", fmts, format_func=lambda x: f"{x[0]} · {x[1]}")
    cutoff = c2.selectbox("Rating ≥", ["1760", "1630", "1500", "0"], index=0,
                          help="Μόνο ομάδες παικτών με rating πάνω από αυτό (καλύτεροι παίκτες = πιο «σωστές» ομάδες).")
    models = ml.available_models()
    kind = c3.selectbox("Μοντέλο ML", list(models), format_func=models.get)
    if GAME == "za":
        st.caption("Το Legends Z-A δεν έχει VGC ladder — χρησιμοποιούνται τα δεδομένα του Champions ως προσέγγιση.")
    try:
        comp = get_smogon_usage(SRC, mf[0], mf[1], cutoff)
    except Exception as ex:
        st.error(f"Αποτυχία λήψης στατιστικών: {ex}")
        return
    usage = Usage(comp, resolver())
    profiles, learn = roster_profiles(SRC, GAME, LEVEL, tuple(IN_ROSTER))
    feats = {n: ml.mon_features(p, usage.detail.get(n), [sd_id(m) for m in learn.get(n, [])])
             for n, p in profiles.items()}
    info = comp.get("info", {})
    st.caption(f"📊 {info.get('metagame', mf[1])} · {mf[0]} · rating ≥ {info.get('cutoff_used', cutoff)} · "
               f"{info.get('number of battles', 0):,} μάχες · {len(usage.w)} Pokémon με δεδομένα")

    with st.expander("🧠 Μοντέλο ML: πώς εκπαιδεύεται & πόσο καλά προβλέπει", expanded=False):
        n_train = st.slider("Pokémon για εκπαίδευση (τα πιο χρησιμοποιημένα)", 30, 200, 120, 10)
        model = trained_model(SRC, GAME, LEVEL, mf[0], mf[1], cutoff, kind, n_train, usage, profiles, feats)
        if model is None:
            st.warning("Πολύ λίγα δεδομένα για εκπαίδευση.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("R² (5-fold CV)", f"{model['r2']:.2f}")
            m2.metric("Spearman ρ", f"{model['spearman']:.2f}")
            m3.metric("MAE (log₂ lift)", f"{model['mae']:.2f}")
            m4.metric("Top-10% recall", f"{model['top10_recall'] * 100:.0f}%")
            st.caption(f"Στόχος: log₂ lift = πόσες φορές πιο συχνά από την τύχη εμφανίζονται μαζί δύο Pokémon · "
                       f"{model['n_pairs']:,} ζευγάρια από {model['n_mons']} Pokémon · "
                       "αξιολόγηση σε δεδομένα που το μοντέλο δεν είδε (cross-validation).")
            imp = model["importances"].head(15).sort_values()
            st.bar_chart(imp, horizontal=True)
            st.caption("Permutation importance: πόσο χειροτερεύει η πρόβλεψη αν «ανακατέψουμε» το χαρακτηριστικό.")

    default = next((n for n in ("Archaludon", *usage.top(1)) if n in profiles), list(profiles)[0])
    a = st.selectbox("Pokémon-άγκυρα", list(profiles), index=list(profiles).index(default), format_func=label,
                     key="mate_anchor")
    pa, fa = profiles[a], feats[a]
    sp_a = BY_NAME[a]["species"]
    cands = [b for b in profiles if b != a and BY_NAME[b]["species"] != sp_a]
    rows = [ml.pair_features(pa, profiles[b], fa, feats[b]) for b in cands]
    X = pd.DataFrame(rows)
    pred = model["model"].predict(X[model["columns"]]) if model else np.zeros(len(X))
    out = []
    for b, x, pr in zip(cands, rows, pred):
        real = usage.log_lift(a, b)
        final = pr if real is None else 0.7 * real + 0.3 * pr
        out.append({"Εικόνα": sprite(BY_NAME[b], small=True), "Pokémon": label(b),
                    "Σκορ": float(ml.to_score(final)),
                    "Μαζί σε ομάδες %": usage.teammate_pct(a, b) * 100 if real is not None else None,
                    "Lift (πραγμ.)": 2 ** real if real is not None else None,
                    "ML πρόβλεψη": float(ml.to_score(pr)),
                    "Heuristic": 100 * sum(x[k] * v for k, v in SYN_WEIGHTS.items()) / sum(SYN_WEIGHTS.values()),
                    "Γιατί": reasons(pa, profiles[b], fa, feats[b], x), "_name": b, "_final": final})
    res = pd.DataFrame(out).sort_values("Σκορ", ascending=False).reset_index(drop=True)
    res.index = res.index + 1
    det = usage.detail.get(a)
    if det:
        top_moves = ", ".join(f"{k} {v * 100:.0f}%" for k, v in list(det["moves"].items())[:6])
        top_items = ", ".join(f"{k} {v * 100:.0f}%" for k, v in list(det["items"].items())[:3])
        st.caption(f"**{a}** · usage {usage.usage(a) * 100:.1f}% · κινήσεις: {top_moves} · items: {top_items}")
    else:
        st.caption(f"Το {a} δεν έχει αρκετά δεδομένα στο ladder — οι προτάσεις βγαίνουν μόνο από το μοντέλο ML.")
    top_n = st.slider("Πόσους συμπαίκτες να δείξω", 5, 40, 15)
    show = res.head(top_n).drop(columns=["_name", "_final"])
    st.dataframe(show, width="stretch", height=min(38 + 35 * len(show), 720), column_config={
        "Εικόνα": st.column_config.ImageColumn("", width="small"),
        "Σκορ": st.column_config.ProgressColumn("Σκορ", min_value=0, max_value=100, format="%.0f"),
        "Μαζί σε ομάδες %": st.column_config.NumberColumn(format="%.1f%%",
                                                          help=f"Από τις ομάδες που έχουν {a}, πόσες έχουν και αυτό"),
        "Lift (πραγμ.)": st.column_config.NumberColumn(format="×%.2f", help="×1 = όσο συχνά θα περίμενε κανείς τυχαία"),
        "ML πρόβλεψη": st.column_config.NumberColumn(format="%.0f"),
        "Heuristic": st.column_config.NumberColumn(format="%.0f"),
        "Γιατί": st.column_config.TextColumn(width="large"),
    })
    st.caption("Σκορ 50 = όσο συχνά θα έμπαιναν μαζί τυχαία · >50 = ταιριάζουν · Τελικό σκορ = 70% πραγματικά "
               "δεδομένα + 30% ML (μόνο ML όταν λείπουν δεδομένα).")

    st.subheader("🧩 Φτιάξε 6άδα γύρω από το " + a)
    k1, k2 = st.columns(2)
    lam = k1.slider("Βάρος «viability» (πόσο δημοφιλές είναι στο ladder)", 0.0, 1.0, 0.3, 0.05)
    pool_n = k2.slider("Υποψήφιοι από τους καλύτερους συμπαίκτες", 10, 60, 30, 5)
    pool = [a] + res["_name"].head(pool_n).tolist()
    pair = {}
    feats_rows, keys = [], []
    for x, y_ in combinations(pool, 2):
        keys.append((x, y_))
        feats_rows.append(ml.pair_features(profiles[x], profiles[y_], feats[x], feats[y_]))
    P = model["model"].predict(pd.DataFrame(feats_rows)[model["columns"]]) if model else np.zeros(len(keys))
    for (x, y_), pr in zip(keys, P):
        real = usage.log_lift(x, y_)
        pair[frozenset((x, y_))] = pr if real is None else 0.7 * real + 0.3 * pr
    vmax = max((usage.usage(n) for n in pool), default=0) or 1
    team = [a]
    while len(team) < 6:
        best, best_v = None, -1e9
        for c in pool:
            if c in team or any(BY_NAME[c]["species"] == BY_NAME[t]["species"] for t in team):
                continue
            if BY_NAME[c].get("form") == "mega" and any(BY_NAME[t].get("form") == "mega" for t in team):
                continue  # one Mega per team
            v = np.mean([pair[frozenset((c, t))] for t in team]) + lam * usage.usage(c) / vmax
            if v > best_v:
                best, best_v = c, v
        if best is None:
            break
        team.append(best)
    st.markdown("<div class='pk-row' style='justify-content:flex-start'>" +
                "".join(box_html(BY_NAME[n], cur=(n == a)) for n in team) + "</div>", unsafe_allow_html=True)
    st.caption("Άπληστος αλγόριθμος: σε κάθε βήμα μπαίνει το Pokémon με το καλύτερο μέσο σκορ με όσα έχουν "
               "ήδη μπει (+ viability). Ένα Mega ανά ομάδα, όχι ίδιο είδος δύο φορές.")
    if st.button("➡️ Χρήση ως ομάδα στη σελίδα 🤝 Synergy & τετράδες"):
        st.session_state["team_names"] = team
        st.session_state["w_team_names"] = team
        st.success("Έτοιμο — άνοιξε τη σελίδα «🤝 Synergy & τετράδες» από το sidebar.")


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

**VGC συμπαίκτες (ML)**: από τα στατιστικά του Smogon (Showdown ladder, π.χ. `gen9championsvgc2026regmb`,
rating ≥ 1760) μετράμε πόσο συχνά δύο Pokémon είναι στην ίδια ομάδα. Στόχος του μοντέλου είναι το
$\log_2 \text{lift} = \log_2 \frac{c_{AB} + m}{E_{AB} + m}$, όπου $c_{AB}$ οι κοινές ομάδες,
$E_{AB} = N\,p_A\,p_B$ οι αναμενόμενες αν ήταν ανεξάρτητα και $m$ μικρό pseudo-count (smoothing).
Χαρακτηριστικά ζεύγους: αμυντική/επιθετική κάλυψη, ρόλοι, κοινοί τύποι, setter + abuser καιρού/terrain
(από abilities **και** κινήσεις, π.χ. Drizzle + Electro Shot), Trick Room + αργό, Tailwind, Follow Me,
Fake Out, Earthquake + ανοσία Ground, Intimidate, Speed, BST, τύποι. Μοντέλα: Gradient Boosting
(scikit-learn ή XGBoost), Random Forest, Ridge. Αξιολόγηση με 5-fold cross-validation (R², Spearman, MAE,
top-10% recall) και permutation importance. Τελικό σκορ = 0.7·πραγματικό + 0.3·ML (μόνο ML αν λείπουν δεδομένα).
Η 6άδα χτίζεται άπληστα: σε κάθε βήμα προστίθεται όποιο Pokémon έχει το μεγαλύτερο μέσο σκορ με την ομάδα.

**Σημειώσεις**: Το Legends Z-A έχει real-time μάχες (ο τύπος ζημιάς είναι προσέγγιση) και δεν έχει abilities.
Αν το PokeAPI δεν έχει ακόμα learnset για Z-A/Champions, χρησιμοποιείται του Scarlet/Violet.
""")


if PAGE_KEY != "dex":
    sync_url()
{"dex": page_pokedex, "dmg": page_damage, "rank": page_ranking,
 "mate": page_mates, "syn": page_synergy, "info": page_method}[PAGE_KEY]()
