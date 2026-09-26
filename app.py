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


@st.cache_data(show_spinner="Λήψη λεπτομερειών κινήσεων…")
def get_moves_many(source: str, names: tuple) -> dict:
    return get_provider(source).moves_many(list(names))


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
         "syn": "🤝 Synergy & τετράδες", "info": "ℹ️ Μεθοδολογία"}
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
        st.dataframe(res.round(4), width="stretch")
    with c2:
        st.bar_chart(res[score_col].sort_values(ascending=True), horizontal=True)
    if method == "Και τα δύο":
        rho = spearman(res["Κατάταξη TOPSIS"], res["Κατάταξη PROMETHEE"])
        st.info(f"Συσχέτιση Spearman ανάμεσα σε TOPSIS και PROMETHEE: **ρ = {rho:.3f}**")
    st.download_button(f"⬇️ Αποτελέσματα {label} (CSV)", res.to_csv().encode("utf-8-sig"),
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
                     "Prio": d.get("priority"), "Περιγραφή": d.get("desc", "")})
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
    mv = Move(mname, md["type"], power, md["category"], spread)
    r = calc(A, D, mv, fld)
    hp = D.stats["hp"]
    cur = max(1, round(hp * D.hp_pct / 100))

    mtype = r.get("type", md["type"])
    tc = TYPE_COLOR.get(mtype, "#888")
    st.markdown(f"### {A.name} — <span style='color:{tc}'>**{pretty(mname)}**</span> {badge(mtype)}"
                f"{cat_badge(md['category'])} {power} BP → {D.name}", unsafe_allow_html=True)
    if md.get("desc"):
        st.caption(md["desc"])
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


if PAGE_KEY != "dex":
    sync_url()
{"dex": page_pokedex, "dmg": page_damage, "rank": page_ranking,
 "syn": page_synergy, "info": page_method}[PAGE_KEY]()
