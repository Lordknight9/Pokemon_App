"""ML synergy model: learn from real VGC teams which Pokémon pairs work together.

Target  : smoothed log-lift of two Pokémon appearing on the same team (Smogon ladder stats).
Features: type/stat complementarity, weather & terrain setter–abuser links, Trick Room /
          Tailwind / redirection / Fake Out support, Earthquake + Ground-immune partner, …
          built from each Pokémon's real moves & abilities (usage data) or, if it has no
          usage data, from its learnset / ability list.
Model   : gradient-boosted trees (XGBoost if installed, else scikit-learn), Random Forest
          or Ridge regression as alternatives.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .analysis import SUPPORT, TERRAIN_SET, WEATHER_ABUSE, WEATHER_SET, WEATHER_TYPE, pair_components
from .typechart import TYPES

WEATHER_MOVES = {"rain": {"electroshot", "thunder", "hurricane", "weatherball", "wavecrash"},
                 "sun": {"solarbeam", "solarblade", "weatherball", "hydrosteam"},
                 "sand": {"weatherball", "shoreup"},
                 "snow": {"blizzard", "auroraveil", "weatherball"}}
MOVE_GROUPS = {
    "fakeout": {"fakeout"}, "tailwind": {"tailwind"}, "trickroom": {"trickroom"},
    "redirect": {"followme", "ragepowder"}, "helpinghand": {"helpinghand"},
    "screens": {"reflect", "lightscreen", "auroraveil"}, "allyheal": {"pollenpuff", "healpulse", "lifedew"},
    "spread_ground": {"earthquake", "bulldoze"}, "spread_electric": {"discharge"}, "spread_water": {"surf"},
    "setup": {"swordsdance", "nastyplot", "dragondance", "calmmind", "bellydrum", "quiverdance", "bulkup"},
}
GROUND_IMMUNE_AB = {"levitate", "earthmaker", "earth-eater", "eartheater"}
ELEC_IMMUNE_AB = {"voltabsorb", "lightningrod", "motordrive"}
WATER_IMMUNE_AB = {"waterabsorb", "stormdrain", "dryskin"}


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def mon_features(profile: dict, usage_detail: dict | None, learnset: list[str]) -> dict:
    """Per-Pokémon descriptors (probabilities in [0,1])."""
    if usage_detail:
        moves = {_norm(k): v for k, v in usage_detail.get("moves", {}).items()}
        abil = {_norm(k): v for k, v in usage_detail.get("abilities", {}).items()}
    else:
        moves = {_norm(m): 0.25 for m in learnset}
        n = max(len(profile["abilities"]), 1)
        abil = {_norm(a): 1 / n for a in profile["abilities"]}
    f = {"has_usage": 1.0 if usage_detail else 0.0}
    for g, ms in MOVE_GROUPS.items():
        f[g] = min(1.0, sum(moves.get(m, 0) for m in ms))
    for w in WEATHER_ABUSE:
        f[f"set_{w}"] = min(1.0, sum(p for a, p in abil.items() if WEATHER_SET.get(_dash(a)) == w))
        f[f"abuse_{w}"] = min(1.0, sum(p for a, p in abil.items() if _dash(a) in WEATHER_ABUSE[w])
                              + 0.5 * (WEATHER_TYPE[w] in profile["types"])
                              + sum(moves.get(m, 0) for m in WEATHER_MOVES[w]))
    for t in set(TERRAIN_SET.values()):
        f[f"terrain_{t}"] = min(1.0, sum(p for a, p in abil.items() if TERRAIN_SET.get(_dash(a)) == t))
    f["intimidate"] = abil.get("intimidate", 0.0)
    f["support_ab"] = min(1.0, sum(p for a, p in abil.items() if _dash(a) in SUPPORT))
    f["ground_immune"] = max(1.0 if "flying" in profile["types"] else 0.0,
                             sum(p for a, p in abil.items() if a in GROUND_IMMUNE_AB))
    f["elec_immune"] = max(1.0 if "ground" in profile["types"] else 0.0,
                           sum(p for a, p in abil.items() if a in ELEC_IMMUNE_AB))
    f["water_immune"] = sum(p for a, p in abil.items() if a in WATER_IMMUNE_AB)
    s = profile["stats"]
    f["spe"] = s["spe"]
    f["slow"] = 1.0 if profile["base"]["spe"] < 60 else 0.0
    f["offense"] = max(profile["base"]["atk"], profile["base"]["spa"]) / 150
    f["bst"] = profile["bst"]
    return f


_ABIL_DASH = {_norm(k): k for k in list(WEATHER_SET) + list(TERRAIN_SET) + [a for s in WEATHER_ABUSE.values()
                                                                            for a in s] + list(SUPPORT)}


def _dash(a: str) -> str:
    return _ABIL_DASH.get(a, a)


def pair_features(pa: dict, pb: dict, fa: dict, fb: dict) -> dict:
    c = pair_components(pa, pb)
    x = {k: c[k] for k in ("defense", "offense", "roles", "type_div", "abilities", "cover_pairs", "shared_weak")}
    x["weather_combo"] = max(fa[f"set_{w}"] * fb[f"abuse_{w}"] + fb[f"set_{w}"] * fa[f"abuse_{w}"]
                             for w in WEATHER_ABUSE)
    x["weather_conflict"] = sum(fa[f"set_{w1}"] * fb[f"set_{w2}"] for w1 in WEATHER_ABUSE for w2 in WEATHER_ABUSE
                                if w1 != w2)
    x["terrain_combo"] = max(fa[f"terrain_{t}"] * (t in pb["types"]) + fb[f"terrain_{t}"] * (t in pa["types"])
                             for t in set(TERRAIN_SET.values()))
    x["trickroom_combo"] = fa["trickroom"] * fb["slow"] + fb["trickroom"] * fa["slow"]
    x["tailwind_combo"] = fa["tailwind"] * (1 - fb["slow"]) + fb["tailwind"] * (1 - fa["slow"])
    x["redirect_combo"] = fa["redirect"] * fb["offense"] + fb["redirect"] * fa["offense"]
    x["redirect_setup"] = fa["redirect"] * fb["setup"] + fb["redirect"] * fa["setup"]
    x["fakeout_sum"] = fa["fakeout"] + fb["fakeout"]
    x["helpinghand_combo"] = fa["helpinghand"] * fb["offense"] + fb["helpinghand"] * fa["offense"]
    x["screens_sum"] = fa["screens"] + fb["screens"]
    x["allyheal"] = fa["allyheal"] + fb["allyheal"]
    x["eq_immune"] = fa["spread_ground"] * fb["ground_immune"] + fb["spread_ground"] * fa["ground_immune"]
    x["elec_immune"] = fa["spread_electric"] * fb["elec_immune"] + fb["spread_electric"] * fa["elec_immune"]
    x["water_immune"] = fa["spread_water"] * fb["water_immune"] + fb["spread_water"] * fa["water_immune"]
    x["intimidate_sum"] = fa["intimidate"] + fb["intimidate"]
    x["support_sum"] = fa["support_ab"] + fb["support_ab"]
    x["spe_min"], x["spe_max"] = sorted((fa["spe"], fb["spe"]))
    x["both_slow"] = fa["slow"] * fb["slow"]
    x["bst_min"], x["bst_max"] = sorted((fa["bst"], fb["bst"]))
    x["usage_data"] = fa["has_usage"] + fb["has_usage"]
    for t in TYPES:
        x[f"type_{t}"] = (t in pa["types"]) + (t in pb["types"])
    return x


def training_set(names: list, profiles: dict, feats: dict, usage) -> tuple:
    """Pairs of Pokémon that both have usage data -> (X, y, sample_weight, pair_index)."""
    rows, y, w, idx = [], [], [], []
    for a, b in combinations(names, 2):
        ll = usage.log_lift(a, b)
        if ll is None:
            continue
        rows.append(pair_features(profiles[a], profiles[b], feats[a], feats[b]))
        y.append(ll)
        w.append(np.sqrt(min(usage.w[a], usage.w[b])))
        idx.append((a, b))
    return pd.DataFrame(rows), np.array(y), np.array(w), idx


def available_models() -> dict:
    out = {}
    try:
        import xgboost  # noqa: F401
        out["xgboost"] = "XGBoost (gradient boosting)"
    except Exception:
        pass
    out["hgb"] = "Gradient Boosting (scikit-learn HistGradientBoosting)"
    out["rf"] = "Random Forest"
    out["ridge"] = "Ridge regression (γραμμικό baseline)"
    return out


def make_model(kind: str):
    if kind == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, min_child_weight=5,
                            colsample_bytree=0.8, reg_lambda=2.0, n_jobs=2, random_state=0)
    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=10, max_features=0.5, n_jobs=2,
                                     random_state=0)
    if kind == "ridge":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=4, min_samples_leaf=30,
                                         l2_regularization=1.0, random_state=0)


def train_and_evaluate(X: pd.DataFrame, y, w, kind: str = "hgb", folds: int = 5) -> dict:
    """Cross-validated metrics, final model on all data and permutation importances."""
    from scipy.stats import spearmanr
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import KFold

    pred = np.zeros_like(y, dtype=float)
    for tr, te in KFold(folds, shuffle=True, random_state=0).split(X):
        m = make_model(kind)
        _fit(m, X.iloc[tr], y[tr], w[tr])
        pred[te] = m.predict(X.iloc[te])
    model = make_model(kind)
    _fit(model, X, y, w)
    sub = np.random.default_rng(0).choice(len(X), size=min(len(X), 2500), replace=False)
    imp = permutation_importance(model, X.iloc[sub], y[sub], n_repeats=2, random_state=0, n_jobs=1)
    importances = pd.Series(imp.importances_mean, index=X.columns).sort_values(ascending=False)
    top = np.argsort(-y)[: max(1, len(y) // 10)]
    return {
        "model": model, "columns": list(X.columns),
        "r2": float(r2_score(y, pred)), "mae": float(mean_absolute_error(y, pred)),
        "spearman": float(spearmanr(y, pred).statistic),
        # how many of the real top-10% pairs are also predicted in the top 10%
        "top10_recall": float(len(set(top) & set(np.argsort(-pred)[: len(top)])) / len(top)),
        "importances": importances, "cv_pred": pred,
    }


def _fit(model, X, y, w):
    if hasattr(model, "steps"):  # sklearn Pipeline: sample_weight goes to the last step
        model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": w})
    else:
        model.fit(X, y, sample_weight=w)


def to_score(log_lift) -> np.ndarray:
    """log2-lift -> 0..100 (50 = paired as often as chance)."""
    return np.clip(50 + 25 * np.asarray(log_lift, dtype=float), 0, 100)
