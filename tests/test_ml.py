"""Synthetic check: the ML model must discover a planted rain setter + rain abuser synergy."""
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import ml  # noqa: E402
from core.analysis import build_profile  # noqa: E402
from core.typechart import TYPES  # noqa: E402
from core.usage import Usage, compact_chaos  # noqa: E402


def synthetic(n_mons=60, n_teams=6000, seed=0):
    rnd = random.Random(seed)
    names = [f"Mon{i}" for i in range(n_mons)]
    setters, abusers = set(names[:4]), set(names[4:12])
    teams = []
    for _ in range(n_teams):
        t = set(rnd.sample(names, 6))
        if t & abusers and rnd.random() < 0.7:          # rain abusers bring a rain setter
            t.discard(rnd.choice(sorted(t - abusers) or sorted(t)))
            t.add(rnd.choice(sorted(setters)))
        teams.append(t)
    data = {}
    for n in names:
        with_n = [t for t in teams if n in t]
        tm = {}
        for t in with_n:
            for o in t - {n}:
                tm[o] = tm.get(o, 0) + 1
        data[n] = {"Abilities": {"drizzle" if n in setters else "pressure": len(with_n)},
                   "Moves": {"electroshot" if n in abusers else "tackle": len(with_n), "protect": len(with_n)},
                   "Items": {"leftovers": len(with_n)}, "Spreads": {}, "Teammates": tm}
    profiles = {}
    for i, n in enumerate(names):
        p = {"name": n, "display": n, "types": [TYPES[i % 18]], "abilities": ["pressure"],
             "base": {k: 60 + (i * 7 + j * 13) % 60 for j, k in enumerate(["hp", "atk", "def", "spa", "spd", "spe"])}}
        profiles[n] = build_profile(p, 50, "recommended", [], {})
    return compact_chaos({"info": {"metagame": "test"}, "data": data}), profiles, setters, abusers


def test_usage_and_model():
    comp, profiles, setters, abusers = synthetic()
    u = Usage(comp, lambda n: n)
    s, a = sorted(setters)[0], sorted(abusers)[0]
    o = sorted(set(profiles) - setters - abusers)[0]
    assert u.log_lift(s, a) > 0.2 > u.log_lift(s, o)    # planted pairs co-occur more than chance
    assert abs(u.teammate_pct(s, a) - u.joint[frozenset((s, a))] / u.w[s]) < 1e-12
    feats = {n: ml.mon_features(p, u.detail.get(n), []) for n, p in profiles.items()}
    assert feats[s]["set_rain"] == 1.0 and feats[a]["abuse_rain"] >= 1.0
    X, y, w, idx = ml.training_set(list(profiles), profiles, feats, u)
    planted = np.array([(p in setters and q in abusers) or (q in setters and p in abusers) for p, q in idx])
    for kind in ("hgb", "rf", "ridge"):
        res = ml.train_and_evaluate(X, y, w, kind, folds=3)
        # out-of-fold predictions rank the planted pairs clearly above the rest
        assert res["cv_pred"][planted].mean() > np.percentile(res["cv_pred"], 75), kind
        assert res["importances"].index[0] == "weather_combo", (kind, res["importances"].head())
    x = ml.pair_features(profiles[s], profiles[a], feats[s], feats[a])
    import pandas as pd
    pr = res["model"].predict(pd.DataFrame([x])[res["columns"]])[0]
    assert pr > np.median(y)


if __name__ == "__main__":
    test_usage_and_model()
    print("✓ test_usage_and_model")
