import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import donuts as dn  # noqa: E402


def test_evaluate_and_stars():
    e = dn.evaluate({"Hyper Haban": 8})           # 8 x (85 sweet + 65 fresh) = 1200
    assert e["score"] == 1200 and e["stars"] == 5 and e["mult"] == 1.5
    assert e["calories"] == int(370 * 8 * 1.5) and e["level"] == int(8 * 8 * 1.5)
    assert e["possible"]["Sparkling"] == 3 and e["possible"]["Catch"] == 3 and e["possible"]["Item"] == 0
    assert dn.evaluate({"Cheri": 3})["stars"] == 0
    assert [int(dn.stars_of(x)) for x in (119, 120, 239, 240, 350, 700, 959, 960)] == [0, 1, 1, 2, 3, 4, 4, 5]


def test_search_respects_targets_and_constraints():
    res = dn.search({"Sparkling": 3, "Size": 3}, n=8, min_stars=5, top=3)
    assert res and all(r["stars"] == 5 and dn.feasible(r, {"Sparkling": 3, "Size": 3}) for r in res)
    assert all(r["order"][0] == "Sweet" for r in res)
    reg = [b for b in dn.NAMES if not dn.BERRIES[b]["hyper"]]
    r = dn.search({}, n=8, exact_stars=2, dominant="Spicy", allowed=reg, top=1)[0]
    assert r["stars"] == 2 and all(not dn.BERRIES[b]["hyper"] for b in r["recipe"]) and r["n"] == 8
    assert dn.search({"Item": 3, "Sparkling": 3, "Catch": 3, "Move": 1}, n=8) == []   # > 3 powers


def test_special_donuts_solvable_with_8_berries():
    for d in dn.SPECIAL_DONUTS:
        res = dn.search_special(d["req"], n=8, top=1)
        assert res, d["name"]
        r = res[0]
        assert r["n"] == 8 and all(r["flavors"][f] >= v for f, v in d["req"].items()), d["name"]
    # Game8's Bad Dreams Cruller recipe meets the requirements with our berry data
    ev = dn.evaluate({"Hyper Tanga": 3, "Hyper Kasib": 3, "Hyper Coba": 1, "Hyper Yache": 1})
    assert all(ev["flavors"][f] >= v for f, v in dn.SPECIAL_DONUTS[0]["req"].items())


if __name__ == "__main__":
    test_special_donuts_solvable_with_8_berries()
    test_evaluate_and_stars()
    test_search_respects_targets_and_constraints()
    print("✓ donuts")
