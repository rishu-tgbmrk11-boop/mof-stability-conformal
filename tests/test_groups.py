import pandas as pd

from src.features.groups import assign_groups, build_groups, family_key, random_assign, size_summary


def _toy():
    names = ["ABCDAA", "ABCDBB", "XYZWAA", "XYZWBB", "QQQQAA", "MNOPAA", "RSTUAA"]
    doi = {"QQQQAA": "10.1/x", "MNOPAA": "10.1/x", "RSTUAA": "10.1/y"}
    return names, doi


def test_family_and_doi_links_join_groups():
    names, doi = _toy()
    g = build_groups(names, doi, k=4).set_index("name")
    assert g.loc["ABCDAA", "group_id"] == g.loc["ABCDBB", "group_id"]        # same 4-letter family
    assert g.loc["QQQQAA", "group_id"] == g.loc["MNOPAA", "group_id"]        # same DOI, different family
    assert g.loc["RSTUAA", "group_id"] != g.loc["QQQQAA", "group_id"]
    assert g.loc["ABCDAA", "group_size"] == 2 and g.loc["QQQQAA", "group_size"] == 2


def test_chain_through_doi_and_family_is_one_component():
    names = ["AAAAXX", "AAAAYY", "BBBBXX", "CCCCXX"]
    g = build_groups(names, {"AAAAXX": "d1", "BBBBXX": "d1"}, k=4).set_index("name")   # AAAA family + DOI d1 links AAAA* with BBBB*
    assert g["group_id"].nunique() == 2 and g.loc["AAAAYY", "group_id"] == g.loc["BBBBXX", "group_id"]


def test_assign_keeps_groups_whole_and_fractions_close():
    names = [f"{a}{b}{c}{d}{i:02d}" for a in "AB" for b in "CDE" for c in "FG" for d in "HIJ" for i in range(3)]
    g = build_groups(names, None, k=4)
    s = assign_groups(g, {"train": 0.6, "cal": 0.2, "test": 0.2}, seed=0)
    chk = pd.concat([g, s], axis=1).groupby("group_id")["split"].nunique()
    assert (chk == 1).all()                                  # every group wholly in one split
    frac = s.value_counts(normalize=True)
    assert abs(frac["train"] - 0.6) < 0.08 and abs(frac["test"] - 0.2) < 0.08
    assert (assign_groups(g, {"train": 0.6, "cal": 0.2, "test": 0.2}, seed=0) == s).all()   # deterministic


def test_random_assign_sizes():
    s = random_assign([f"M{i}" for i in range(100)], {"train": 0.6, "cal": 0.2, "test": 0.2}, seed=1)
    assert s.value_counts().to_dict() == {"train": 60, "cal": 20, "test": 20}


def test_size_summary_totals():
    names, doi = _toy()
    g = build_groups(names, doi)
    t = size_summary(g)
    assert t["mofs"].sum() == len(names) and t["groups"].sum() == g["group_id"].nunique()
