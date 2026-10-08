import re
from datetime import date, timedelta

import pytest

from ranking.engine import Candidate, rank
from ranking.preferences import CRITERIA, Destination, Preferences

TODAY = date(2026, 10, 8)


def cand(i, **kw):
    base = dict(
        id=i, address=f"{i} Test St", price=400_000, beds=3, baths=2.0, sqft=1800,
        lot_sqft=8000, year_built=2000, property_type="Single Family", status="active",
        lat=33.75, lon=-84.39, list_date=TODAY - timedelta(days=10),
    )  # fmt: skip
    base.update(kw)
    return Candidate(**base)


def prefs(**kw):
    base = dict(max_price=450_000, min_beds=3, min_baths=2, min_sqft=1500, ideal_sqft=2000)
    base.update(kw)
    return Preferences(**base)


def run(cands, p, **kw):
    return rank(cands, p, today=TODAY, **kw)


# ------------------------------------------------------------------ preferences are strict


def test_unknown_preference_fields_are_rejected():
    with pytest.raises(ValueError, match="Unknown preference"):
        Preferences.from_dict({"max_price": 1, "median_income": 100})


@pytest.mark.parametrize("bad", ["demographics", "median_income", "crime", "school_rank", "race"])
def test_unknown_ranking_criteria_are_rejected(bad):
    with pytest.raises(ValueError, match="Unknown ranking criteria"):
        Preferences(weights={bad: 5}).validate()


def test_weights_and_cutoffs_are_validated():
    with pytest.raises(ValueError, match="between 0 and 10"):
        Preferences(weights={"price_fit": 11}).validate()
    with pytest.raises(ValueError, match="cutoffs"):
        Preferences(tier_cutoffs={"S": 50, "A": 60, "B": 40, "C": 30}).validate()
    with pytest.raises(ValueError, match="condition_mode"):
        Preferences(condition_mode="whatever").validate()
    with pytest.raises(ValueError, match="Minimum price"):
        Preferences(min_price=500, max_price=400).validate()


def test_preferences_round_trip():
    p = prefs(
        destinations=[Destination("Work", 33.8, -84.4, max_minutes=30)],
        condition_mode="move_in_ready",
    )
    again = Preferences.from_dict(p.to_dict())
    assert again == p


def test_ranking_data_model_has_no_demographic_style_fields():
    banned = {
        "race",
        "ethnicity",
        "religion",
        "income",
        "demographic",
        "crime",
        "school",
        "population",
        "census",
    }
    names = (
        set(Candidate.__dataclass_fields__) | set(Preferences.__dataclass_fields__) | set(CRITERIA)
    )
    assert not [n for n in names if any(b in n.lower() for b in banned)]


# ------------------------------------------------------------------ hard requirements


def test_listings_failing_requirements_are_excluded_with_reasons():
    cands = [
        cand(1),
        cand(2, price=500_000),
        cand(3, beds=2),
        cand(4, baths=1.0),
        cand(5, sqft=1200),
        cand(6, property_type="Condo"),
        cand(7, status="pending"),
        cand(8, year_built=1950),
    ]
    res = run(cands, prefs(property_types=["Single Family"], min_year_built=1980))
    assert [r.candidate.id for r in res.ranked] == [1]
    why = {r.candidate.id: " ".join(r.excluded_reasons) for r in res.excluded}
    assert "over the $450,000" in why[2] and "needs 3+" in why[3] and "1 bathroom;" in why[4]
    assert "sq ft" in why[5] and "Condo" in why[6] and "pending" in why[7] and "1950" in why[8]
    assert all(r.tier == "X" for r in res.excluded)


def test_missing_facts_do_not_exclude_but_are_flagged():
    res = run([cand(1, beds=None, price=None)], prefs())
    r = res.ranked[0]
    assert any("Bedrooms not listed" in u for u in r.unknowns)
    assert any("Price not listed" in u for u in r.unknowns)


# ------------------------------------------------------------------ scoring behaviour


def test_cheaper_and_roomier_homes_score_higher():
    cands = [cand(1, price=440_000, sqft=1550), cand(2, price=380_000, sqft=2100)]
    res = run(cands, prefs())
    assert [r.candidate.id for r in res.ranked] == [2, 1]
    assert res.ranked[0].score > res.ranked[1].score


def test_tiers_follow_cutoffs():
    great = cand(1, price=350_000, beds=5, baths=3.5, sqft=2400)
    ok = cand(2, price=449_000, beds=3, baths=2.0, sqft=1500)
    res = run([great, ok], prefs())
    tiers = {r.candidate.id: r.tier for r in res.ranked}
    assert tiers[1] in {"S", "A"} and tiers[2] in {"B", "C", "D"}
    assert res.ranked[0].score >= res.ranked[1].score


def test_score_is_renormalized_when_a_criterion_has_no_data():
    full = run([cand(1)], prefs()).ranked[0]
    partial = run([cand(1, sqft=None)], prefs()).ranked[0]
    assert partial.coverage < 1.0 <= full.coverage + 1e-9
    assert partial.score is not None  # re-weighted over what is known, not zeroed


def test_not_enough_data_gives_question_mark_tier_not_a_guess():
    bare = Candidate(id=1, address="1 Bare St")  # nothing known
    res = run([bare], prefs())
    assert res.ranked[0].tier == "?" and res.ranked[0].score is None


def test_value_needs_enough_comparable_homes():
    few = run([cand(i) for i in range(1, 4)], prefs())
    assert not any(f.key == "value" for f in few.ranked[0].factors)
    many = [cand(i, price=300_000 + i * 20_000) for i in range(1, 9)]
    res = run(many, prefs(max_price=600_000))
    assert any(f.key == "value" and f.score is not None for f in res.ranked[0].factors)
    cheapest = next(r for r in res.ranked if r.candidate.id == 1)
    priciest = next(r for r in res.ranked if r.candidate.id == 8)
    v = lambda r: next(f.score for f in r.factors if f.key == "value")  # noqa: E731
    assert v(cheapest) > v(priciest)


# ------------------------------------------------------------------ commute


def test_commute_uses_client_named_places_and_limits():
    near, far = cand(1, lat=33.76, lon=-84.39), cand(2, lat=34.20, lon=-84.39)
    work = Destination("Work", 33.75, -84.39, max_minutes=60)
    res = run([near, far], prefs(destinations=[work]))
    ids = [r.candidate.id for r in res.ranked]
    assert ids[0] == 1
    near_f = next(f for f in res.ranked[0].factors if f.key == "commute")
    assert "min to Work" in near_f.detail and "estimate" in near_f.detail


def test_commute_limit_excludes_and_missing_coordinates_are_unknown():
    work = Destination("Work", 33.75, -84.39, max_minutes=15)
    far = cand(1, lat=34.2, lon=-84.39)
    unmapped = cand(2, lat=None, lon=None)
    res = run([far, unmapped], prefs(destinations=[work]), travel=lambda *a: 99.0)
    assert [r.candidate.id for r in res.excluded] == [1]
    assert [r.candidate.id for r in res.ranked] == [2]
    assert any("No data for" in u for u in res.ranked[0].unknowns)


# ------------------------------------------------------------------ condition modes


def test_condition_modes_change_what_is_preferred():
    nice, rough = (
        cand(1, condition=4.8, condition_photos=12),
        cand(2, condition=1.8, condition_photos=12),
    )

    def order(mode):
        res = run([nice, rough], prefs(condition_mode=mode))
        return [r.candidate.id for r in res.ranked], res

    assert order("move_in_ready")[0] == [1, 2]
    assert order("renovation_ok")[0] == [2, 1]
    ids, res = order("neutral")
    assert all(f.key != "condition" for r in res.ranked for f in r.factors)


def test_condition_factor_reports_photo_basis():
    res = run([cand(1, condition=4.0, condition_photos=9)], prefs(condition_mode="move_in_ready"))
    f = next(f for f in res.ranked[0].factors if f.key == "condition")
    assert "4.0/5 from 9 photo(s)" in f.detail


# ------------------------------------------------------------------ explanations


STEERING_WORDS = re.compile(
    r"\b(safe|unsafe|dangerous|desirable|undesirable|exclusive|upscale|prestigious|quiet|"
    r"family|families|diverse|diversity|gated|ghetto|sketchy|nice neighborhood|good neighborhood|"
    r"bad neighborhood|good schools|bad schools)\b",
    re.I,
)


def test_generated_text_is_factual_and_free_of_steering_language():
    work = Destination("Work", 33.80, -84.40, max_minutes=45)
    pool = [
        cand(
            i,
            price=300_000 + i * 15_000,
            condition=1 + i % 5,
            condition_photos=6,
            lat=33.7 + i / 100,
        )
        for i in range(1, 12)
    ]
    res = run(
        pool,
        prefs(
            max_price=600_000,
            destinations=[work],
            condition_mode="move_in_ready",
            ideal_lot_sqft=9000,
        ),
    )
    texts = []
    for r in res.ranked + res.excluded:
        texts += r.strengths + r.watchouts + r.unknowns + r.excluded_reasons
        texts += [f.detail for f in r.factors] + [f.label for f in r.factors]
    assert texts
    assert not [t for t in texts if STEERING_WORDS.search(t)]


def test_negotiation_signal_is_opt_in():
    old = cand(1, list_date=TODAY - timedelta(days=120))
    default = run([old], prefs()).ranked[0]
    assert all(f.key != "negotiation" for f in default.factors)
    opted = run([old], prefs(weights={"negotiation": 5})).ranked[0]
    f = next(f for f in opted.factors if f.key == "negotiation")
    assert f.score == 1.0 and "120 days on market" in f.detail


def test_exclusion_text_uses_correct_plurals():
    res = run([cand(1, beds=1, baths=1.0), cand(2, beds=2, baths=1.5)], prefs())
    why = {r.candidate.id: " ".join(r.excluded_reasons) for r in res.excluded}
    assert "1 bedroom;" in why[1] and "1 bathroom;" in why[1]
    assert "2 bedrooms;" in why[2] and "1.5 bathrooms;" in why[2]
