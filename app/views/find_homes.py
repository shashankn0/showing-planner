"""Find Homes: describe what the client wants, get a tier list with the reasons."""

from __future__ import annotations

import streamlit as st

from app import services
from app.ui import context, disclaimer, md, thumbnail
from app.views.listings import demo_button
from ranking.engine import Ranked, RankingResult
from ranking.preferences import (
    CONDITION_MODES,
    CRITERIA,
    DEFAULT_WEIGHTS,
    Destination,
    Preferences,
)

TIER_COLORS = {"S": "green", "A": "blue", "B": "violet", "C": "orange", "D": "red", "?": "gray"}
TIER_BLURBS = {
    "S": "Best matches",
    "A": "Strong matches",
    "B": "Good matches",
    "C": "Partial matches",
    "D": "Weak matches",
    "?": "Not enough data to place in a tier",
}
STATUS_CHOICES = ["active", "coming_soon", "pending"]
N_PLACES = 3


def render() -> None:
    settings = context()
    st.title("Find Homes")
    disclaimer()
    st.caption(
        "Enter what the client is looking for. Listings you've loaded are ranked into tiers using "
        "only the listing facts and the criteria below, and every card shows why. Tiers do not "
        "use neighborhood demographics."
    )
    pool = services.candidates_from_db(settings)
    if not pool:
        st.info(
            "No listings to rank yet. Load some on the Listings page (upload a file, search, or "
            "paste addresses), or try the demo first."
        )
        demo_button(settings, key="demo_listings_home")
        return

    p0, key_prefix = _starting_prefs(settings)
    type_options = sorted({c.property_type for c in pool if c.property_type})
    submitted, built = _form(p0, key_prefix, type_options)

    if submitted and built is not None:
        prefs, places, save_name = built
        dests: list[Destination] = []
        for name, address, max_min in places:
            dest, msg = services.resolve_destination(
                settings, name, address, max_minutes=max_min or None
            )
            if dest is None:
                st.warning(msg)
            else:
                dests.append(dest)
        prefs.destinations = dests
        try:
            prefs.validate()
        except ValueError as exc:
            st.error(str(exc))
            return
        st.session_state["ranking"] = (services.run_ranking(settings, prefs), prefs)
        if save_name.strip():
            services.save_client(settings, save_name, prefs)
            st.toast(f"Saved criteria for “{save_name.strip()}”.")

    shown = st.session_state.get("ranking")
    if shown:
        _show(settings, *shown)


# ------------------------------------------------------------------ form


def _starting_prefs(settings) -> tuple[Preferences, str]:  # noqa: ANN001
    clients = services.list_clients(settings)
    options: list[int | None] = [None, *[cid for cid, _ in clients]]
    names = dict(clients)
    chosen = st.selectbox(
        "Start from a saved client",
        options,
        format_func=lambda i: "New search" if i is None else names[i],
    )
    prefs = services.load_client_prefs(settings, chosen) if chosen else None
    # widget keys include the client so switching clients reloads the form's defaults
    return prefs or Preferences(), f"c{chosen or 0}_"


def _num(col, label: str, value, key: str, **kw):  # noqa: ANN001, ANN202
    return col.number_input(label, value=value, key=key, **kw)


def _form(p0: Preferences, kp: str, type_options: list[str]):  # noqa: ANN202
    with st.form("criteria"):
        st.subheader("Budget")
        c1, c2, c3 = st.columns(3)
        max_price = _num(
            c1, "Maximum price ($)", int(p0.max_price or 0), kp + "maxp", min_value=0, step=10_000
        )
        target = _num(
            c2,
            "Ideal price ($, optional)",
            int(p0.target_price or 0),
            kp + "tgt",
            min_value=0,
            step=10_000,
        )
        min_price = _num(
            c3,
            "Minimum price ($, optional)",
            int(p0.min_price or 0),
            kp + "minp",
            min_value=0,
            step=10_000,
        )

        st.subheader("The home")
        c1, c2, c3, c4 = st.columns(4)
        min_beds = _num(
            c1, "Bedrooms (at least)", int(p0.min_beds or 0), kp + "beds", min_value=0, max_value=12
        )
        min_baths = _num(
            c2,
            "Bathrooms (at least)",
            float(p0.min_baths or 0.0),
            kp + "baths",
            min_value=0.0,
            max_value=12.0,
            step=0.5,
        )
        min_sqft = _num(
            c3, "Sq ft (at least)", int(p0.min_sqft or 0), kp + "sqft", min_value=0, step=100
        )
        ideal_sqft = _num(
            c4, "Sq ft (ideal)", int(p0.ideal_sqft or 0), kp + "isqft", min_value=0, step=100
        )
        c1, c2 = st.columns(2)
        ideal_lot = _num(
            c1,
            "Lot size, sq ft (ideal, optional)",
            int(p0.ideal_lot_sqft or 0),
            kp + "lot",
            min_value=0,
            step=500,
        )
        min_year = _num(
            c2,
            "Built in or after (optional)",
            int(p0.min_year_built or 0),
            kp + "year",
            min_value=0,
            max_value=2100,
        )
        types = st.multiselect(
            "Property types (none selected = any)",
            type_options,
            default=[t for t in p0.property_types if t in type_options],
            key=kp + "types",
        )
        statuses = st.multiselect(
            "Listing status",
            STATUS_CHOICES,
            default=[s for s in p0.statuses if s in STATUS_CHOICES],
            key=kp + "status",
        )

        st.subheader("Places they'll drive to")
        st.caption(
            "Optional. Work, family, anywhere the client names. Drive times are straight-line estimates for now."
        )
        places = []
        for i in range(N_PLACES):
            d0 = p0.destinations[i] if i < len(p0.destinations) else None
            c1, c2, c3 = st.columns([2, 4, 2])
            name = c1.text_input(
                "Name", value=d0.name if d0 else "", key=f"{kp}pname{i}", placeholder="Work"
            )
            addr = c2.text_input(
                "Address",
                value=d0.address if d0 else "",
                key=f"{kp}paddr{i}",
                placeholder="123 Main St, Atlanta, GA 30303",
            )
            mx = _num(
                c3,
                "Limit (min, 0 = none)",
                int(d0.max_minutes or 0) if d0 else 0,
                f"{kp}pmax{i}",
                min_value=0,
                max_value=240,
                step=5,
            )
            if addr.strip():
                places.append((name or f"Place {i + 1}", addr, float(mx)))

        st.subheader("Condition")
        modes = list(CONDITION_MODES)
        mode = st.radio(
            "What did the client say about condition?",
            modes,
            index=modes.index(p0.condition_mode),
            format_func=CONDITION_MODES.get,
            key=kp + "mode",
            horizontal=True,
        )

        with st.expander("How much each factor matters (0 = ignore, 10 = critical)"):
            weights = {}
            cols = st.columns(3)
            for i, (k, label) in enumerate(CRITERIA.items()):
                weights[k] = cols[i % 3].slider(
                    label, 0, 10, int(p0.weights.get(k, DEFAULT_WEIGHTS[k])), key=f"{kp}w_{k}"
                )

        save_name = st.text_input(
            "Save these criteria under a client nickname (optional)", key=kp + "save"
        )
        submitted = st.form_submit_button("Build tier list", type="primary")

    prefs = Preferences(
        max_price=max_price or None,
        target_price=target or None,
        min_price=min_price or None,
        min_beds=min_beds or None,
        min_baths=min_baths or None,
        min_sqft=min_sqft or None,
        ideal_sqft=ideal_sqft or None,
        ideal_lot_sqft=ideal_lot or None,
        min_year_built=min_year or None,
        property_types=types,
        statuses=statuses,
        condition_mode=mode,
        weights={k: float(v) for k, v in weights.items()},
    )
    return submitted, (prefs, places, save_name)


# ------------------------------------------------------------------ results


def _show(settings, result: RankingResult, prefs: Preferences) -> None:  # noqa: ANN001
    st.divider()
    ranked, excluded = result.ranked, result.excluded
    if not ranked:
        st.warning(
            f"None of the {len(excluded)} listings meet the must-haves. Loosen the budget or "
            "minimums and rebuild."
            if excluded
            else "There are no listings to rank."
        )
    by_tier = result.by_tier()
    if prefs.condition_mode != "neutral" and not any(r.candidate.condition for r in ranked):
        st.info(
            "Photo-based condition isn't scored yet for these listings, so it isn't part of the "
            "tiers. It will be added automatically once photos are scored."
        )
    if ranked:
        cols = st.columns(len(by_tier))
        for col, tier in zip(cols, [t for t in TIER_BLURBS if t in by_tier], strict=False):
            col.metric(f"Tier {tier}", len(by_tier[tier]))
        photos = services.first_photos(settings, [r.candidate.id for r in ranked])
        saved = services.saved_ids(settings)
        for tier in TIER_BLURBS:
            group = by_tier.get(tier)
            if not group:
                continue
            st.subheader(f"Tier {tier} · {TIER_BLURBS[tier]} ({len(group)})")
            for r in group:
                _card(settings, r, photos.get(r.candidate.id), r.candidate.id in saved)
        st.download_button(
            "Download tier list (CSV)",
            services.ranking_table(result).to_csv(index=False),
            file_name="tier_list.csv",
            mime="text/csv",
        )
    if excluded:
        with st.expander(f"Not considered ({len(excluded)}): didn't meet a must-have"):
            for r in excluded:
                st.markdown(md(f"**{r.candidate.address}**: " + "; ".join(r.excluded_reasons)))


def _card(settings, r: Ranked, photo, saved: bool) -> None:  # noqa: ANN001
    c = r.candidate
    with st.container(border=True):
        left, mid, right = st.columns([1, 3, 4])
        with left:
            st.badge(f"Tier {r.tier}", color=TIER_COLORS[r.tier])
            st.metric("Score", "—" if r.score is None else f"{r.score:.0f}")
            st.caption(f"{r.coverage:.0%} of factors had data")
            if saved:
                st.caption("★ Saved")
            else:
                st.button(
                    "☆ Save",
                    key=f"save_{c.id}",
                    on_click=services.save_home,
                    args=(settings, c.id),
                )
        with mid:
            if photo is not None:
                thumb = thumbnail(settings, photo, 360)
                if thumb:
                    st.image(str(thumb), width="stretch")
            st.markdown(md(f"**{c.address}**"))
            facts = [
                f"${c.price:,}" if c.price else None,
                f"{c.beds} bd" if c.beds is not None else None,
                f"{c.baths:g} ba" if c.baths is not None else None,
                f"{c.sqft:,} sq ft" if c.sqft else None,
            ]
            st.caption(
                md(" · ".join(f for f in facts if f) + (" · demo data" if c.is_synthetic else ""))
            )
        with right:
            for s in r.strengths:
                st.markdown(md(f"✅ {s}"))
            for w in r.watchouts:
                st.markdown(md(f"⚠️ {w}"))
            for u in r.unknowns:
                st.caption(md(f"ℹ️ {u}"))
        with st.expander("Score breakdown"):
            st.dataframe(
                [
                    {
                        "Factor": f.label,
                        "Detail": f.detail or "no data",
                        "Fit": None if f.score is None else round(100 * f.score),
                        "Importance": f.weight,
                    }
                    for f in r.factors
                ],
                hide_index=True,
                width="stretch",
            )
