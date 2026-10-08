# Phase 0 Research: Showing Route Planner + Photo Condition Scorer

Researched 2026-10-08. Every claim carries a status tag:

- **[V]** verified against a primary source I fetched (link given)
- **[S]** secondary source or search snippet only; confirm before relying on it
- **[?]** could not verify; do not build on it without checking

## TL;DR

1. **Data:** Day one is Matrix Custom Export (CSV) plus a local photo folder. FMLS has a real, documented vendor path (FMLS Marketplace -> Bridge API / RESO Web API), and it looks cheap on paper, but whether *you* qualify and what you may store is a broker/FMLS question, not an engineering one (section 1.3).
2. **Vision:** Frozen embeddings + small ordinal head, with a zero-shot CLIP-style baseline. No public dataset matches this task, so your labels are the asset. Plan on ~300-500 labels as the working set.
3. **Labeling:** Streamlit page with keyboard shortcuts; cluster-based cold start, then uncertainty + diversity sampling.
4. **Routing:** OR-Tools single-vehicle VRPTW with disjunction penalties. For travel times, start with **OpenRouteService (free key)** + SQLite cache + haversine fallback. Google is opt-in later for traffic-aware times.
5. **Gap:** ShowingTime's ShowingCart/SmartRoute already orders showings on a map. I found nothing public that does time-window-aware optional-stop dropping with explanations, or uses photo condition as a routing input.
6. **Fair housing:** Score physical condition from pixels only. No location, demographics, or people-related output.

---

## 1. FMLS / Matrix data access

### 1.1 What you can do today, as a member

- FMLS runs on Matrix. [S: [FMLS Matrix page](https://firstmls.com/technology/matrix)]
- Matrix has **Custom Exports**: Settings -> Custom Exports -> Add Export, pick fields from "Available Fields", CSV with comma or tab. Then run a search, select listings, click Export, and choose your template. [S: these are other MLSs' help pages for the same Matrix product: [CRMLS](https://kb.crmls.org/knowledgebase/matrix-how-to-create-a-custom-export), [SmartMLS](https://matrix.smartmls.com/hc/en-us/articles/13118336768283-Create-Custom-Export-Excel)]. FMLS's exact field list and UI may differ, so **the importer does not depend on any one layout** (see D1 and `sources/columns.py`): it auto-detects columns from header synonyms plus what the values look like, and any export with an address will import.
- **Photos: [?]** I found no evidence that Matrix offers a bulk photo download for listings you don't own; the "Download Photos" hits were tour-vendor tools. Day-one design: you place photos in `data/photos/<MLS#>/`. Whether you may save them is a broker question (1.3).
- Matrix cannot export agent information. [S: CRMLS KB above] Not needed for this project.

### 1.2 API / feed path (what FMLS actually offers)

- FMLS distributes data through **Bridge Interactive**, with three access methods: the native Bridge API, a **RESO Web API**, and **vRETS** (a RETS emulator). [V: [FMLS Marketplace info](https://fmls.com/marketplace-info); [V: Zillow Group release on FMLS adopting Bridge](https://zillowgroup.mediaroom.com/2019-10-16-Bridge-API-Deployed-by-First-Multiple-Listing-Service)]
- Third parties are approved through the **FMLS Marketplace**: a vendor creates a "Product" per service, and once approved it shows as "Approved" to roughly 60,000 FMLS members. [V: same page]
- **Fees:** $250 application fee, **waived for member brokerages**. Monthly fee is based on the number of active FMLS members subscribed. Typical review turnaround is 2 business days. [V: same page]
- **Fee schedule** (effective Aug 1, 2019, so it may be outdated): 0-2 subscriptions $0, 3-10 $22 each, 11-25 $18 each, and so on down to $0.10 each at 10,001+. A one-user personal tool would sit in the $0 tier *if* it qualifies as a Product. [V: [FMLS pricing](https://www.fmls.com/marketplace-pricing/); confirm current numbers]
- **Contact:** Data@FMLS.com, 678-904-0446. [V: FMLS Marketplace info]
- Bridge's own sign-up docs ([Signing up with Bridge API](https://bridgedataoutput.com/docs/platform/Introduction/Signing-up-with-Bridge-API)) did not render for me. **[?]** Read them yourself; I won't guess endpoints, dataset names, or token types.
- General pattern across MLSs: RESO is a standards body, not a data source, and a data license is granted for a scope (what, in what context, for which audience). [S: [RESO FAQ](https://www.reso.org/faq/), [Repliers](https://help.repliers.com/en/article/understanding-mls-data-licensing-for-using-our-apis-12fknh6/)]
- The RESO field that maps to showing logistics is **`ShowingRequirements`** (e.g. Appointment Required, Courtesy Call Only, Go Direct). [V: [RESO DD wiki](https://ddwiki.reso.org/display/DDW17/ShowingRequirements+Field)] The CSV importer should map it if your export has it.

### 1.3 What to ask your broker (in order)

1. May I store MLS listing data and photos locally on my own computer for my own business use? For how long, and what must I delete when a listing goes off market?
2. May I run **automated analysis** (ML scoring) on listing photos? Does the broker agreement or FMLS rules treat that as a permitted use of the content?
3. Is my brokerage willing to register a Product on the **FMLS Marketplace** (application fee is waived for member brokerages), or does FMLS allow an individual agent to register as a vendor?
4. If yes: which data scope applies (IDX-style display, broker back-office, other), and does it include photo URLs and showing instructions?
5. Who signs the data agreement, and who holds the credentials (broker vs me)?
6. Are derived scores (my condition scores) something I can share with clients, or internal only?
7. Does Dream Realty have rules on third-party AI tools touching MLS content?

Until you have written answers: CSV + local photos only, nothing leaves your machine, no redistribution.

---

## 2. Condition-scoring approaches

### 2.1 Options

| Approach | Labels needed | Strength | Weakness |
|---|---|---|---|
| (a) Zero-shot CLIP/SigLIP prompts | 0 | Free baseline; good for **room type**, junk detection (floor plan, map, logo) | Weak on subtle condition and renovation level; scores are poorly calibrated |
| (b) Fine-tune CNN/ViT with ordinal head | Many (likely 1,500+) | Best ceiling with enough data | Overfits at 300-500 labels; GPU time; more to maintain |
| (c) **Frozen embeddings + small ordinal head** | ~100 to start, 300-500 for a usable model | Cheap, CPU-friendly, embeddings cached, retrain in seconds, easy active learning | Ceiling limited by the frozen features |

### 2.2 Ordinal head

- **CORAL** (rank-consistent, shared weights) and **CORN** (conditional-probability chain, rank-consistent without weight sharing) both ship in the `coral-pytorch` package. [V: [CORN paper, arXiv 2111.08851](https://arxiv.org/pdf/2111.08851); [coral-pytorch on PyPI](https://pypi.org/project/coral-pytorch)]
- Plan: compare **ridge regression + learned thresholds** against a **CORN MLP head** with cross-validation. With a few hundred labels the simple model is often competitive.

### 2.3 Backbone choice

- Fine-grained semantics tend to favor language-supervised features (CLIP/SigLIP); pure visual quality favors DINOv2. [S: [Data or Language Supervision: What Makes CLIP Better than DINO?](https://arxiv.org/html/2510.11835v1); [DINOv2 paper](https://arxiv.org/pdf/2304.07193)] None of this is specific to property condition.
- **Decision:** an `Embedder` interface with two backends (OpenCLIP/SigLIP ViT-B, DINOv2 ViT-B). Embeddings are cached, so comparing them on your first labels costs minutes. Pick by cross-validated QWK.
- Hardware: your RTX 3050 6 GB handles ViT-B inference and head training easily, and CPU fallback is fine for batch scoring. I'll benchmark rather than guess throughput.

### 2.4 Public datasets and papers

I did not find a public dataset of listing photos with 1-5 condition or kitchen/bath renovation labels. **[?: not an exhaustive search.]**

| Resource | What it is | Use for us |
|---|---|---|
| [Visual Estimation of Building Condition with Patch-level ConvNets (arXiv 1804.10113)](https://arxiv.org/abs/1804.10113) | 10,636 exterior images of 6,273 Austrian houses, appraiser condition c1-c9 collapsed to 3 classes; 65.4% accuracy; **images proprietary**, only features/networks promised [V] | Evidence the task is learnable but hard. Noisy labels are normal. Not usable as training data |
| [Vision-based Real Estate Price Estimation (arXiv 1707.05489)](https://arxiv.org/abs/1707.05489) | 7 room types; DenseNet room classifier ~91% [V]; "luxury level" labeled by **triplet comparisons** on MTurk, then clustered [V]; a 9k-house Zillow dataset is mentioned but public release is unconfirmed [?] | Precedent for room-type-first pipelines and **comparison-based labeling** (possible later labeling mode) |
| [Places365](http://places2.csail.mit.edu/) and Houzz-derived sets | Scene/room categories | Sanity-check the room-type classifier only; I did not verify licenses, so check before any use |
| Fannie Mae [Selling Guide B4-1.3-06](https://selling-guide.fanniemae.com/sel/b4-1.3-06/property-condition-and-quality-construction-improvements) | C1-C6 condition, Q1-Q6 quality definitions [V] | Anchor language for the rubric (section 3) |

### 2.5 Pipeline (per photo)

1. Load -> EXIF-orient -> resize -> content hash (sha256) -> perceptual hash for near-duplicate grouping.
2. **Junk filter** (zero-shot + threshold): floor plan, map, logo/graphic, aerial, document. Excluded photos are kept but marked `excluded` with a reason.
3. **Room type** (zero-shot first; swap in the trained head once labels exist): kitchen, bath, living, bedroom, exterior, other.
4. **Condition** 1-5 with confidence (ordinal head; confidence from ordinal probabilities plus bootstrap/ensemble disagreement).
5. **Flags:** water stain, damaged flooring, dated finishes, clutter, poor photo quality. Zero-shot only until labeled; show as "possible", not fact.
6. **Rollup per listing:** room-weighted mean (kitchen/baths highest, weights in config), worst-room indicator, coverage (which room types were actually seen).

### 2.6 Fair housing guardrails (design rules, not afterthoughts)

- Input is **pixels only**. The scorer never receives address, ZIP, neighborhood, price, listing remarks, or any location feature.
- No outputs about people, families, religion, or "who would like this home". Photos of personal items count only as a `clutter` flag, never a description.
- Output language is about surfaces and components ("water stain on ceiling"), not people or places.
- UI disclaimer on every scoring and planning page: scores are an aid for the agent, not an appraisal or inspection.
- "Client mode" (prefer move-in ready vs prefer fixer-upper) is a **preference the client stated**, chosen by the agent per client, and logged. It is never derived from client attributes.
- HUD's 2024 guidance on AI and advertising stresses that liability does not depend on intent. [S: [Planetizen summary](https://planetizen.com/node/128865); read HUD's primary documents before any client-facing use]

---

## 3. Labeling strategy

### 3.1 Rubric (1-5, judged on **visible surfaces**, not furniture or staging)

The scale blends wear/maintenance with update level, following the spirit of appraisal condition ratings (C1 no depreciation, C2 no deferred maintenance, C3 well maintained, C4 minor deferred maintenance, C5 obvious deferred maintenance, C6 substantial damage). [V: Fannie Mae link above] Appraisals rate the whole dwelling, so this is an adaptation, not an equivalence. The `dated_finishes` flag keeps *update level* separable from *wear*, so the two can become separate heads later.

| Score | General meaning |
|---|---|
| 5 | Like new or recently renovated; no visible wear; current finishes |
| 4 | Well maintained; clean; maybe lightly dated; no defects visible |
| 3 | Functional and average; normal wear; dated but intact; minor cosmetic issues |
| 2 | Visible deferred maintenance; damaged or worn surfaces; needs real work |
| 1 | Major problems visible: water damage, severe damage, gutted or unsafe-looking |

Room anchors for what to look at:

| Room | Look at |
|---|---|
| Kitchen | Cabinets (doors, finish), countertops, backsplash, appliances (age, matching), flooring, sink/faucet |
| Bath | Tile and grout, vanity, tub/shower surround, caulk, fixtures, mold/staining |
| Living | Flooring, paint, ceiling, windows, fireplace, lighting |
| Bedroom | Flooring or carpet, walls and ceiling, windows, closets |
| Exterior | Roof (visible), siding/paint/brick, gutters, windows, porch/deck, driveway, lawn and landscaping |
| Other | Garage, basement, laundry, dining, hall: use the surfaces that are visible |

Flags (multi-select, independent of the score): `water_stain`, `damaged_flooring`, `dated_finishes`, `clutter`, `poor_photo_quality`, plus `unsure` to skip bad examples without polluting labels. Also a **room-type correction** key.

### 3.2 Labeling UI (Streamlit)

- One photo at a time, large; keys **1-5** score, letter keys toggle flags, **S** skip, **U** undo, **Space/Enter** save and advance.
- Streamlit has no native key bindings. Options: a third-party package such as `streamlit-shortcuts`, or a ~20-line JS component. **[?]** Verify which works on the current Streamlit version in Phase 2; button fallback exists either way.
- Show the model's current guess *after* you label (avoid anchoring), with a toggle to hide it entirely.
- Store: label, flags, room type, timestamp, labeling-time ms, rubric version.

### 3.3 Active learning schedule

1. **Cold start (first ~100):** k-means on embeddings, label the photo nearest each centroid, stratified by zero-shot room type so each room type gets coverage.
2. **Then rounds of ~50:** train the head, pick photos by **uncertainty + diversity** (high ordinal entropy or ensemble disagreement, then k-center/cluster selection to avoid near-duplicates). Hybrid uncertainty-plus-diversity is the standard recommendation. [S: [Revisiting Active Learning in the Era of Vision Foundation Models](https://arxiv.org/pdf/2401.14555), [Hybrid Active Learning with Uncertainty-Weighted Embeddings](https://mlanthology.org/tmlr/2024/he2024tmlr-hybrid/)]
3. Keep a **random holdout** (~20%, chosen before any active selection) so evaluation is not biased by what the model asked about.
4. **Relabel ~10% after a week** to measure your own test-retest agreement (QWK). That is the ceiling for any model.
5. Rare classes (1 and 5) will be scarce; add a "find me candidates" query using zero-shot flags and nearest neighbors.

### 3.4 Metrics and fine-tune trigger

- Report MAE, **quadratic weighted kappa**, and per-room-type confusion matrices, on the holdout only.
- **Fine-tune trigger (my heuristic, not from literature):** consider partial fine-tuning (last blocks or LoRA) only when labels exceed ~1,500 *and* the learning curve is still falling at the current size. If the curve is flat, the bottleneck is rubric noise, not model capacity.

---

## 4. Routing stack

### 4.1 OR-Tools VRPTW best practices

From the [OR-Tools VRPTW guide](https://developers.google.com/optimization/routing/vrptw) [V] and standard practice:

- Time dimension via `AddDimension(transit_cb, slack, horizon, fix_start_cumul_to_zero=False, "Time")`. **Slack must be > 0** to allow waiting. Per-stop windows via `CumulVar(index).SetRange(open, close)`. [V]
- **Service time goes inside the transit callback** (travel i->j plus service at i). [V]
- Vehicle start/end windows set on `routing.Start(v)` / `routing.End(v)` cumul vars, which encodes day start/end and a flexible departure time. [V] `AddVariableMinimizedByFinalizer` tightens the final schedule. [V]
- Use **integers**: I'll use whole minutes (rounded up). Showing windows are minute-granular anyway.
- **Optional visits:** `routing.AddDisjunction([index], penalty)` per showing. [S: search results quoting the OR-Tools docs and [a disjunction write-up](https://activimetrics.com/blog/ortools/exploring_disjunctions/); not on the VRPTW page I fetched] The penalty must exceed the largest detour cost in the same units, otherwise the solver drops stops to save a few minutes of driving. Set penalty = large base x priority.
- Search: `PATH_CHEAPEST_ARC` first solution + `GUIDED_LOCAL_SEARCH` with a short time limit (a few seconds is plenty for ~15 stops). [from practice; tune in Phase 3]
- **Lunch:** model as a pseudo-stop with zero travel time to everywhere and a window (e.g. 11:30-13:30). Simple, visible in the itinerary, and avoids OR-Tools break-interval subtleties. [from practice; native `SetBreakIntervalsOfVehicle` exists but I haven't verified its behavior]
- **"Which stop was dropped and why":** after solving, for each dropped stop, diagnose (a) window already closed given earliest feasible arrival, (b) no slot between neighbors given service + buffer, (c) day too short, (d) only fits by displacing a higher-priority stop. Implemented by trial insertion into the final route and reporting the first violated constraint.
- **Traffic:** OR-Tools has no time-dependent travel. Plan: solve with free-flow times, compute arrival times, re-price legs with period multipliers (off-peak / AM peak / PM peak), re-solve, repeat 2-3 passes.

### 4.2 Travel-time sources

| Source | Cost / limits | Traffic | Cache-friendly | Verdict |
|---|---|---|---|---|
| **OSRM public demo** | Free; **1 request/sec**; "restricted to reasonable, non-commercial use"; no uptime guarantee; car/foot/bike profiles [V: [OSRM demo server wiki](https://github.com/Project-OSRM/osrm-backend/wiki/Demo-server)] | No | Yes (ODbL data) | **Dev only.** Your use is commercial |
| **OSRM self-hosted** (Georgia extract from [Geofabrik](https://download.geofabrik.de/north-america/us/georgia.html)) | Free, unlimited | No | Yes | Best long-term, but **Docker is not installed** on your machine [V: checked] |
| **OpenRouteService** (free key) | Standard plan reported at Matrix **500 req/day, 40/min** [S: [ORS plans](https://openrouteservice.org/plans/), [apis.io summary](https://apis.io/plans/openrouteservice/openrouteservice-plans-pricing/)]; max locations per request [?] | No | Likely (OSM-based); confirm terms | **MVP pick.** No install; one matrix call per plan |
| **Google Routes API: Compute Route Matrix** | Billed **per element**. Essentials: 10,000 free/month then $5 per 1,000; Pro: 5,000 free then $10 per 1,000 [V: [Google pricing](https://developers.google.com/maps/billing-and-pricing/pricing)]. A 12-stop day plus depot is 13x13 = 169 elements, so ~59 plans/month fit in the Essentials free cap | Yes, but which SKU includes traffic-aware times is [?] | **Caching Google content is restricted by their terms [?]; verify before storing** | Opt-in later if traffic accuracy matters; needs a billing account |
| **Haversine fallback** | Free | No | n/a | Offline dev and tests. Straight-line x circuity factor (start ~1.35) / assumed speed (start ~28 mph). Both are guesses; calibrate against cached provider results |

**Decision:** a `TravelTimeProvider` interface with Haversine, OSRM, ORS, and Google implementations. MVP default is **ORS with a free key**, falling back to Haversine. The SQLite `travel_time_cache` is keyed by (provider, origin hash, dest hash, time-profile) with a per-provider TTL, so Google rows can be disabled or short-lived without schema changes.

### 4.3 Geocoding

- **US Census Geocoder:** free, batch up to 10,000 addresses, built on MAF/TIGER. [V: [Census Geocoder docs](https://www.census.gov/programs-surveys/geography/technical-documentation/complete-technical-documentation/census-geocoder.html)] Newly built subdivisions may fail to match (common in metro Atlanta suburbs) **[?]**.
- **Nominatim (public):** max 1 request/sec, must cache, identify your app in the User-Agent, bulk geocoding discouraged, no use as a primary service. [V: [OSMF policy](https://operations.osmfoundation.org/policies/nominatim/)] Fine as a *fallback* for single-address misses.
- **Decision:** CSV lat/long if present -> Census (batch) -> Nominatim fallback (throttled, cached) -> manual pin override in the UI. All results land in `geocode_cache`.

---

## 5. Showing logistics in Georgia (what to model)

Practical constraints, with sources where I have them:

- **Showing instructions bind you.** The principle (listing agents must report accurate instructions; showing agents must follow them even if the home is vacant or has a lockbox) comes from a New Mexico association's page, [GAAR](https://www.gaar.com/blog/article/gaar-lockbox-and-swmls-showing-requirements) **[S, not Georgia]**. I did not read a Georgia or FMLS rule on this; the Georgia REALTORS [best-practices PDF](https://garealtor.com/wp-content/uploads/2020/05/GAR-Best-Practices-for-Conducting-Real-Estate-Spring-Summer-2020.pdf) surfaced in search but I only saw its title **[?]**. Confirm FMLS rules. Model `access_type` from `ShowingRequirements`: lockbox/go-direct, call first, appointment with occupant, appointment with listing agent.
- **Appointment status matters.** In ShowingTime, a request is not a showing until the seller side confirms; buffers (15-minute steps up to 90 minutes) and "secure access" lockbox-visibility windows (0-60 minutes) are configurable per listing. [S: SmartMLS (Connecticut) docs for the same ShowingTime product: [buffer](https://showings.smartmls.com/hc/en-us/articles/13120094261915-ShowingTime-buffer-setting), [Secure Access](https://showings.smartmls.com/hc/en-us/articles/13120092568859-ShowingTime-s-Secure-Access-setting)] Defaults are set by each MLS, so treat as a model of the *shape*, not Georgia's values.
- **Model per showing:** earliest start, latest start (hard window), duration (default 30), buffer before/after (default 5-10 for parking, lockbox, shoes), status (requested/confirmed/denied), access type, locked-in flag for re-optimization.
- **Day-level:** start location, end location, day start/end, lunch window and length, client pickup and drop-off as extra fixed nodes.
- **Metro Atlanta traffic:** weekday morning and evening peaks matter; the rush-hour sources I found were SEO-grade **[S, low quality]**, so I'll model peak periods as configurable multipliers and calibrate against your own experience or Google's typical-traffic view, not these pages.
- **Priority** (Phase 3): configurable mix of client preference rank, condition score by client mode (move-in ready vs fixer-upper), and window tightness (tight windows get higher penalty so they aren't the ones dropped).

---

## 6. Landscape and gap

- **ShowingTime** is integrated into Matrix. **ShowingCart** groups multiple showings with a map, driving directions and drive times; **SmartRoute** re-maps when you add a stop. [S: [SmartMLS ShowingCart article](https://showings.smartmls.com/hc/en-us/articles/13119924170267-Use-the-ShowingCart-to-schedule-showings-for-multiple-listings), [OneKey MLS](https://corporate.onekeymls.com/products/showingtime)]
- **General route planners** pitched at agents: [Upper](https://www.upperinc.com/blog/real-estate-route-planner/), [RoadWarrior](https://roadwarrior.app/businesses/real-estate-agents/), [Map It](https://apps.apple.com/us/app/map-it-client/id1621458700), [ShowingSmart](https://www.showingsmart.com). Marketing pages describe sequencing by distance/traffic and appointment windows.
- **Gap this fills** (based only on public help and marketing pages; **[?]** I can't see their algorithms):
  1. Explicit **optional-stop optimization** that tells you *which* showing to drop and *why*.
  2. **Photo-derived condition** as a routing and prioritization input, with a per-room explanation.
  3. **Local, agent-owned** data and scores, with a labeling loop tuned to your own eye.
- Be honest in positioning: ShowingTime owns the *scheduling/confirmation* workflow. This tool plans around confirmed windows; it does not replace them and sends nothing to clients.

---

## 7. Unverified / open items

- FMLS-specific Matrix export fields and whether photos can be bulk-downloaded **[?]**
- Whether an individual agent can be an FMLS Marketplace vendor, and what scope covers photo analysis **[?]**
- Bridge API sign-up details and auth model **[?]**
- Current FMLS fee schedule (published one is from 2019) **[V but dated]**
- ORS official Matrix limits (max locations, caching terms) **[S]**
- Which Google SKU gives traffic-aware matrix times, and Google caching terms **[?]**
- Census Geocoder match rate on new construction **[?]**
- Atlanta peak-period multipliers: calibrate empirically **[S, low quality]**
- Streamlit keyboard-shortcut approach on the current version **[?]**
- Real-world accuracy of any approach here for *listing photos*: nothing published I could verify; the only comparable result is 65% on 3-class exterior condition [V]

---

## 8. Decisions

| # | Decision | Why |
|---|---|---|
| D1 | `ListingSource` interface; ship `CsvImportSource` with **layout auto-detection** (no fixed template; synonyms in `config/column_aliases.yaml`, value-pattern checks, confirm/override in the UI, remembered layouts) and `ManualEntrySource`; `ResoWebApiSource` is a documented stub. *Revised after review: the tool must work as a universal product, not only for one MLS template.* | No scraping; works with any export that has an address, so it needs no sample file and is usable by any agent |
| D2 | Vision MVP = zero-shot baseline + **frozen embeddings + small ordinal head** (ridge+thresholds vs CORN, chosen by CV) | No labeled data, limited GPU; embeddings cache; retrains in seconds |
| D3 | Embedder interface with OpenCLIP/SigLIP and DINOv2 backends; bake-off on first ~150 labels | Literature doesn't settle it for this task; cheap to test |
| D4 | Fine-tuning deferred until >~1,500 labels and a still-falling learning curve | Overfit risk at 300-500 labels |
| D5 | Labeling: Streamlit keyboard UI, cluster cold start, uncertainty+diversity rounds, random 20% holdout, 10% relabel | Fast for a single domain expert; honest evaluation |
| D6 | Rubric 1-5 with room anchors; `dated_finishes` flag keeps update level separable from wear | Matches your flags; allows a two-head model later |
| D7 | Fair housing: pixels-only input, no people/location outputs, UI disclaimer, client mode logged as stated preference | Required by your constraints; reduces legal risk |
| D8 | Routing: OR-Tools single vehicle, integer minutes, disjunction penalties scaled by priority, lunch as pseudo-stop, dropped-stop diagnostics, multi-pass traffic repricing | Meets the spec with the fewest moving parts |
| D9 | Travel times: `TravelTimeProvider` with Haversine, OSRM, ORS, Google; **default ORS free key** -> Haversine fallback; cache keyed by provider with TTL | No Docker installed; OSRM demo is non-commercial; Google caching terms unverified |
| D10 | Geocoding: CSV lat/long -> Census batch -> Nominatim fallback -> manual pin; all cached | Free, policy-compliant |
| D11 | Storage: SQLite with SQLAlchemy 2.0 + Alembic; embeddings on disk under `data/embeddings/<model_id>/`, keyed by photo sha256 | Boring and standard; keeps the DB small |
| D12 | Tooling: Python 3.12 venv + pip (`uv` not installed), `pyproject.toml` + pinned `requirements.txt`, ruff, pytest. Torch installed separately (CUDA wheel for your RTX 3050, CPU wheel otherwise) | Matches your machine; no new tool to learn |
| D13 | Map: decide in Phase 3 between pydeck (ships with Streamlit) and folium; export `.ics` and PDF/text; nothing auto-sent | Don't commit before testing basemap behavior |

---

## 9. Proposed architecture (for your approval)

```
                  Matrix CSV export + local photo folder
                                   |
        sources/  ListingSource -> CsvImportSource | ManualEntrySource | ResoWebApiSource (stub)
                                   |
   core/   SQLite (data/app.db): listings, photos, photo_scores, listing_scores, labels,
           agents, clients, showings, routes, route_stops, travel_time_cache, geocode_cache
           config (.env + column aliases + saved layouts + room weights), geocoding w/ cache, logging
                    |                                          |
   vision/  preprocess (EXIF, resize, pHash dedupe, junk filter)      routing/  travel providers + cache
            -> Embedder (OpenCLIP | DINOv2, disk cache)                         priority.py  (client rank,
            -> room type -> condition head -> flags                              condition, window tightness)
            -> rollup (weights, worst-room)                                      solver.py  (OR-Tools VRPTW)
            train.py / eval.py (MAE, QWK, confusion)                             explain.py (dropped stops)
            models/<version>/ + model card                                       export: .ics, PDF/text, map
                    \                                          /
                     app/  Streamlit: Listings | Photo Review | Labeling | Plan a Day | Settings
```

Build order (each phase ends with tests, an app run, and a 5-line status): **Phase 1** schema + CSV import + geocoding + page skeletons -> **Phase 2** vision MVP + labeling UI + evaluation -> **Phase 3** routing -> **Phase 4** polish, explainability, demo mode with clearly labeled synthetic data.
