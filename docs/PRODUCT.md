# What this product is (and isn't)

**Showing Planner** helps an agent turn a client's wish list into a ranked **tier list** of
listings (S, A, B, C, D), with the reasons visible on every card. Later phases add photo-based
condition scoring and a day-of-showings route planner.

```
  Load listings            Tell it what the client wants          Tier list with reasons
  (file / paste /    -->   budget, beds & baths, size, places  --> S  best matches
   photos folder)          they drive to, condition attitude,      A  strong
                           how much each factor matters            B  good ... D  weak
```

## What goes into a tier

Every criterion is a fact about the **property** or a **need the client stated**. The complete
list is enforced in code (`ranking/preferences.py`); anything else is rejected.

| Criterion | Based on | Status |
|---|---|---|
| Price vs. budget | List price, the client's max and ideal price | Built |
| Bedrooms / baths | Listing facts vs. client minimums | Built |
| Living space | Sq ft vs. client minimum and ideal | Built |
| Value | $/sq ft vs. the other homes being compared (needs 5+ similar homes) | Built |
| Commute | Distance to places the **client names**, as a straight-line estimate | Built (real drive times in the routing phase) |
| Condition from photos | Photo scoring, with the client's attitude (move-in ready / project OK / don't care) | **Next phase.** Not scored yet, so it's skipped, not guessed |
| Year built, lot size, time on market | Listing facts | Built (time on market is opt-in) |
| Flood zone | FEMA flood map lookup (free, keyless) | Planned |

A tier needs at least half of the chosen criteria to have data; otherwise the card says "not
enough data" instead of showing a falsely precise score.

## What it deliberately does not do

- **No neighborhood demographics, and no "neighborhood quality" score.** Ranking homes by who
  lives nearby is steering under the Fair Housing Act, and a licensed agent carries that risk.
  Income and similar "economic" neighborhood statistics are strong stand-ins for protected
  classes, so they're excluded too. "Economic" factors in the tier mean the *property's*
  economics: price vs. budget, price per sq ft, time on market.
- **Schools and crime.** HUD's April 24, 2026 letter says sharing objective, verifiable school
  and crime information isn't by itself a Fair Housing violation, provided it's neutral,
  consistent across clients, and free of discriminatory intent; steering remains prohibited
  ([NAR summary](https://www.nar.realtor/fair-housing/faqs-on-steering-crime-and-schools);
  I could not read the HUD letter itself). NAR's guidance is to give every client the same
  verified data from official sources and avoid "good/bad neighborhood" characterizations.
  So schools are **not in the score**. If you want them, the proposal is an opt-in, client-stated
  filter ("only homes zoned to schools meeting X by the state's published ratings") that shows the
  source and treats every client the same. That's your call and your broker's. See "Decisions".
- **No scraping** of Zillow, Realtor.com, Redfin, Matrix or any listing site. Their terms forbid
  it, and your own project rules do too.
- **No appraisal or inspection claims.** Every scoring page carries the disclaimer.

## Where listings come from ("scouring")

| Source | Status | Notes |
|---|---|---|
| Your own export or any spreadsheet | Built | Auto-detected layout; the free, fully compliant default |
| Paste addresses / photo folders | Built | |
| RentCast API | **Built** (Listings > Search) | Bring your own key. Free plan = 50 calls/month; paid from $74/mo. Sale listings, prices, days on market. **Its documented listing fields include no photos**, so those listings have none unless you add photos. Each search uses one call and repeats within 6 hours reuse the saved result |
| FMLS / Bridge licensed feed | Needs broker | Includes photos; requires a data license (see `RESO_WEB_API.md`) |

## Decisions made (2026-10-08)

1. **Audience: a public demo anyone can try.** Hosted mode gives every visitor an isolated
   temporary workspace; a "Try it with demo listings" button loads clearly-labeled synthetic
   listings so nobody has to upload anything first. No MLS data is stored on the server.
2. **Listing source: RentCast search**, with each user bringing their own key. RentCast's API terms
   say a key may not be shared with others for their own use and the key holder is liable for
   overage charges, so the public site never uses the owner's key: it is typed by the visitor,
   held only for their session, sent only to RentCast, and never saved (tests check the key is
   not on disk). Storing and displaying the data is permitted by their terms.
3. **Schools: opt-in filter requested, but not built yet** (see below).

### Why the school filter isn't built

A filter like "only homes zoned to a school the state rates at X or better" needs two things:
which school an address is zoned to, and an official rating. I looked for a verified, current,
free source and didn't find one:

- **Zoning:** the only nationwide attendance-boundary dataset (NCES SABS) covers just the
  2013-14 and 2015-16 school years and was an experimental survey that has ended. Boundaries
  change, so it would give wrong answers. [S: [NCES EDGE SABS](https://nces.ed.gov/programs/edge/SABS)]
- **Ratings:** I found no confirmed bulk download or API for Georgia's CCRPI scores (GOSA publishes
  school reports). [S: search results; unverified]
- **Commercial ratings are a known problem:** analyses of GreatSchools' 1-10 ratings found they
  track school demographics (schools with the most low-income and Black and Hispanic students
  averaged 4 to 6 points lower), and that they nudge families toward whiter, more affluent
  schools. Zillow removed the color-coding for that reason. [S:
  [Chalkbeat](https://www.chalkbeat.org/posts/us/2019/12/05/greatschools-ratings-poverty-race-segregation-housing/),
  [Inman](https://www.inman.com/2016/12/20/zillow-wipes-colors-school-ratings-media-coverage/)]
  I would not use them in a tier.

Building it responsibly means: (a) picking the school districts you actually work in, using each
district's own published zone map or GIS layer, (b) taking ratings only from the state's official
report card, with the source shown, and (c) your broker's sign-off. Tell me which districts you
cover and I can scope it. Until then no school data enters any score.

## Roadmap

1. **Done:** import any listing file, photos, geocoding, preference-driven tier list, client
   profiles, hosting protections, demo listings, RentCast search (bring your own key).
2. **Next:** photo condition scoring (zero-shot baseline, then your labels train a small ordinal
   model) feeding the "condition" criterion.
3. Then: real drive times (OpenRouteService) and flood zones; optional listing API; route planner
   for a day of showings.
