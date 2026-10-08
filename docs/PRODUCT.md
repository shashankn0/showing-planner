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
| RentCast API | Option | Free plan = 50 calls/month; paid from $74/mo. Sale listings, prices, days on market. **Its documented listing fields include no photos**, so it can't feed photo analysis by itself |
| FMLS / Bridge licensed feed | Needs broker | Includes photos; requires a data license (see `RESO_WEB_API.md`) |

## Decisions that are yours

1. **Who uses the website?** Only you (private link, your real data stays on your PC), or a public
   site anyone can try with their own uploads (no MLS data stored on the server)?
2. **Listing source:** keep to file uploads for now, add RentCast for search by city/ZIP/criteria
   (and bring your own photos), or wait for a broker-approved FMLS feed?
3. **Schools:** leave them out of tiers (current), or build the opt-in, source-cited filter after
   you've checked with your broker?

## Roadmap

1. **Done:** import any listing file, photos, geocoding, preference-driven tier list, client
   profiles, hosting protections.
2. **Next:** photo condition scoring (zero-shot baseline, then your labels train a small ordinal
   model) feeding the "condition" criterion.
3. Then: real drive times (OpenRouteService) and flood zones; optional listing API; route planner
   for a day of showings.
