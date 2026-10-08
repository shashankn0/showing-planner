# Showing Planner

A web app for real estate agents. Tell it what a client wants and it ranks your listings into a
**tier list (S to D)**, showing the reasons behind every placement:

1. **Find Homes**: enter budget, beds/baths, size, places the client drives to, and how they feel
   about condition; get a tier list with strengths, watch-outs and a score breakdown.
2. **Photo condition scoring** *(next phase)*: scores listing photos for physical condition and
   feeds the tiers.
3. **Showing route planning** *(later phase)*: sequence a day of showings with time windows.

It is **not tied to any MLS**: import any spreadsheet or export that has addresses, and photos
from any folder or zip. It never scrapes listing sites, and tiers never use neighborhood
demographics. See [docs/PRODUCT.md](docs/PRODUCT.md) for exactly what goes into a tier and why.

> **Status:** listings, photos, geocoding, the tier-list engine, client profiles and hosting
> protections are built and tested. Photo scoring and routing are not built yet.
>
> **Put it online free** (no domain purchase): [docs/DEPLOY.md](docs/DEPLOY.md).
>
> **Try it instantly:** on Find Homes click *Try it with demo listings* (synthetic data).
> **Search real listings:** Listings > Search (RentCast) with your own key; no scraping.

## Set up (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env        # optional: add keys later
streamlit run app/main.py
```

Developers: `pip install -r requirements-dev.txt`, then `pytest` and `ruff check .`.

## Using it

0. **Find Homes** (the landing page). Once listings are loaded, fill in what the client wants and
   press *Build tier list*. Save the criteria under a client nickname to reuse them. Listings that
   miss a must-have (over budget, too few beds...) are listed separately with the reason.
1. **Listings -> Import a file.** Drop in any CSV, TSV, TXT or Excel (.xlsx) file. The app works
   out which column is which from the headers *and* the values (zip codes, prices, states...).
   Check the preview, fix anything it got wrong, and import. Tick "Remember this layout" and the
   next export of the same shape imports with no clicks. Only an address is required; if your file
   has no city/state, the defaults in Settings are used.
2. **Listings -> Add by address.** Paste addresses, one per line.
3. **Listings -> Photos.** Point to a folder or zip. Photos are matched to listings by MLS number
   or address in the folder/file names; anything unmatched you assign by hand. Folders named like
   addresses can create listings (photos-only mode).
4. **Photo Review** lets you look through a listing's photos and exclude floor plans, maps, logos.
5. **Settings** holds defaults, key status and saved layouts.

Old Excel `.xls` files aren't supported: in Excel, use *Save As -> CSV* or *.xlsx*.

If an export uses unusual column names, add them to
[config/column_aliases.yaml](config/column_aliases.yaml); no code changes needed.

## Data handling: read this

- **Listing data and photos stay on this computer** in the `data/` folder, which is gitignored.
  Never commit listing data, photos, or keys.
- **Confirm with your broker what you may store and for how long**, and whether automated analysis
  of listing photos is allowed. MLS data and photos are governed by MLS rules and your broker
  agreement. See [docs/RESO_WEB_API.md](docs/RESO_WEB_API.md) for the questions to ask.
- This tool **never scrapes** MLS or listing sites. It does not download photos from URLs in an
  export.
- Map lookups send **street addresses to the free US Census geocoder** (and, only if you enable it
  in `.env`, to OpenStreetMap's Nominatim). Untick "Look up map coordinates" to stay offline.
- Don't import alarm or lockbox codes. The importer masks digits following words like "code" or
  "alarm" in showing instructions as a safety net; it is not a guarantee.
- Keep client records to nicknames.

## Fair housing

Condition scores describe **visible physical condition in photos only**. The scorer will never be
given addresses, neighborhoods, prices, listing remarks, or any demographic or location signal,
and will not produce language about people. Scores are an aid for the agent, **not an appraisal or
inspection**.

## Layout

```
app/       Streamlit pages (views/) and UI-independent actions (services.py)
core/      settings, SQLite models + Alembic migrations, store, geocoding
sources/   ListingSource implementations, column detection, address/photo parsing
vision/    (next phase) embeddings, classifiers, scoring, training
routing/   (later phase) travel times, OR-Tools model, exporters
tests/     pytest suite; fixtures are clearly labeled synthetic data
docs/      RESEARCH.md (decisions and sources), RESO_WEB_API.md
data/      local only, gitignored
models/    model cards and versions
```

## Tests

`pytest` runs the whole suite offline. It covers the importer against several export styles,
photo matching, geocoding (with a fake HTTP layer), and renders every Streamlit page.
