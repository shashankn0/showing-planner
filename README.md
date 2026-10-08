# Showing Planner

A local-first tool for real estate agents:

1. **Photo condition scoring** *(next phase)*: score listing photos for physical condition per
   photo, per room type, and per listing.
2. **Showing route planning** *(later phase)*: sequence a day of showings with time windows,
   service times and optional stops.

It is **not tied to any MLS**. You can import any spreadsheet or export that has addresses
(an MLS export, a portal download, a list you made yourself), and photos from any folder or zip.

> **Status:** Phase 1 (listings, photos, geocoding, app skeleton) is done. Scoring and routing
> are not built yet; their pages are placeholders.

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
