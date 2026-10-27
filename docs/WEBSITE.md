# From local tool to website: the pathway

Checked 2026-10-27. Tags: **[V]** verified (official page, or I ran it), **[S]** secondary source,
**[?]** not verified. Companion to [DEPLOY.md](DEPLOY.md) (getting it online) and
[PRODUCT.md](PRODUCT.md) (what goes into a tier).

## Where it stands

| Stage | What | Status |
|---|---|---|
| 1 | Themed Streamlit site, **Saved Homes** pipeline, Google viewing, optional Google sign-in | **Built.** Tested in-process and in a real browser, except the live Google 3D map and a real sign-in (below) |
| 2 | Accounts that keep saved homes on a public site | Needs a database that outlives the host. Steps below |
| 3 | A separate web frontend | Only if Streamlit's layout becomes the limit. Triggers below |

## Stage 1: what you have

- **Theme:** teal on warm neutrals, separate light and dark palettes, in `.streamlit/config.toml`.
- **Saved Homes** (sidebar): press **☆ Save** on any tier-list card. Each saved home has a status
  (interested, toured, offer, passed), notes and a photo, and the page counts homes per status.
  Notes are for nicknames and logistics, not sensitive details.
- **Seeing a home:**
  - *Open in Google Maps* and *Satellite view*: plain links, no key. [V: [Maps URLs](https://developers.google.com/maps/documentation/urls/get-started)]
  - **Google Earth:** it has no documented embed or link-to-location scheme; its docs cover import
    from a file or Drive only [V: [import KML](https://developers.google.com/maps/documentation/earth/import-kml)].
    So **Download for Google Earth (KML)** exports your saved homes; in Earth use *File > Import KML file*.
    The KML has addresses, status and price, never your notes.
  - **3D view inside the page:** Google's Maps JavaScript API `<gmp-map-3d>` element, shown only when
    you turn on *Show 3D view* for one home and a key is configured (see next section).
- **Sign-in (optional):** Streamlit's built-in Google login. Without an `[auth]` section nothing changes.

## Turn on the 3D view

1. In Google Cloud, create a project with a **billing account**, enable the **Maps JavaScript API**,
   and create an API key. Which billing SKU a 3D map load uses I could not confirm [?]; pricing
   sources disagreed on the free allowance, so read the current price list first.
2. Restrict the key: **HTTP referrers** (your site address) and **API restrictions** (Maps JavaScript
   API only). Also set a **daily quota cap** in the console, so a bug or a crowd can't run up a bill.
3. Put it in `.env` (`GOOGLE_MAPS_API_KEY=...`) or the host's Secrets. A Maps JavaScript key is visible in
   the page by design, which is why step 2's restrictions matter.
4. Open Saved Homes, pick a home, switch on *Show 3D view*.

Things to know, in order of how likely they are to bite:

- **[?] Not tested live.** I had no key. The generated page matches Google's documented element and
  library name, but nobody has seen it render yet. Two things to check first: that **Atlanta has 3D
  coverage** (Google's table lists the US only at country level [V]), and that a **referrer-restricted
  key works inside Streamlit's embedded frame**, which may not send the referrer Google expects. If the
  map stays blank with a restricted key, loosen the restriction to test, then rethink.
- The map centers on an altitude above sea level, so the page asks **USGS** for the ground height
  at the home's coordinates, from the visitor's browser. It's a free US-government service that needs
  no key [V: returned 320 m for downtown Atlanta]. If it fails the map still opens, possibly aimed too low.
- **Google's tile terms:** visualization only; no caching or storing; no image analysis, machine
  interpretation or object detection; attribution must stay visible [V:
  [Map Tiles policies](https://developers.google.com/maps/documentation/tile/policies)]. So **never feed Google
  3D imagery to the photo scorer**. It keeps using the listing photos you supply.
- One map per home, off by default, because each load is billed. On a public site every visitor's
  3D view bills **your** key (unlike RentCast, there's no bring-your-own-key here), so the quota cap
  in step 2 is not optional.

## Turn on Google sign-in

1. Google Cloud > APIs & Services > Credentials > **OAuth client ID**, type *Web application*.
   Authorized redirect URI: `http://localhost:8501/oauth2callback` for local, and
   `https://YOUR-SITE/oauth2callback` for the website. Configure the consent screen; while it's in
   "testing" only listed test users can sign in.
2. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` (gitignored) and fill it in, or
   paste the same lines into the host's Secrets. Settings > Google sign-in shows the template and
   whether sign-in is on.
3. Restart. Visitors now see only a *Sign in with Google* button until they sign in. Verified here with
   placeholder credentials: the gate holds and the button redirects to Google [V]. A full sign-in with
   real credentials is untested [?].

What it does with the account:

- **Self-hosted (not "shared website" mode):** each signed-in email gets its own folder
  (`data/users/<hash>/`) with its own listings, photos, clients and saved homes. Data persists only if the
  host's disk does.
- **Shared website mode** (always on in Streamlit Community Cloud): sign-in is only a gate. Workspaces are
  still temporary, so **saved homes disappear** after about 2 idle hours. That's Stage 2.
- Streamlit doesn't check token expiry itself, and signing out clears only the current browser session
  [V: [docs](https://docs.streamlit.io/develop/concepts/connections/authentication)].

## Stage 2: saved homes that survive on a public site

The free hosts have no persistent disk ([DEPLOY.md](DEPLOY.md)). Two ways to fix it, and I'd pick by
where you'll host:

- **A host with a disk** (a small VM, or a platform volume): Stage 1's per-user folders already persist.
  Nothing to build; also keeps photos and embeddings, which are files anyway. Cost: a paid host.
- **Supabase** (the option you chose): hosted Postgres plus login. Worth it if you stay on a free
  disk-less host. What it takes, none of which I've done because creating the project is yours to approve:
  1. You create a Supabase project (check its current free-tier limits first [?]).
  2. A `saved_homes` table holding a **snapshot** (address, price, beds, baths, sqft, coordinates, status, notes)
     keyed by user, with a row-level-security rule so each user reads only their rows. Snapshots, because
     listing rows live in the temporary workspace.
  3. The app reads and writes it through the same four functions in `app/services.py` (`save_home`,
     `update_saved`, `unsave_home`, `saved_homes`), so the pages don't change.
  4. Store only what the visitor supplied (their uploads, their own-key RentCast results) plus status and
     notes. Never MLS photos or licensed data on a public site ([DEPLOY.md](DEPLOY.md) rule 1).

## Stage 3: leaving Streamlit

Do it when you need something Streamlit can't lay out well: a **map-and-list explorer** that stays in sync,
**read-only share links for clients**, polished **mobile** screens, or permissions beyond "one user, own data".
What carries over unchanged: `ranking/`, `vision/`, `sources/` and `core/` are plain Python with no
Streamlit in them, so a FastAPI layer over `app/services.py` plus a React or Next.js frontend can reuse them.
The cost is rebuilding six pages' UI (the labeling page's keyboard handling cost five browser-level fixes
last time, [VISION.md](VISION.md)), so keep Streamlit for the agent-only tools even then.

## Fair housing and data

Nothing here adds a ranking input. Saved status and notes don't affect tiers, and the Google views show
the property, not neighborhood scores. Keep notes to showing logistics and the client's nickname.
