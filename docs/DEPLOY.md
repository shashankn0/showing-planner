# Launch it on the web for free (no domain purchase)

Checked 2026-10-08. Tags: **[V]** verified on an official page, **[S]** secondary source,
**[?]** not verified (I couldn't confirm it, or haven't run it end to end).

## Pick one

| | You get a URL like | Cost | Good for | Catches |
|---|---|---|---|---|
| **A. Streamlit Community Cloud** (recommended) | `your-name.streamlit.app` (you choose the name, 6-63 characters) [V] | Free | A public demo / "bring your own data" site; stable link to share | Needs a GitHub repo. Apps with no traffic for **12 hours go to sleep** (anyone can wake them) [V]. Limits (as of Feb 2024): up to 2 CPU cores, **2.7 GB RAM**, 50 GB storage [V]. Disk doesn't persist. Apps are public by default; **one** private app is allowed, shared with specific emails [V] |
| **B. Cloudflare quick tunnel** | `random-words.trycloudflare.com`, changes every run [V] | Free; **no account and no domain** [V] | Showing *your own running copy* (with your real data) to someone for a while | Only works while your PC and the app are running. "No uptime guarantee," 200 in-flight requests max, meant for testing [V]. I haven't run Streamlit through it end to end [?] |
| **C. Render free web service** | `your-name.onrender.com` [S] | Free | An always-reachable fallback | Spins down after 15 min idle (about 1 min to wake), 750 free hours/month, no persistent disk [V]. RAM on the free plan not confirmed [?] |

Not free anymore: Hugging Face Spaces. Their docs say Streamlit isn't an SDK option and Docker/Gradio
Spaces need a paid plan; only static Spaces are free [V]. GitHub Pages can't run Python.

## Before you put anything online (read this)

1. **Never put MLS-licensed listings or photos on a public site** unless your broker has approved a
   hosted tool. Public display and storage are governed by MLS/IDX rules. A public deployment
   should hold only demo data (the synthetic samples in `tests/fixtures/`) or data a visitor
   uploads into their *own* temporary workspace.
2. **Turn on hosted mode for any public site.** Without it, every visitor would share one
   database and see each other's uploads. Hosted mode gives each browser session an isolated
   temporary workspace, deleted after about 2 idle hours or on server restart, and the visitor can delete it any time from Settings. The app warns in
   the sidebar if it's reachable from the internet without a password or hosted mode.
3. **Keys go in the host's Secrets**, never in the repo. `.gitignore` already excludes `data/`,
   `.env` and databases.
4. The password gate (`APP_PASSWORD`) slows guessing (8 tries per page load, a growing delay after
   each wrong guess, and a 15-minute lock after 20 failures). Without `TRUST_PROXY_HEADERS=true`
   all visitors share one counter, so a persistent attacker can lock the gate for everyone; with it
   (honored only when the app is started with `--server.address 127.0.0.1`, so a proxy you
   control, like the Cloudflare tunnel in option B, is the only way in) failures are counted per
   client address; the sidebar warns if the setting is on but being ignored. It is not strong authentication. Use a long passphrase, and use
   Community Cloud's private app (email sign-in) for anything that matters.

## Option A: Streamlit Community Cloud, step by step

1. **Check what will be uploaded.** In PowerShell, from the project folder:
   ```powershell
   cd C:\Users\snell\showing-planner
   git ls-files | Select-String -Pattern "^data/|\.env$|\.db$"     # should print nothing
   ```
2. **Create a GitHub repository.** On github.com: *New repository*, name it `showing-planner`.
   Public is simplest. (Private may work if you authorize Streamlit's GitHub app for private
   repos; I couldn't confirm that in the docs [?].) Don't add a README or .gitignore there.
3. **Push the code:**
   ```powershell
   git remote add origin https://github.com/YOUR-USERNAME/showing-planner.git
   git push -u origin main
   ```
4. **Deploy.** Go to <https://share.streamlit.io>, sign in with GitHub, click **Create app**, and
   choose **Yup, I have an app**. Set:
   - Repository: `YOUR-USERNAME/showing-planner`, branch `main`
   - Main file path: `app/main.py`
   - App URL: pick a subdomain (this is your free address, `something.streamlit.app`)
   - **Advanced settings**: Python 3.12 (the default) and paste into *Secrets*:
     ```toml
     SHOWING_HOSTED = "true"
     NOMINATIM_USER_AGENT = "showing-planner/0.1 (contact: you@example.com)"
     ```
     The first line is the important one. The second is optional (enables the fallback
     geocoder; use your real contact email).
5. **Click Deploy.** The first build takes a few minutes. When it opens, check the sidebar says
   **"Mode: shared website"**. If it says "Mode: private", the secret didn't take effect: fix it
   before sharing the link (Manage app, then Settings, then Secrets).
6. **Try it** by uploading `tests/fixtures/synthetic_portal_style.csv` on the Listings page, then
   building a tier list on Find Homes.
7. **Updating:** every `git push` redeploys automatically. To restrict viewers to specific emails,
   use the app's *Share* settings (only one private app is allowed on the free plan).

## Option B: show your own copy with a Cloudflare quick tunnel (Windows)

This keeps your data on your PC. **Always set a password**: the link is reachable by anyone who
has it.

```powershell
winget install --id Cloudflare.cloudflared          # one time

# window 1: start the app with a password
cd C:\Users\snell\showing-planner
$env:APP_PASSWORD = "choose-something-long"
$env:TRUST_PROXY_HEADERS = "true"     # honored only because of --server.address 127.0.0.1 below
.\.venv\Scripts\streamlit.exe run app/main.py --server.port 8501 --server.address 127.0.0.1

# window 2: open the tunnel
cloudflared tunnel --url http://localhost:8501
```

`cloudflared` prints an `https://….trycloudflare.com` address; open it and sign in with the
password. Press Ctrl+C in each window to stop. The address is new every time.

## Photo scoring on a public site

Leave it off. It needs about 1 GB of extra packages and a ~600 MB model, which doesn't fit
Community Cloud's free memory comfortably (2.7 GB at most [V]) and would make every build slow.
`requirements.txt` deliberately excludes it, so the hosted app shows Photo Review and Labeling with
a note that scoring isn't enabled, and the tier list simply omits the photo-condition factor.
Run photo scoring on your own computer (private mode) where your photos and labels stay local.

## What a public-demo visitor sees

1. They open your `*.streamlit.app` link and see Find Homes with a **"Try it with demo listings"**
   button (36 made-up listings, labeled as demo data), so they can build a tier list in seconds.
2. They can instead upload their own spreadsheet or paste addresses; it lives only in their private
   temporary workspace (deleted after about 2 idle hours, or on demand from Settings).
3. They can search real listings with **their own** RentCast key (Listings > Search). Do **not**
   add `RENTCAST_API_KEY` to the site's Secrets: the hosted site ignores it by design, because
   RentCast forbids sharing one key across users and you'd be liable for their overage charges.

## If something goes wrong

- **"This app has gone over its resource limits"** (Community Cloud): reboot from *Manage app*; keep
  uploads small. The photo-scoring model (next phase) will need care to fit in 2.7 GB.
- **App asleep**: open the link and click *Yes, get this app back up!*
- **Build fails**: open *Manage app* and read the logs; the usual cause is a Python version or a
  package that won't install. `requirements.txt` pins what was tested (Python 3.12).
- **Geocoding/lookups fail**: the host must be able to reach the US Census geocoder over the
  internet, which free hosts normally allow.
