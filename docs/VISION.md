# Photo condition scoring

What it does, how to use it, and where it falls short. Everything here describes the code in
`vision/`; numbers from a real-model test are at the bottom.

## What you get

For every listing photo:

- **Room type** (kitchen, bath, living, bedroom, exterior, other)
- **Condition 1 to 5** (1 poor, 5 like new) with a **confidence**
- **Flags**: water stain, damaged flooring, dated finishes, clutter, poor photo quality
- Photos that aren't of the property (**floor plans, maps, logos, aerial views, documents,
  headshots**) and **near-duplicates** are set aside automatically. You can override any of it.

For every listing: one **overall condition score**, a **worst room** marker, and a coverage
warning when there's no usable kitchen, bath or exterior photo. Rooms are weighted
(kitchen 3, bath 2.5, exterior 2, living 1.5, bedroom 1, other 0.5); edit them in Settings.
That score feeds the **condition** factor in the Find Homes tier list.

The model sees **only pixels**: never the address, price, listing text, or anything about the
neighborhood or people. Photos of people are never described; headshots are just excluded.

## Install (optional, one time)

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements-vision.txt
```

This adds PyTorch (CPU build, which is plenty), OpenCLIP, timm and scikit-learn. The first scoring
run downloads the model (about 600 MB) into your user cache. Without this install the rest of the
app works; the tier list simply omits photo condition. The public demo deliberately leaves it out.

## Day-one workflow

1. **Listings > Photos**: add photos to listings.
2. **Photo Review > Score this listing**. This uses the **zero-shot baseline** (the model compares
   each photo to written descriptions; no training). Good at room types and spotting floor plans;
   rough on condition.
3. **Labeling**: click *Prepare photos* once, then rate photos 1 to 5. Keyboard: `1`-`5` rate,
   `K B L R E O` room, `W F T C P` flags, `N` unsure, `S` skip, `U` undo, `X` exclude.
4. After ~50 labels you can **Train from my labels**. It compares two small models, tests the
   winner on a permanent holdout, and shows how it does against the baseline. If it's better,
   press **Use this model for scoring**, then re-score on Photo Review.
5. Keep labeling toward 300 to 500. The app shows the photos where a label helps most.

### The rubric (rate what you can see)

| Score | Meaning |
|---|---|
| 5 Like new | New or recently renovated; no visible wear; current finishes |
| 4 Good | Well maintained and clean; maybe lightly dated; no defects visible |
| 3 Average | Functional with normal wear; dated but intact; minor cosmetic issues |
| 2 Fair | Visible deferred maintenance; worn or damaged surfaces; needs real work |
| 1 Poor | Major problems: water damage, severe damage, gutted or unsafe-looking |

Judge surfaces, fixtures and damage, **not** furniture, staging or photography. This scale blends
wear with update level; the `dated finishes` flag keeps update level separable. It borrows the
spirit of appraisal condition ratings (Fannie Mae C1 to C6) but is not the same thing: an
appraisal rates a whole dwelling.

## How it works (short)

1. Each photo is turned into 512 numbers by a pretrained model ("embedding"), cached on disk by
   the photo's content hash so it's computed once.
2. **Baseline**: those numbers are compared with text prompts ("a kitchen that looks brand new..."),
   no labels needed.
3. **Your model**: a small head on the same numbers learns from your labels. Two are tried:
   *ridge + learned cut points* and *CORN-style chained classifiers* (rank-consistent, gives a
   probability per level). The better one on cross-validation wins. Models are plain arrays in
   `.npz` + JSON, never pickle.
4. **Honesty rules**: 1 photo in 5 (by file hash) is a permanent holdout that is never used to
   train or choose anything; the baseline is scored on the same photos; your own re-label
   consistency is reported as the realistic ceiling; a learning curve shows whether more labels
   still help.

Fine-tuning the whole network is deliberately not done: with a few hundred labels it overfits.
Consider it only past roughly 1,500 labels *and* a learning curve that is still dropping. That
threshold is a rule of thumb, not a result from the literature.

## Known limits (read these)

- **Single labeler.** The model learns your judgment, quirks included. Re-label ~10% a week later to
  measure your own consistency.
- **Staging and editing** can flatter a photo; the model judges what it sees.
- **Coverage matters.** A listing with only bedroom photos is flagged *limited coverage*.
- **Zero-shot flags are guesses.** They're shown as "possible", and trained only when you've
  labeled enough examples of each.
- **Thresholds are heuristics**: blur, brightness, resolution and duplicate cut-offs (in
  `config/vision.yaml`) were set by judgment, not tuned on a large real set.
- **Not an appraisal or inspection.** Use it to decide which homes to look at, not to value or
  describe a home to clients.

## Measured so far (read before trusting any number)

A smoke test of the **real** model (OpenCLIP ViT-B/32 `laion2b_s34b_b79k`, CPU only, no GPU) on 43
freely licensed Wikimedia Commons images that I picked by eye: 28 photos of rooms and house
exteriors (8 kitchens, 6 bedrooms, 6 living rooms including 3 scans of one photo, 8 exteriors)
and 15 non-property images (5 floor plans, 5 maps, 5 logos). Many room photos are **historical or
museum interiors**, not modern listing photos, and I had **no bathroom photos** (the download was
rate-limited), so this says little about real listings.

| What | Result |
|---|---|
| Speed | 6.4 photos per second; 63 s first load (includes the model download), 5 s afterwards |
| Room type (zero-shot) | 26 of 28 correct (kitchen 7/8, bedroom 5/6, living 6/6, exterior 8/8) |
| Floor plans, maps, logos auto-excluded | 15 of 15 |
| Real room photos wrongly excluded as non-property | 0 of 28 |
| Duplicates | the 3 scans of one photo collapse to 1, nothing else flagged (see the fix below) |
| Condition (zero-shot) | spread 1.3 to 4.0, mostly low confidence. A modern white kitchen scored highest of the kitchens (3.9) and two old narrow kitchens lowest (1.3). **There are no ground-truth ratings, so accuracy is unknown.** |

What this does **not** show: how well condition scores match your judgment, how well flags work on
modern listing photos, or how bathrooms behave. The **training loop** (label, train, evaluate,
activate, re-score) is tested on synthetic photos only; it has not been run on a real set of your
labels. Those numbers will appear in the model card the first time you train.

Real-data findings that changed the code:

- **Duplicates.** Re-scans of one photo differed by 5 to 9 bits of the image hash (the limit was 4)
  yet had embeddings 0.97 to 0.98 similar, versus 0.61 to 0.70 for different living-room photos.
  Duplicates are now "hash very close" or "hash fairly close AND embeddings nearly identical".
- **Flat images.** Plain-colored images all hash to the same value and would be called duplicates of
  each other, so near-flat images are skipped in duplicate checks.

Real-browser findings (Playwright + headless Chromium) that changed the code:

- After a shortcut clicked a radio button or checkbox, the browser kept focus on it and the key
  handler then ignored all further keys. Only text boxes should be ignored.
- Streamlit ignores a scripted click on a checkbox or radio label; the underlying input must be
  clicked.
- Text that looks like HTML tags inside the keyboard script made Streamlit skip the whole script.
- Giving the default page its own URL path produced a "Page not found" dialog.
- Two dollar signs in one line were rendered as a math formula in the tier-list cards.
