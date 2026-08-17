# Building the Asset Library

The pipeline needs 3D models to place. This guide gets you a few hundred
game-ready models for free, with no licensing complications.

---

## Why these packs

We use **Kenney's** and **Quaternius's** asset packs. Both are **CC0** (public
domain): free for any use, commercial included, with **no attribution
required**. Nothing to credit in your report, no licence to comply with.

### Modular kits vs. complete models — read this first

Many free packs are **modular**: instead of one `house.glb`, you get
`wall.glb`, `roofCorner.glb`, `chimneyBase.glb`, meant to be snapped together
on a grid by an artist. Those are useless to this pipeline as standalone
objects — a scene asking for a house cannot be handed a roof corner.

We learned this the hard way. A 862-asset library built from Kenney's Castle
Kit and Fantasy Town Kit contained **zero complete houses**: 98 of Fantasy Town
Kit's assets were wall/fence/fountain segments and only 8 were whole objects.
A village prompt resolved "house" to a *tent* and "forge" to a *chimney base*.

So: **you need at least one pack of complete buildings.** The ingest script
now flags modular fragments and retrieval ranks them below whole models, but
it can't invent a house that isn't there.

---

## 1. Download the packs

### Essential — complete buildings

| Pack | Covers | Link |
|---|---|---|
| **Ultimate Fantasy RTS** (Quaternius) | **complete houses, huts, barracks, farms, windmills, towers** | https://poly.pizza/bundle/Ultimate-Fantasy-RTS-nSDjmACoSU |

107 whole models, CC0. This is the pack that makes medieval village prompts
work. Download the glTF/GLB version if offered.

### Essential — everything else

| Pack | Covers | Link |
|---|---|---|
| **Nature Kit** (Kenney) | trees, rocks, bushes, logs | https://kenney.nl/assets/nature-kit |
| **Survival Kit** (Kenney) | tents, campfires, crates, lanterns | https://kenney.nl/assets/survival-kit |
| **Graveyard Kit** (Kenney) | lanterns, torches, medieval props | https://kenney.nl/assets/graveyard-kit |
| **Space Kit** (Kenney) | domes, platforms, sci-fi props | https://kenney.nl/assets/space-kit |

### Optional — modular detail

| Pack | Covers | Link |
|---|---|---|
| Fantasy Town Kit (Kenney) | walls, fences, fountains, market stalls | https://kenney.nl/assets/fantasy-town-kit |
| Castle Kit (Kenney) | modular castle walls, towers, gates | https://kenney.nl/assets/castle-kit |

Keep these — their fences, fountains and stalls are genuinely useful as
`structure` and `prop` assets. Just don't rely on them for buildings.

You want coverage of all six categories (`building`, `prop`, `vegetation`,
`rock`, `structure`, `light_source`) — the ingest script reports any that are
thin.

---

## 2. Unzip them into place

Create `data/asset_library/raw/` and give each pack its own folder:

```
data/asset_library/raw/
    kenney_nature-kit/
    kenney_survival-kit/
    kenney_castle-kit/
    kenney_space-kit/
```

Folder names matter a little: the ingest script reads them to guess which
themes a pack suits (a folder with "space" in the name gets boosted for
sci-fi scenes). Keeping the `kenney_` prefix and the pack name works.

> These files are **not committed to git** — they're large and freely
> re-downloadable. `.gitignore` already excludes them. Your teammates follow
> this same guide to get an identical library.

---

## 3. Build the manifest

```powershell
python scripts/ingest_assets.py --verbose
```

This scans every model file, cleans up the names, sorts them into categories,
measures each mesh's real footprint, and writes
`data/asset_library/manifest.json`.

You'll get a report like:

```
Indexed 412 assets from 4 pack(s)

By category:
  building         67
  prop             94
  vegetation      131
  rock             38
  structure        52
  light_source     30
```

**If a category shows 0**, download another pack that covers it. Scenes will
still generate (the resolver substitutes a related category) but they'll look
wrong — a village lit by barrels instead of torches.

**If `light_source` is under ~20**, add the Graveyard Kit. Lights are the
thinnest category across Kenney's packs and the easiest to end up short on.

A healthy library looks roughly like:

| Category | Want at least |
|---|---|
| building | 40 |
| vegetation | 60 |
| prop | 60 |
| rock | 30 |
| structure | 40 |
| light_source | 15 |

Re-run this script any time you add packs. It rebuilds from scratch.

---

## 4. Install the semantic embedder (recommended)

Retrieval works out of the box using a built-in character-matching embedder —
no download needed. But it only matches spelling, so it can't tell that
"smithy" and "forge" mean the same thing.

For proper semantic matching:

```powershell
pip install sentence-transformers
```

First run downloads a ~90 MB model, then works offline forever. It's small
enough to run comfortably alongside Ollama on your 8 GB card.

The pipeline picks the semantic embedder automatically when it's installed and
falls back to character matching when it isn't — so nothing breaks either way.

---

## 5. Try it

```powershell
python -m src.cli "a foggy medieval village at dusk with a blacksmith forge" --resolve
```

You should see a table mapping each requested object to a real model file:

```
requested            ->  asset                        cat           qty  score
------------------------------------------------------------------------------
house                ->  house small                  building        5  0.91
blacksmith forge     ->  building smithy              building        1  0.63
tree                 ->  tree pine a                  vegetation     18  0.99
bridge               ->  bridge wood                  structure       1  0.86
```

Assets marked `*` are low-confidence matches — the library has nothing good for
that object. That's your signal to download another pack.

**Read the scores, not just the names.** A match around 0.85–0.99 is a real
hit. Anything in the 0.5–0.65 range usually means the library has no such
object and retrieval is reaching for the closest wrong thing. We hit exactly
this: with no house model in the library, "house" resolved to a *tent* at 0.62
and the scene would have rendered a village made of tents.

---

## Diagnosing a bad result

When an object resolves to something obviously wrong, check whether the model
exists at all before assuming retrieval is broken:

```powershell
python scripts/inspect_library.py --search house
python scripts/inspect_library.py --category building --limit 70
python scripts/inspect_library.py --pack fantasy
```

`--search` is the decisive one. If it returns **0 assets**, no amount of
tuning will help: retrieval can only return what's in the library. Download a
pack that contains the missing object.

Assets marked `[mod]` are modular fragments and are ranked below complete
models automatically.

---

## Troubleshooting

**"No manifest at data/asset_library/manifest.json"**
Run `python scripts/ingest_assets.py`.

**"no pack folders found"**
Your packs are unzipped one level too deep or too shallow. You want
`raw/<pack-name>/<model files somewhere inside>`.

**Lots of assets "could not be measured"**
Some formats don't load cleanly in trimesh. Harmless — those fall back to
category default sizes. Prefer packs offering `.glb` or `.obj` files.

**Everything resolves to the same asset**
Your library is too small or missing a category. Add another pack.
