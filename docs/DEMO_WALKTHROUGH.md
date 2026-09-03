# Demo Walkthrough

## Text-to-3D Game Environment Generation

Everything to run and say on the day, plus a full account of the repository
and where the project goes next.

---

# PART 1 — BEFORE THE DEMO

## Setup on the laptop

```powershell
cd C:\Users\rohit\OneDrive\Desktop\text-to-3d-env-gen
.venv\Scripts\Activate.ps1

python -m src.cli --check         # confirm the local model is up
python -m pytest tests/ -q        # 247 passed
python -m http.server -d outputs 8000
```

Leave the server running. Open `http://localhost:8000` in one tab. Keep a
second terminal ready for the live generation.

## Pre-generate the scenes

Generate on the desktop (RTX 5080) and copy the `outputs/` folder across. The
viewer is static files — HTML, glTF, JSON — so displaying a scene needs no GPU
at all. Only generating one does.

```powershell
python scripts/generate.py "a dense forest camp at night with a campfire" --size large --seed 3 --palette
python scripts/generate.py "a foggy medieval village at dusk with a blacksmith forge" --size large --seed 42
python scripts/generate.py "a medieval village on a sunny day" --size large --seed 7
python scripts/generate.py "a small desert outpost in a sandstorm" --size large --seed 11

# for the before/after comparison
python scripts/generate.py "a medieval village on a sunny day" --seed 7 --density 0 --out outputs_bare
```

**Lead with the forest camp.** It's the strongest image: green canopy, brown
trunks, tents, log piles, and campfires casting real pools of warm light.

---

# PART 2 — THE DEMO SCRIPT

## 1. The problem (20 seconds)

Building a 3D game environment means blocking out terrain, sourcing assets,
placing hundreds of objects, lighting the scene, then testing that it's
actually walkable. Days of work for a prototype that may be thrown away.
Small teams and students can't afford that.

## 2. Show the result first (30 seconds)

Open the forest camp. Orbit once. Press **W** and walk down the path.

> "This was generated from one sentence. Two and a half thousand objects, no
> manual placement, no 3D modelling, about twenty seconds."

## 3. Then show how — run it live (2 minutes)

```powershell
python scripts/generate.py "a foggy medieval village at dusk with a blacksmith forge"
```

Talk through the stages as they print.

**[1/5] Prompt decomposition.** A language model running locally on our own
GPU reads the sentence and produces a structured scene specification. Point
out that it inferred houses, cottages and trees — *the prompt never mentions
them*. That's the model understanding what a village implies, not keyword
matching.

**[2/5] Asset resolution.** Each object is matched to a real 3D model from a
1001-model library, by meaning rather than spelling — "blacksmith forge" finds
a smithy despite sharing no whole word. Confidence scores are shown, and weak
matches are flagged rather than hidden.

**[3/5] Terrain.** A heightmap generated to suit the terrain type, then graded
flat along the road so the route reads as built rather than painted on.

**[4/5] Placement.** Every object gets a position. Point at the validation
line: **zero overlaps, zero floating objects, nothing blocking the path** —
checked programmatically, not by eye.

**[5/5] Export.** One glTF scene plus a viewer. glTF is the standard
interchange format, so the same file opens in Unity, Unreal or Blender.

Then open the result.

## 4. The before/after (30 seconds)

Show `outputs_bare` beside the normal output.

> "The language model lists what the prompt *mentions*. A believable place is
> mostly the things nobody thinks to say — grass, pebbles, fence posts,
> lanterns at dusk. Thirty-five objects versus two thousand."

## 5. What's measured, not claimed (30 seconds)

```powershell
python scripts/evaluate.py --fallback --weak
```

| Metric | Result |
|---|---|
| Structurally valid | 26/26 prompts |
| Prompt coverage | 100% |
| Retrieval confidence | 0.80 – 0.91 |
| Density spread across themes | 1.6x |
| Generation time | 0.2–1.6s without the LLM, 8–11s with |
| Automated tests | 247 passing |

> "Twenty-six prompts covering all four themes, awkward paraphrases, and
> deliberately out-of-scope requests. Every one produces a structurally valid
> scene."

## 6. What's novel (45 seconds)

> "Published systems in this space — including Tencent's WorldClaw this year —
> generate environments optimised to *look* coherent. None optimise for
> whether the space is interesting to move through. We generate several
> candidate layouts per prompt and score them with a simulated agent that
> explores each one, measuring path variety, chokepoints and how well points
> of interest are spaced. The best-scoring layout wins. That scoring step is
> our contribution and the subject of the research paper."

## 7. Own the limitations (30 seconds)

Say these before you're asked. They're the strongest part of the presentation.

> "Three limits, and we measure all of them.
>
> **Four themes are supported.** Anything else is approximated from the
> closest one — and the system says so, printing a warning and labelling the
> scene 'approximated' in the viewer. Try asking it for an underwater city.
>
> **Retrieval can only return what the library holds.** We have no well, no
> forge, and twelve light fixtures across a thousand models. The system
> reports those as low-confidence matches instead of silently substituting.
>
> **Placement is statistical, not relational.** The prompt says 'a campfire',
> singular, and the system treats it as a density parameter. It has no notion
> that a campfire is a focal point that tents should face. That's the clearest
> limitation we've found and the next stage of the work."

---

# PART 3 — WHAT'S IN THE REPOSITORY

## The pipeline

```
   TEXT PROMPT
        |
   [1] PROMPT DECOMPOSITION        prompt_decomposition.py, llm_client.py
        |   local language model -> structured Scene Specification
        v
   SCENE SPECIFICATION (JSON)      schema.py, vocab.py
        |
   [2] ASSET RESOLUTION            asset_index.py, asset_resolution.py
        |   each object matched to a real 3D model, by meaning
        v
   RESOLVED SCENE                  objects + meshes + real dimensions
        |
   [3] TERRAIN + LAYOUT            terrain.py, layout.py, dressing.py
        |   ground, road, settlement, hundreds of scattered details
        v
   PLACED SCENE                    every instance positioned
        |
   [4] EXPORT + VIEW               export_scene.py, viewer.py, appearance.py
             glTF scene + real-time browser viewer
```

## Source files

| File | What it does |
|---|---|
| `vocab.py` | The controlled vocabulary — every theme, mood, object category and placement rule the system understands. Single source of truth: adding a theme means editing this file only. |
| `schema.py` | The Scene Specification data structure, with validation. Anything the language model returns is checked and corrected here before reaching the rest of the pipeline. |
| `llm_client.py` | Talks to the local Ollama server. |
| `prompt_decomposition.py` | **Stage 1.** Builds the model instruction, parses the reply, validates it, falls back if anything fails. |
| `fallback_parser.py` | Keyword-only parser. Never fails. Used when the model is unavailable, and as the baseline that demonstrates what the model adds. |
| `asset_rules.py` | Turns messy model filenames into clean searchable names, sorts each into a category, and flags modular fragments. |
| `asset_index.py` | **Stage 2 search.** Converts names to vectors, finds closest matches, prefers literal word matches over embedding similarity. |
| `asset_resolution.py` | **Stage 2 logic.** Binds every object to a real mesh, handles missing categories, stops two objects sharing one model. |
| `terrain.py` | **Stage 3.** Heightmap generation, height and slope queries, road grading, building pads. |
| `layout.py` | **Stage 3.** Object placement, collision avoidance via a spatial grid, settlement structure, path clearance. |
| `dressing.py` | **Stage 3.** Fills the world with the background detail no prompt mentions, to a per-theme density target. |
| `appearance.py` | Real-world scale normalisation and the colour palette, including roofs on buildings and trunks on trees. |
| `export_scene.py` | **Stage 5.** Writes terrain, copies each model once, records every instance transform. Preserves original materials. |
| `export_gltf.py` | Alternative export that bakes everything into one file with flattened colours. |
| `viewer.py` | Generates the real-time browser viewer: lighting, weather fog, shadows, tone mapping, walk mode. |
| `cli.py` | Command-line interface for individual stages. |

## Scripts

| Script | What it does |
|---|---|
| `generate.py` | **End to end.** Prompt in, viewable 3D scene out. This is the demo command. |
| `ingest_assets.py` | Scans downloaded asset packs, cleans names, categorises, measures each mesh, writes the manifest. |
| `inspect_library.py` | "What is actually in my library?" Built to debug a bad result — see below. |
| `inspect_scene.py` | "What is actually in this scene?" Reports the largest objects and each model's colour. |
| `evaluate.py` | Runs 26 prompts and reports validity, coverage, confidence, density and timing. The source of every number quoted above. |

## Documentation

| File | Contents |
|---|---|
| `docs/SETUP.md` | Getting the project running from scratch |
| `docs/ASSETS.md` | Building the 3D asset library, and why some packs work and others don't |
| `docs/WALKTHROUGH.md` | Full technical explanation of what's built and why |
| `docs/DEMO.md` | This runbook |

## The design principle worth pointing out

Every stage communicates through **one shared data structure** — the Scene
Specification. Stage 1 writes it, stage 2 reads and extends it, stage 3 does
the same. Each stage can be built and tested in isolation, and a change to one
cannot silently break another.

The 247 tests run **without** a GPU, the language model, or the asset library,
using a synthetic mini-library built in code. They pass on any machine.

---

# PART 4 — GENERATED VS. RETRIEVED

Expect the question "so what is the AI actually generating?" Be precise.

| Component | Generated or retrieved |
|---|---|
| Scene specification — objects, counts, mood, terrain | **Generated** — language model inference |
| Terrain heightmap | **Generated** — fractal noise |
| Road network and graded corridor | **Generated** |
| Ground painting — dirt, wear, colour variation | **Generated** |
| Settlement structure and every object's position, rotation, scale | **Generated** — this is where the novelty lives |
| The 3D object meshes | **Retrieved** from a curated library |

> "We generate the scene; we retrieve the assets. That's what the published
> systems do too — AutoUE retrieves from a database of 858,000 models, and
> WorldClaw's paper describes reusable assets. Retrieval gives game-ready
> geometry and a consistent art style, which generative 3D still doesn't."

---

# PART 5 — WHAT COMES NEXT

Have this ready — it shows the project is a stage in a plan, not a finish line.

| Work | Why | Cost |
|---|---|---|
| **Relational scene specification** | The prompt says "a campfire", singular, and the system treats it as a density parameter. The specification needs spatial *relations* — campfire at the camp centre, tents facing it, path leading to it — not just objects and counts. | Free |
| **Constraint-based layout** | Placement should solve those relations rather than sampling each object independently. | Free |
| **Gameplay-flow scoring** | The novelty: a simulated agent explores candidate layouts; the best-scoring one wins. Needs the metric suite, an ablation against random selection, and a preference study. | Free |
| **One commercial asset pack** | Would fix the material problems outright and remove the collage look of mixing packs. | One-off, ~$20–60 |
| **More themes** | Four is the current limit. Each new theme needs vocabulary, filler plans and assets. | Mostly free |
| **Confidence-gated asset generation** | The weak-match report is already the trigger: when retrieval can't serve a request, generate the mesh and cache it into the library. TRELLIS is MIT-licensed, does text-to-3D, and wants a 16 GB GPU — which the desktop has. | Free, needs setup |

**The honest comparison to WorldClaw:** their paper states the pipeline
requires Claude Opus 4.8, GPT-Image-2 and Hunyuan3D — a paid frontier language
model, a paid image model, and a large 3D generation backbone. We can't match
that fidelity. But the architecture is the same shape: planning agent,
structured specification, terrain foundation, asset placement. We built it to
run on one consumer GPU at zero cost.

---

# PART 6 — QUESTIONS YOU'LL GET

**"Is the AI really doing anything, or is this keyword matching?"**

Two models. A 7-billion-parameter language model interprets the prompt and
infers unstated content — a village gets houses even though the prompt never
says "houses". A sentence-embedding model matches objects to assets by meaning.
We deliberately kept a keyword-only parser as a baseline: run `--fallback` and
compare. The keyword version finds four object types where the model finds
seven, and has to be told a village needs vegetation.

**"What's the polygon count?"**

Around 300,000 triangles rendered for a large scene. Across the 1001-model
library the median model is 140 triangles and the mean 572 — the mean skewed
by a few outliers, the heaviest a crops model at 35,076. Each distinct model
is stored once and instanced, so stored geometry is far smaller than rendered.
Modern GPUs handle millions, so there's plenty of headroom.

On edges specifically: smooth vertex normals stop curved surfaces showing
facets. That's a shading choice costing no extra triangles — an early version
exported no normals at all and the ground rendered as harsh flat panels.

**"Does it only work for one prompt?"**

No, and we measure it rather than assert it. `evaluate.py` runs 26 prompts and
reports the table above. Every one produces a structurally valid scene.

**"What if I ask for something you don't support?"**

Four themes are supported; anything else is approximated from the closest one —
and the system says so. Ask it for an underwater city and show the warning.

**"Why not train your own model?"**

We evaluated it and wrote it up. Training a 3D generative model from scratch
needs datasets of millions of models and far more compute than one consumer
GPU offers in a semester. Even the 2026 papers in our review orchestrate or
fine-tune existing models rather than pretraining. Our original contribution
is the gameplay-flow scoring instead.

**"How do you know it's working, rather than just running?"**

Confidence scores and validation. Our best example: the system once reported
complete success — every object resolved, no errors — while producing a village
made of *tents*, because the asset library contained no houses. The scores
caught it: 0.62 for "house" against 0.90 for genuine matches. Measuring output
quality rather than checking for exceptions is how we found most of our bugs.

**"Does it run in a game engine?"**

We export glTF, the standard interchange format, rather than to any one engine.
The same file imports into Unity, Unreal or Blender. We use a browser viewer
for the demo because it needs no install and runs anywhere.

**"What was hardest?"**

The bugs that didn't announce themselves. A village of tents that reported
success. A hash function randomised per process, so the same prompt returned
different assets every run. Keywords matching as prefixes — "inn" matching
"inner", filing cliff faces as buildings. A size cap applied before the random
jitter that then pushed objects past it. Sky colour fed into the ambient light,
tinting green foliage teal. Each one was found by measuring output rather than
watching for crashes.

---

# PART 7 — IF SOMETHING GOES WRONG

| Symptom | Fix |
|---|---|
| `--check` says NOT REACHABLE | `ollama serve` in another terminal, or add `--fallback` |
| Generation is slow | Use a pre-generated scene; the model is loading into VRAM |
| Viewer shows "Could not load" | It needs HTTP, not `file://`. Use `python -m http.server` |
| Scene looks sparse | Wrong folder — check you're serving `outputs`, not `outputs_bare` |
| Colours look wrong | Regenerate with `--palette` |
| Laptop struggles | Add `--density 0.8` |

Worst case: everything is pre-generated and static. Open the folder and present
from that.

---

## The three things to make sure you say

1. **The keyword-versus-model comparison.** It proves the AI is doing real
   work, in one command.
2. **The village of tents.** A silent failure caught by confidence scores, not
   by an error. It shows you measure quality rather than absence of crashes.
3. **Placement is statistical, not relational.** Naming your own most
   interesting limitation, before anyone asks, is the strongest thing in the
   presentation.
