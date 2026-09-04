# Project Technical Reference

## Generative AI Model for 3D Game Environment Generation from Text Prompts

Prepared for Mock-1. This is a study document, not a script — it explains the
project from first principles so that any question can be answered from
understanding rather than memory.

**How to use it:** Part 1 gives the fifteen-second and two-minute answers.
Parts 2–7 explain every stage, including the algorithms and why each was
chosen. Part 8 maps everything onto the mock's marking rubric. Part 9 is a
bank of hard questions with worked answers.

---

# PART 1 — THE PROJECT IN THREE DEPTHS

## Fifteen seconds

A system that turns one sentence of English into a complete, walkable 3D game
environment — terrain, road, buildings, vegetation, props, lighting — in about
twenty seconds, running entirely on one consumer laptop at zero cost.

## Two minutes

Building a 3D game environment is slow: block out terrain, source assets,
place hundreds of objects, light it, test that it's walkable. Days of work for
a prototype that may be thrown away. That cost falls hardest on students and
small teams.

Our system takes a sentence — *"a foggy medieval village at dusk with a
blacksmith forge"* — and produces a finished environment through five stages:

1. A **language model** reads the sentence and produces a structured scene
   specification, inferring what the prompt implies but never says (a village
   needs houses, a dusk scene needs lanterns).
2. A **semantic retrieval** stage matches each named object to a real 3D model
   from a 1,001-model library, by meaning rather than spelling.
3. A **procedural stage** generates terrain, carves a road, and places every
   object — around 2,000 of them — with no overlaps, nothing floating, and a
   clear walkable path.
4. **Scene dressing** fills in the hundreds of details no prompt mentions.
5. **Export** writes standard glTF plus a real-time browser viewer, and the
   same file imports directly into Unity.

We measured it across 26 prompts: every one produces a structurally valid
scene, retrieval confidence sits between 0.80 and 0.91, and scene density
stays within a 1.6× band across all four themes. 259 automated tests.

## The numbers

| | |
|---|---|
| Pipeline code | 5,920 lines of Python across 22 files |
| Test code | 2,444 lines, 259 tests |
| Asset library | 1,001 models, all CC0 |
| Typical scene | 2,000–2,500 objects, ~300,000 triangles |
| Generation time | 8–11s with the language model, under 2s without |
| Cost | Zero — no APIs, no subscriptions, no cloud |

---

# PART 2 — ARCHITECTURE

## The pipeline

```
   "a foggy medieval village at dusk with a blacksmith forge"
        |
   [1] PROMPT DECOMPOSITION           prompt_decomposition.py, llm_client.py
        |   Qwen2.5-7B via Ollama, locally
        v
   SCENE SPECIFICATION (JSON)         schema.py, vocab.py
        |   theme, terrain, lighting, object list with quantities
        v
   [2] SCENE DRESSING                 dressing.py
        |   + ~2,000 background objects the prompt never mentioned
        v
   [3] ASSET RESOLUTION               asset_index.py, asset_resolution.py
        |   each object -> a real mesh, by semantic similarity
        v
   [4] TERRAIN + LAYOUT               terrain.py, layout.py, appearance.py
        |   heightmap, road, settlement, every position/rotation/scale
        v
   [5] EXPORT                         export_scene.py, export_gltf.py, viewer.py
            glTF + browser viewer + Unity bundle + lighting data
```

## The single most important design decision

**Every stage communicates through one shared data structure: the Scene
Specification.** Stage 1 writes it, stage 2 extends it, stage 3 reads it,
stage 4 consumes the result.

Why this matters:

- Each stage can be built and tested in isolation
- A change in one cannot silently break another
- The tests can run without a GPU, without the language model, and without
  the asset library, because they construct specifications directly

If asked "what would you say is well-engineered about this?", this is the
answer.

## The Scene Specification

```json
{
  "theme": "medieval_village",
  "art_style": "low_poly",
  "terrain":  { "type": "grassland", "size": "medium" },
  "lighting": { "time_of_day": "dusk", "weather": "fog", "mood": "mysterious" },
  "objects": [
    { "name": "house", "category": "building",
      "placement": "along_path", "quantity": 7 }
  ],
  "source_prompt": "...",
  "parser": "llm",
  "theme_recognised": true,
  "warnings": []
}
```

Validated by `schema.py` using **pydantic**, which enforces types and runs
custom validators on every field.

---

# PART 3 — STAGE 1: PROMPT DECOMPOSITION

## What it does

Turns ambiguous English into a strict machine-readable structure.

## The model

**Qwen2.5-7B-Instruct**, run locally through **Ollama** — a server that hosts
open language models on your own hardware, listening on `localhost:11434`.

**Why local rather than an API:** zero cost, no API keys, no internet
dependency during a demo, and a pinned model version so results stay
reproducible. The project has no budget; a paid API would have made it
unrunnable.

**"7B"** means seven billion parameters. Quantised to 4-bit it occupies about
4.7 GB of the laptop's 8 GB of VRAM.

## How the call works

```python
requests.post("http://localhost:11434/api/generate", json={
    "model": "qwen2.5:7b-instruct",
    "system": SYSTEM_PROMPT,     # the rules and the allowed vocabulary
    "prompt": user_prompt,
    "format": "json",            # constrains decoding to valid JSON
    "options": {"temperature": 0.2},
})
```

Two details worth understanding:

**`format: "json"`** constrains the model's token sampling so it *cannot*
emit anything that isn't valid JSON. This is enforcement at the decoding
level, not a polite instruction in the prompt — far more reliable.

**`temperature: 0.2`.** Temperature controls randomness in token selection.
At 0 the model always picks its highest-probability token; at 1.0 it samples
freely. We want consistent structured output, so we keep it low but not zero —
a little variation avoids identical scenes for identical prompts.

## Why this is genuine inference, not keyword matching

Given *"a foggy medieval village at dusk with a blacksmith forge"*, the model
outputs houses, cottages, trees and bushes. **None of those words appear in
the prompt.** It knows a village implies dwellings.

We kept a deliberately dumb keyword parser (`fallback_parser.py`, 259 lines)
as a control:

| | LLM | Keyword parser |
|---|---|---|
| Object types found | 7 | 4 |
| Inferred houses? | Yes | No — had to be told a village needs vegetation |
| Mood | mysterious | abandoned (matched the word "broken") |

Run `--fallback` to show this live. It's the cleanest single demonstration
that the AI is doing real work.

## Validation and repair

The model's output is never trusted. `schema.py` does four things:

1. **Snapping** — near-miss values are pulled onto the vocabulary. "foggy"
   becomes "fog", "lowpoly" becomes "low_poly", using exact match, then
   underscore-insensitive match, then fuzzy match at 0.75 similarity.
2. **Category overrides** — the model reliably calls a forge a "structure";
   it's a building you enter. A lookup table corrects known mistakes.
3. **Non-physical filtering** — the model once listed "smoke" as an object.
   Smoke has no mesh. Atmosphere and effects are dropped with a warning.
4. **Caps and merging** — quantities clamp to 40, object types to 12,
   duplicates merge.

## Graceful degradation

If Ollama isn't running, the keyword parser takes over automatically and the
pipeline still produces a valid scene. **Every stage has a fallback like
this** — no embedding model falls back to character matching, a missing asset
category substitutes a related one. A live demo cannot hard-fail.

## Honest handling of unsupported prompts

Four themes are supported. Ask for an underwater city and the system
approximates from the closest — **and says so**, printing a warning and
labelling the viewer "(approximated)".

It cross-checks the model's theme choice against keyword evidence: if the
prompt contains no word associated with any supported theme, the choice was a
guess and the user is told.

---

# PART 4 — STAGE 2: ASSET RESOLUTION

## The problem

The specification says *"blacksmith forge"*. The library contains
`building_smithy.glb`. Exact string matching finds nothing — they share no
whole word.

## Embeddings, from first principles

An **embedding** converts text into a list of numbers (a vector) representing
its meaning. Similar meanings produce vectors that point in similar
directions.

We use **all-MiniLM-L6-v2** from the Sentence-Transformers project, hosted on
Hugging Face, Apache-2.0 licensed. It's a 6-layer transformer producing
**384-dimensional** vectors, about 90 MB, and runs on CPU in milliseconds. It
was trained on over a billion sentence pairs.

## Cosine similarity

To compare two vectors we measure the **cosine of the angle between them**:

```
similarity = (A · B) / (|A| × |B|)
```

Identical direction gives 1.0, perpendicular gives 0.0. We normalise every
vector to unit length in advance, so the denominator becomes 1 and the whole
comparison reduces to a dot product.

With all 1,001 asset vectors stacked into a matrix, one query is a single
matrix multiplication:

```python
scores = self._matrix @ query_vector     # 1001 similarities at once
```

**Why not a vector database like FAISS or ChromaDB?** At 1,001 assets a numpy
dot product takes microseconds. A database would add an install dependency and
complexity for no measurable gain. We chose the simpler thing deliberately —
and that's a defensible engineering answer, not a shortcut.

## Four ranking signals

Pure embedding similarity wasn't enough. Retrieval combines:

| Signal | Weight | Why |
|---|---|---|
| Cosine similarity | base | semantic meaning |
| **Literal word match** | +0.15, and filters | if any asset's name contains a query word, search only those |
| Modular fragment penalty | −0.20 | roof corners aren't houses |
| Theme match / mismatch | +0.05 / −0.04 | a nudge, never a veto |

**The literal-match filter is the most important and was the hardest to
find.** Our library has ~100 medieval buildings and ~10 sci-fi ones. Embedding
similarity alone let the medieval majority win: a sci-fi scene asking for a
"platform" matched an *archery building*, "crystal" matched an *astronaut*.
Every one of those had a correctly-named asset available. Restricting to
literal name matches when any exist moved sci-fi confidence from 0.70 to 0.87.

## Modular fragments

Many free packs are **modular**: instead of `house.glb` you get `wall.glb`,
`roofCorner.glb`, `chimneyBase.glb`, meant to be snapped together by an
artist. Those are useless as standalone objects.

We detect them by positional words (`corner`, `inner`, `mid`, `edge`, `base`,
`cliff`, `terrain`, `road`) and rank them below complete models. 286 of 1,001
assets are flagged.

## Confidence and honesty

Every match carries a score. Below 0.70 (for the semantic embedder) it's
flagged as weak. The remaining weak matches in our library are genuine gaps —
we have no well, no forge, and only 12 light fixtures across 1,001 models.

**The system reports this rather than silently substituting.** The `--weak`
report is effectively a shopping list of what to add next.

---

# PART 5 — STAGES 3 & 4: TERRAIN, DRESSING, LAYOUT

## Terrain generation

### Fractal value noise, explained

Random numbers alone give static, not landscape. **Value noise** works by:

1. Generate random values on a coarse grid (say 4×4)
2. Interpolate smoothly between them to fill a fine grid (128×128)
3. Repeat at doubled frequency and halved amplitude — each repetition is an
   **octave**
4. Sum the octaves

Low octaves give the broad shape of hills; high octaves add fine roughness.
**Persistence** (0.5 in our presets) controls how fast later octaves fade.

For interpolation we use the **smoothstep** curve `t²(3 − 2t)` rather than
straight linear blending, which removes the visible grid artefacts linear
interpolation produces.

**Ridged noise** — used for rocky terrain — folds the noise around its
midpoint (`1 − |2n − 1|`), turning smooth hills into sharp crests.

### Why numpy rather than the `noise` package

The standard PyPI `noise` package requires a C compiler, a common source of
installation failure on Windows. Writing it in numpy costs a few lines,
removes a whole class of setup problems, and is fully deterministic given a
seed.

### The five presets

| Terrain | Amplitude | Octaves | Ridged | Flat bias |
|---|---|---|---|---|
| grassland | 4.5 m | 4 | no | 0.18 |
| forest_floor | 3.5 m | 4 | no | 0.20 |
| desert_sand | 5.5 m | 3 | no | 0.12 |
| rocky | 9.0 m | 5 | **yes** | 0 |
| barren_rock | 7.0 m | 4 | **yes** | 0.10 |

Amplitudes are deliberately modest. Dramatic terrain looks good in isolation
but makes a village unplaceable — playability matters more.

### Querying the surface

Objects must sit *exactly* on the ground. `height_at(x, z)` uses **bilinear
interpolation**: it finds the four surrounding grid points and blends them by
distance. Sampling the nearest grid cell instead would leave objects visibly
floating or sunken on slopes.

`slope_at(x, z)` samples nearby heights and returns the gradient magnitude,
used to keep buildings off cliff faces.

Two shaping operations: `flatten_disc()` levels a pad under each building
(feathered at the edge so there's no visible step), and `flatten_path()`
grades the road corridor so it reads as built rather than painted on.

## Scene dressing

**The insight:** the language model lists what the prompt *mentions*. A
believable place is mostly what nobody thinks to say — grass tufts, pebbles,
fence posts, fallen logs, lanterns at dusk.

Without dressing: 35 objects across a 120 m scene. With it: over 2,000.

Each theme has a filler plan with densities per 100 m², plus:

- **Settlement buildings** counted directly, not by area — a village's size is
  set by how many people live there, not how much terrain surrounds it
- **Night lights** added automatically when the scene is dusk or night
- **Density normalisation** so every theme lands near its target instances per
  1,000 m²

That last one fixed a real bug: desert scenes came out at 13 instances per
1,000 m² against a village's 78. The cause was subtle — when a prompt
mentioned "rock", filler *skipped* rocks entirely to avoid duplicating, and
rocks are the largest filler category in desert scenes. Filler now **tops up**
towards a target instead of standing aside. Spread went 5.9× → 1.6×.

## Layout

### Requirements, in priority order

1. Nothing overlaps
2. Nothing floats or sinks — every instance sits on the surface
3. The result is walkable — a clear corridor runs through

### Rejection sampling

For each instance: propose a position, test it, accept or retry. Up to 60
attempts, then give up and record it as skipped.

Tests applied to each proposal:
- inside the terrain, footprint included
- at least `PATH_HALF_WIDTH + radius` from the road
- slope below the category's limit (0.18 for buildings, 0.80 for rocks)
- no collision with anything already placed

### The spatial hash grid

Naive collision checking compares every proposal against every placed object —
**O(n²)**. At 2,000 objects with 60 attempts each that's tens of millions of
distance calculations, far too slow in Python.

A **spatial hash grid** buckets objects into 6 m cells. A proposal only
compares against the cells its footprint could reach. Placement stays under a
second for 2,000 objects.

**A real bug here, worth telling:** the search initially widened by only the
*querying* object's radius. A large object stored a few cells away could still
overlap the proposal but never be checked — so trees overlapped. The grid now
tracks the largest radius stored and widens by that too. A test builds 300
random objects and asserts the grid's answer matches brute force every time.

### Placement order

Big, important objects first — buildings, then structures, lights, props,
rocks, vegetation. Rejection sampling fills space greedily, so whatever goes
last gets the scraps. Buildings must claim their ground before 1,900 trees
crowd them out.

### Settlement structure

A village is not objects sprinkled evenly. Real settlements are dense in the
middle and thin into farmland.

- A **centre** is anchored on the path, so the road runs through the
  settlement rather than past it
- Buildings place within a core radius (30% of scene size for a village), with
  density falling off outward via `random()^0.62` — which concentrates towards
  the middle rather than forming an obvious ring
- Large vegetation is **excluded** from the core — no mature trees in a
  village square
- Buildings **face the road** when near it, otherwise face the centre

### Placement rules

`along_path`, `cluster`, `clump`, `scatter`, `perimeter`, `center`.

`clump` matters most for realism: nature grows in patches, not evenly. Uniform
scatter of 1,900 grass tufts reads as confetti; grouping them into 3–9 patches
reads as meadow. A test measures mean nearest-neighbour distance and asserts
it falls below the uniform expectation.

### Per-instance variation

Identical repeated meshes are the clearest giveaway of procedural placement.
Each instance gets randomised rotation and a scale jitter (±35% for
vegetation, ±8% for buildings — a house twice its neighbour's size looks
wrong, a tree doesn't).

## Scale normalisation

**Packs model at arbitrary scales.** A Kenney house is about 2 units wide. Used
raw, a village on a 120 m terrain becomes 2 m huts scattered across a field —
it reads as debris, not architecture.

Every asset is normalised to a plausible real-world size: houses 8 m,
windmills 12 m, trees 6.5 m, barrels 1 m. Both **height and width** are
capped, because a long flat model (a fallen log, a corridor section) can pass
a height check while being twenty metres across — which is exactly what
happened, filling a forest with enormous grey logs.

**A subtle bug worth telling:** the cap was applied *before* the random
jitter. An 8 m log at 1.35× jitter renders 10.8 m — precisely what we measured.
The clamp now applies after.

---

# PART 6 — STAGE 5: EXPORT AND RENDERING

## Why glTF

**glTF 2.0** is the standard interchange format for real-time 3D — often
called "the JPEG of 3D". One file carries geometry, materials and a scene
graph. Unity, Unreal, Blender and browsers all read it.

Exporting glTF rather than a Unity-specific format keeps the pipeline
**engine-agnostic**. The same file drives the web viewer and the Unity import.

## Two export modes

**Instanced (default).** Each distinct model is copied once into `assets/`,
and `scene_manifest.json` records every instance's position, rotation and
scale. That's how a game engine actually stores a level — one mesh referenced
many times. Preserves the packs' original materials.

**Baked (`--baked`).** Everything merged into one file with palette colours.
Simplest possible engine import: one file, one drag.

## Vertex colours and the palette

Source materials frequently don't survive being loaded and re-exported —
packs use texture atlases and per-part materials, and some arrive with
nothing. The symptom is an entire scene of black silhouettes.

So `--palette` assigns colours from a semantic palette keyed on what an object
*is*: foliage green, timber brown, stone grey, firelight warm.

**Two-tone shading** is the highest-value trick here. Each model is split by
height and the bands coloured separately — houses get walls plus a roof
(thatch for huts, tile for houses, slate for churches), trees get a brown
trunk under green foliage. No textures, no extra geometry. It is the
difference between "grey box" and "cottage".

## Rendering: what makes it look right

| Technique | What it does |
|---|---|
| ACES filmic tone mapping | maps high-dynamic-range light to screen; the standard for film-like rather than washed-out |
| Exponential-squared fog | `exp(−(density × distance)²)` — hides the terrain edge, adds depth |
| Soft shadow mapping | 2048×2048 shadow map, PCF filtering |
| Hemisphere light | sky colour from above, ground bounce from below |
| Gradient sky | a flat background reads as a screenshot; a gradient reads as sky |
| Horizon plane | without it the terrain is a floating island showing its unlit underside |
| Smooth vertex normals | interpolated normals stop curved surfaces showing facets |

## Two rendering bugs worth telling

**Fog scale.** `FogExp2` attenuates by `exp(−(density × distance)²)` — the
density interacts *quadratically* with scene size. A value tuned for a small
scene left a 120 m landscape **93% fogged**; the world was invisible. Fog is
now expressed relative to scene size. There's a test asserting opacity stays
under 55% at 60 m, 120 m and 200 m.

**Light colour tinting.** Sky colours are chosen to look right *as sky*, which
means saturated. Fed into the hemisphere light they tint every surface —
strongly blue light on green foliage renders **cyan**, and orange tents came
out pink. All three light sources are now desaturated. A test measures the
combined light across sun, sky and ground bounce and asserts the
blue-over-green bias stays under 0.15 at every time of day.

## The "poly count around edges" question

Your teacher asked this. The full answer:

**Polygon count.** ~300,000 triangles rendered for a large scene. Across the
library the median model is 140 triangles and the mean 572 — the mean skewed
by a few outliers, the heaviest a crops model at 35,076. Each distinct model
is stored once and instanced, so *stored* geometry is far smaller than
*rendered* geometry.

**Edges specifically.** Low-poly models show flat facets on curved surfaces.
The fix is not more triangles — it's **smooth vertex normals**. Normals are
per-vertex directions used for lighting; interpolating them across a face
makes the surface read as curved at identical triangle count. An early version
exported no NORMAL attribute at all and the ground rendered as harsh flat
panels until we fixed it.

**Level of detail (LOD)** — swapping distant objects for cheaper versions —
is the next optimisation, not yet implemented. We have headroom: modern GPUs
handle millions of triangles.

## Unity integration

Unity doesn't read `.glb` natively. Its official package **`com.unity.cloud.gltfast`**
does, and registers as the default importer, so a GLB in `Assets/` imports
like an FBX.

What makes the import clean, deliberately:

| Property | Why |
|---|---|
| No required glTF extensions | Draco/KTX/meshopt each need another package |
| Explicit non-metallic material | the glTF *default* is metallic 1.0, which renders as dark metal |
| Vertex normals present | otherwise flat shading |
| Metres, Y-up | matches Unity's convention |
| Relative asset paths | no paths from the generating machine |
| Valid container lengths | a mismatch makes importers reject the file |

**Lighting is exported separately** as `lighting.json`, because glTF has no
concept of fog, ambient light or exposure. A Unity editor script applies it —
sun, ambient, fog, and a warm point light at each campfire. Both the browser
viewer and the Unity script read the same Python function, so they cannot
drift apart, and a test asserts they agree.

---

# PART 7 — TESTING

## The numbers

259 tests, 2,444 lines — about 40% of the code base.

| File | Tests | Covers |
|---|---|---|
| `test_asset_stage.py` | 97 | classification, retrieval, resolution |
| `test_layout_stage.py` | 51 | terrain, placement, dressing, settlement |
| `test_appearance.py` | 44 | scale normalisation, palette, two-tone |
| `test_viewer_export.py` | 34 | fog, lighting, ground painting, viewer config |
| `test_prompt_stage.py` | 17 | parsing, schema validation |
| `test_unity_export.py` | 12 | glTF validity, Unity readiness, lighting data |
| `test_ingest.py` | 4 | manifest building, variant handling |

## The strategy

Tests run **without a GPU, without the language model, and without the asset
library**, using a synthetic mini-library built in code. They pass on any
machine in about 17 seconds.

**Roughly half are regression tests** — each encodes a bug we actually hit, so
the same mistake cannot silently return. That is the point worth making: the
test suite is a record of what went wrong, not a box-ticking exercise.

## Evaluation harness

`evaluate.py` runs 26 prompts — the four themes, awkward paraphrases
("somewhere a blacksmith would work"), and deliberately out-of-scope requests
("an underwater city", an empty string, pure keysmash).

| Metric | Result |
|---|---|
| Structurally valid | 26/26 |
| Prompt coverage | 100% |
| Retrieval confidence | 0.80 – 0.91 (median 0.87) |
| Density spread across themes | 1.6× |
| Generation time | 0.2–1.6 s without LLM, 8–11 s with |

`--weak` lists which queries retrieval handled badly and what they matched
instead — which doubles as a list of what the asset library is missing.

---

# PART 8 — MAPPED TO THE MARKING RUBRIC

## Understanding of Topic (5 marks)

Cover: the problem (environment authoring is slow and expensive); the
distinction between *scene* generation and *asset* generation; where AI sits
in the pipeline (language understanding and semantic retrieval) and where
procedural methods sit (terrain, placement); and why that division is the
right one.

**The sentence that shows depth:** "We generate the scene and retrieve the
assets. That's what the published systems do too — retrieval gives game-ready
geometry and a consistent art style, which generative 3D still doesn't."

## Research Quality (5 marks)

Three 2026 papers, read and positioned — not just cited:

**AutoUE: Automated Generation of 3D Games in Unreal Engine via Multi-Agent
Systems** (2026) — <https://arxiv.org/abs/2603.07106>
Multi-agent system coordinating retrieval over 858,000 models, Unreal's PCG
tool, gameplay code synthesis and automated play-testing. Closest to our goal.
We adapted its LLM-as-judge evaluation idea into our confidence scoring.

**Imagine a City: CityGenAgent for Procedural 3D City Generation** (2026) —
<https://arxiv.org/abs/2602.05362>
LLM-driven hierarchical city generation via interpretable "Block Program" and
"Building Program", trained with supervised fine-tuning plus reinforcement
learning. Informed our settlement-structure approach and editability thinking.

**ScenDi: 3D-to-2D Scene Diffusion Cascades for Urban Generation** (2026) —
<https://arxiv.org/abs/2601.15221>
3D latent diffusion producing Gaussians, enhanced by 2D video diffusion. Our
reference point for why pure-diffusion 3D is still hard to make
engine-ready.

**Also studied:** Tencent's **Hunyuan3D-WorldClaw**, whose own paper states it
requires Claude Opus 4.8, GPT-Image-2 and Hunyuan3D — a paid frontier language
model, a paid image model, and a large 3D backbone. Foundational background:
DreamFusion (ICLR 2023), Infinigen (CVPR 2023), SceneX (AAAI 2025),
DiffuScene (CVPR 2024).

**The key research finding:** every current system in this space *retrieves*
assets and *generates* scenes. That convergence justified our architecture.

## Methodology (5 marks)

- **Hybrid by design:** LLM for language understanding, embeddings for
  semantic matching, procedural generation for terrain and placement. Each
  technique used where it's strongest.
- **One data contract** between stages, enabling isolated development and
  testing.
- **Graceful degradation** at every stage — no single point of failure.
- **Measurement over inspection:** a 26-prompt evaluation harness reporting
  validity, coverage, confidence, density and timing.
- **Diagnostic tooling built when needed** — `inspect_library.py` and
  `inspect_scene.py` were both written to answer a specific question that
  screenshots couldn't.

**The methodology point that lands hardest:** we found most of our bugs by
measuring output quality, not by watching for crashes. The clearest example is
in Part 9.

## Originality and Creativity (5 marks)

**The novelty:** published systems generate environments optimised to *look*
coherent. None optimise for whether the space is interesting to *move through*.

Our contribution is **gameplay-flow scoring**: generate several candidate
layouts per prompt, send a simulated agent to explore each, measure path
variety, chokepoints and point-of-interest spacing, and keep the best. The
candidate-generation hook is already in `generate.py` (`--candidates`); the
metric suite is the next stage.

**Secondary originality:**
- Confidence-scored retrieval that reports its own failures rather than
  silently substituting
- Density normalisation so scene richness is consistent across themes
- Two-tone height-based shading, giving roofs and trunks at zero geometry cost
- Running the whole pipeline on one consumer GPU at zero cost, where
  comparable systems need frontier paid models

## Implementation — coding and testing (10 marks)

This is the heaviest single category. Lead with:

- 5,920 lines across 22 modules, clean stage separation
- 259 automated tests, 2,444 lines, running without GPU/LLM/assets
- ~half the tests are regressions encoding real bugs
- A measurement harness producing the quantitative table
- Three diagnostic tools written to answer specific questions
- Engine integration: glTF export plus two Unity editor scripts

**Show, don't assert:** run `pytest` live, run `evaluate.py --weak` live, and
run one generation live.

---

# PART 9 — HARD QUESTIONS, WORKED ANSWERS

**"Where exactly is the AI? This looks like a lot of if-statements."**

Two models. A 7-billion-parameter language model interprets the prompt and
infers unstated content — a village gets houses though the prompt never says
so. A 384-dimensional sentence-embedding model matches objects to assets by
meaning, so "blacksmith forge" finds "smithy" despite sharing no whole word.
The procedural parts — terrain, placement — are deliberately *not* AI, because
rule-based methods give guarantees a neural model can't: no overlaps, nothing
floating, a walkable path. Run `--fallback` to see the same pipeline with the
AI removed; it finds four object types where the model finds seven.

**"Why don't you generate the 3D models themselves?"**

We generate the *scene*: terrain, road network, settlement structure, and every
object's position, rotation and scale. Assets are retrieved. That's what the
published systems do too — AutoUE retrieves from 858,000 models. Retrieval
gives game-ready geometry and a coherent art style; generative 3D still
produces meshes with holes, bad topology and no collision hulls. Our confidence
scores already identify *when* the library can't serve a request, which is the
natural trigger for adding generation. TRELLIS is MIT-licensed, does
text-to-3D, and needs a 16 GB GPU — which our desktop has. That's the next
stage.

**"How do you know it works, rather than just runs?"**

Confidence scores and validation. The best example: the system once reported
complete success — every object resolved, no errors — while producing a village
made of *tents*, because the asset library contained no houses. Nothing
crashed. The scores caught it: 0.62 for "house" against 0.90 for genuine
matches. We built `inspect_library.py`, confirmed zero house models existed,
and fixed it in three parts. That's why we measure output quality rather than
absence of exceptions.

**"What was the hardest bug?"**

The spatial grid. To place 2,000 objects without overlaps, naive checking is
O(n²) — tens of millions of comparisons. We bucketed objects into 6 m cells,
which made it fast, but the search widened by only the *querying* object's
radius. A large object a few cells away could overlap and never be checked, so
trees overlapped. The fix tracks the largest radius stored and widens by that
too. There's now a test placing 300 random objects and asserting the grid's
answer matches brute force exactly.

**"Why 384-dimensional embeddings? Why that model?"**

all-MiniLM-L6-v2 is the standard baseline for semantic search — one of the
most downloaded models on Hugging Face, trained on over a billion sentence
pairs, Apache-2.0. It's 90 MB and runs on CPU in milliseconds, which matters
because our 8 GB of VRAM is already holding the language model. For two- and
three-word queries like "blacksmith forge", 384 dimensions is ample; larger
models give diminishing returns on short text.

**"Your terrain is just noise. How is that generation?"**

It's fractal value noise: random values on a coarse grid, smoothly
interpolated, summed across four or five octaves at doubling frequency and
halving amplitude. Each terrain type has tuned amplitude, octave count,
persistence and a ridged flag that folds the noise to make sharp crests. The
result is then shaped by the scene: building pads are levelled, and the road
corridor is graded flat so it reads as built. Amplitudes are deliberately
modest — dramatic terrain makes a village unplaceable, and playability matters
more than drama.

**"What happens with a prompt you don't support?"**

It's approximated from the nearest of four themes, and the system says so — a
console warning plus an "(approximated)" label in the viewer. We cross-check
the model's theme choice against keyword evidence; if the prompt contains no
word associated with any supported theme, the choice was a guess and we report
it. Silently returning a medieval village for "an underwater city" would be
worse than admitting the limit. Try it live.

**"What's your biggest limitation?"**

Placement is statistical, not relational. The prompt says "a campfire" —
singular, and the focal point a camp should organise around. The system treats
it as a density parameter and places 134 light sources. It has no notion that a
campfire is a focal point, that tents face it, or that "a" means one. Fixing
that means the specification must carry spatial *relations*, not just objects
and counts — which is what WorldClaw's planning stage produces and ours
doesn't. That's the next major piece of work, and it composes with the
gameplay-flow scoring.

**"Could this actually be used?"**

For greyboxing and prototyping, yes today — that's the stated scope. A designer
gets a populated, walkable, engine-ready environment in twenty seconds instead
of a day, then edits it. For shipping art, no: it depends on a curated asset
library and produces a stylistically uniform result. The honest framing is a
rapid-prototyping tool, not an art-replacement tool.

**"How is this different from just downloading an asset pack?"**

An asset pack gives you a thousand models in a folder. It doesn't tell you
which twelve of them make a village, how many of each, where they go, which
way they face, what the ground looks like underneath, where the road runs, or
what colour the light is at dusk. That's the entire problem we solve — the
models are an input, not the output.

---

# PART 10 — THINGS TO HAVE READY

**Run live, in this order:**

```powershell
python -m pytest tests/ -q                    # 259 passed
python scripts/generate.py "a foggy medieval village at dusk with a blacksmith forge"
python scripts/evaluate.py --fallback --weak  # the metrics table
python -m src.cli "<same prompt>" --fallback  # the AI-vs-keyword comparison
```

**Have open:** the browser viewer on the forest camp, Unity with a scene
imported, and this document.

**Know these five numbers cold:** 5,920 lines of code, 259 tests, 1,001
assets, 26/26 prompts valid, 0.80–0.91 confidence.

**The three things to make sure you say:**

1. The keyword-versus-model comparison — proves the AI is doing real work
2. The village of tents — a silent failure caught by measurement, not by an
   error
3. Placement is statistical, not relational — naming your own most interesting
   limitation before you're asked is the strongest move available
