# Demo Runbook

Everything needed to run and explain the demo, in the order you'll say it.

---

## Before the day

### Generate scenes on the desktop, demo from the laptop

The viewer is **static files** — HTML, glTF, JSON. Displaying a scene needs no
GPU compute at all; only *generating* one does. So:

1. On the desktop (RTX 5080), generate the demo scenes at high density.
2. Copy the whole `outputs/` folder to the laptop.
3. Demo from the laptop with pre-generated scenes, and generate **one** live to
   show it's real.

This removes the two things that can go wrong on the day: a slow model load and
an unlucky random seed.

```powershell
# on the desktop
python scripts/generate.py "a foggy medieval village at dusk with a blacksmith forge" --seed 42 --density 1.2
python scripts/generate.py "a medieval village on a sunny day" --seed 7
python scripts/generate.py "a dense forest camp at night with a campfire" --seed 3
python scripts/generate.py "a small desert outpost in a sandstorm" --seed 11
python scripts/generate.py "a sci-fi base on an alien planet" --seed 5

# for the before/after
python scripts/generate.py "a medieval village on a sunny day" --seed 7 --density 0 --out outputs_bare
```

### On the laptop, day of

```powershell
cd text-to-3d-env-gen
.venv\Scripts\Activate.ps1
python -m src.cli --check         # confirm Ollama is up
python -m pytest tests/ -q        # 192 passed
python -m http.server -d outputs 8000
```

Open `http://localhost:8000` and leave it on a tab. Have a second terminal
ready for the live generation.

---

## The demo, in five minutes

### 1. The problem (20 seconds)

Building a 3D game environment means blocking out terrain, sourcing assets,
placing hundreds of objects, lighting it, then testing that it's walkable.
That's days of work for a prototype you might throw away. Small teams and
students can't afford it.

### 2. Show the result first (30 seconds)

Open a pre-generated scene. Orbit once, press **W**, walk down the road.

> "This was generated from one sentence. Eight hundred objects, no manual
> placement, no 3D modelling."

### 3. Then show how (2 minutes) — run it live

```powershell
python scripts/generate.py "a foggy medieval village at dusk with a blacksmith forge"
```

Talk through the five stages as they print:

**[1/5] Prompt decomposition.** A language model running locally on our own GPU
reads the sentence and produces a structured scene specification. Point out
that it inferred houses, cottages and trees — *the prompt never mentions them*.
That's the model understanding what a village implies, not keyword matching.

**[2/5] Asset resolution.** Each object is matched to a real 3D model out of a
thousand-model library, by meaning rather than spelling — "blacksmith forge"
finds a smithy despite sharing no whole word. The confidence scores are shown,
and anything weak is flagged rather than hidden.

**[3/5] Terrain.** A heightmap generated to suit the terrain type, then graded
flat along the road.

**[4/5] Placement.** Every object gets a position. Point at the validation
line: zero overlaps, zero floating objects, nothing blocking the path. That's
checked programmatically, not by eye.

**[5/5] Export.** One glTF scene plus a viewer. glTF is the standard
interchange format, so the same file opens in Unity, Unreal or Blender.

Then open the result.

### 4. The before/after (30 seconds)

Show `--density 0` beside the normal output.

> "The language model lists what the prompt *mentions*. A believable place is
> mostly the things nobody thinks to say — grass, pebbles, fence posts,
> lanterns at dusk. That's thirty-five objects versus eight hundred."

### 5. What's novel (45 seconds)

> "Published systems in this space — including Tencent's WorldClaw this year —
> generate environments optimised to *look* coherent. None optimise for whether
> the space is interesting to move through. We generate several candidate
> layouts per prompt and score them with a simulated agent that explores each
> one, measuring path variety, chokepoints and how well points of interest are
> spaced. The best-scoring layout wins. That scoring step is our contribution,
> and it's what the research paper is about."

### 6. Honesty about the gap (20 seconds)

If asked how it compares to WorldClaw:

> "Their paper states the pipeline requires Claude Opus 4.8, GPT-Image-2 and
> Hunyuan3D — a paid frontier language model, a paid image model, and a large
> 3D generation backbone. We can't match that fidelity. But the architecture is
> the same shape: planning agent, structured specification, terrain foundation,
> asset placement. We built it to run on one consumer GPU at zero cost."

---

## Questions you should expect

**"Is the AI really doing anything, or is this keyword matching?"**

Two models. A 7-billion-parameter language model interprets the prompt and
infers unstated content. A sentence-embedding model matches objects to assets
by meaning. We deliberately kept a keyword-only parser as a baseline — run
`--fallback` and compare. The keyword version finds four object types where the
LLM finds seven, and it has to be told a village needs vegetation.

**"What's the polygon count?"**

Around 150,000 triangles rendered for a typical scene. Across the 1001-model
library the median model is 140 triangles and the mean 572 -- the mean is
skewed by a handful of outliers, the heaviest being a crops model at 35,076.
Each distinct model is stored once and instanced, so stored geometry is far
smaller than rendered geometry. Modern GPUs handle millions of triangles, so
there is plenty of headroom.

On edges specifically: smooth vertex normals are what stop curved surfaces
showing facets. That is a shading choice and costs no extra triangles -- an
early version exported no normals at all, and the ground rendered as harsh
flat panels until we fixed it.

**"What are the system's limits?"**

Three, and we measure all of them.

Four themes are supported; anything else is approximated from the closest one
and labelled as such. Retrieval can only return what the library holds -- we
have no well, no forge and only twelve light fixtures across a thousand
models, and the system reports those as low-confidence matches rather than
hiding them. And we retrieve assets rather than generating them, which is what
the published systems do too, because generative 3D still produces geometry
that is not game-ready and does not match a library's art style.

**"Why not train your own model?"**

We evaluated it and wrote it up. Training a 3D generative model from scratch
needs datasets of millions of models and far more compute than one consumer
GPU offers in a semester. Even the 2026 papers in our review orchestrate or
fine-tune existing models rather than pretraining. Our original contribution is
in the gameplay-flow scoring instead.

**"What happens if something breaks in the demo?"**

Every stage degrades rather than crashes. No language model: keyword parser. No
embedding model: character matching. Missing asset category: substitution with
a warning. The pipeline always produces a scene.

**"How do you know it's working, not just running?"**

Confidence scores and validation. Our best example: the system once reported
complete success — every object resolved, no errors — while producing a village
made of tents, because the asset library contained no houses. The scores caught
it: 0.62 for "house" against 0.90 for genuine matches. Measuring output quality
rather than checking for exceptions is how we found most of our bugs.

**"Does it only work for that one prompt?"**

No, and we measure it rather than assert it. `python scripts/evaluate.py`
runs 26 prompts across all four themes plus deliberately out-of-scope ones,
and reports structural validity, prompt coverage, retrieval confidence, scene
density and timing. Current results across 26 prompts:

| Metric | Result |
|---|---|
| Structurally valid (no overlaps, floating or blocked paths) | 26/26 |
| Prompt coverage | 100% |
| Retrieval confidence | 0.80 - 0.91 (median 0.87) |
| Scene density spread across themes | 1.6x |
| Generation time | 0.2 - 1.6s without the LLM, 8-11s with it |

**"What if I ask for something you don't support?"**

We support four themes. Anything else is approximated from the closest one --
but the system says so. It prints a warning and labels the theme
"(approximated)" in the viewer. Silently returning a medieval village for
"an underwater city" would be worse than admitting the limit. Try it live:
ask for something outside the four and show the warning.

**"Does it run in a game engine?"**

We export glTF, the standard interchange format, rather than to any one engine.
The same file imports into Unity, Unreal or Blender. We use a WebGL viewer for
the demo because it needs no install and runs anywhere.

---

## If something goes wrong

| Symptom | Fix |
|---|---|
| `--check` says NOT REACHABLE | `ollama serve` in another terminal, or just add `--fallback` |
| Generation is slow | Use a pre-generated scene; mention the model is loading into VRAM |
| Viewer shows "Could not load" | It needs HTTP, not `file://`. Use `python -m http.server` |
| A scene looks sparse | Wrong folder — check you're serving `outputs`, not `outputs_bare` |

Worst case: everything is pre-generated and static. Open the folder and present
from that.
