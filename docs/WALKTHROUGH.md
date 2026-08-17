# Project Walkthrough — Weeks 1 & 2

## Text-to-3D Game Environment Generation

A complete account of what has been built, why each decision was made, and what
we learned. Written so it can be talked through end to end.

---

## 1. The one-sentence version

A user types *"a foggy medieval village at dusk with a blacksmith forge and a
broken stone bridge"*, and the system produces a 3D game environment. Two of
the five pipeline stages are complete and working.

---

## 2. The pipeline, and where we are in it

```
   TEXT PROMPT
        |
   [1] PROMPT DECOMPOSITION          <-- DONE (Week 1)
        |    a local language model reads the sentence and
        |    produces a structured Scene Specification
        v
   SCENE SPECIFICATION (JSON)
        |
   [2] ASSET RESOLUTION              <-- DONE (Week 2)
        |    each named object is matched to a real 3D model
        |    from a 940-model library, by meaning not spelling
        v
   RESOLVED SCENE (objects + real meshes + real dimensions)
        |
   [3] TERRAIN + LAYOUT              <-- NEXT (Week 3)
        |    generate ground, decide where every object stands
        v
   [4] GAMEPLAY-FLOW SCORING         <-- our novelty (Week 4)
        |    a bot explores several candidate layouts;
        |    the one that plays best is chosen
        v
   [5] ENGINE ASSEMBLY + EXPORT      <-- Week 5+
             lighting, collision, navmesh, Unity export
```

**Everything before stage 3 is finished, tested and running.**

---

## 3. Stage 1 — Prompt Decomposition

### What it does

Turns an ambiguous English sentence into a strict, machine-readable structure.

**Input:**
> "a foggy medieval village at dusk with a blacksmith forge and a broken stone bridge"

**Output:**
```json
{
  "theme": "medieval_village",
  "art_style": "low_poly",
  "terrain": { "type": "grassland", "size": "medium" },
  "lighting": { "time_of_day": "dusk", "weather": "fog", "mood": "mysterious" },
  "objects": [
    { "name": "house",  "category": "building",   "placement": "cluster", "quantity": 8 },
    { "name": "forge",  "category": "building",   "placement": "center",  "quantity": 1 },
    { "name": "bridge", "category": "structure",  "placement": "center",  "quantity": 1 },
    { "name": "tree",   "category": "vegetation", "placement": "scatter", "quantity": 12 }
  ]
}
```

### The key point to make

The model does **inference, not keyword extraction**. The prompt never mentions
houses or trees — but a village implies houses, and the model adds them. It
also reads "broken" and "foggy" together as a *mysterious* mood rather than
just literal fog.

To prove this, we built a deliberately dumb keyword parser as a comparison. Run
the same prompt through both:

| | LLM parser | Keyword parser |
|---|---|---|
| Objects found | 7 types, incl. houses, cottages, bushes | 4 types, only literal nouns |
| Mood | mysterious | abandoned (matched "broken") |
| Village has houses? | Yes | No |

That side-by-side is the clearest demonstration that the AI component is doing
real work.

### Why a *local* model

We run **Qwen2.5-7B through Ollama** on our own GPU rather than calling OpenAI
or Anthropic. Three reasons:

1. **Zero cost.** A commercial API would charge per prompt; this project has no
   budget. Nothing about the pipeline costs money to run.
2. **No API keys or internet dependency** during a live demo.
3. **Reproducibility.** The same model version stays pinned on our machine.

### The safety net

If Ollama isn't running, the keyword parser takes over automatically and the
pipeline still produces a valid scene. This was a deliberate design choice:
**a live demo cannot hard-fail.** Every stage has a fallback like this.

---

## 4. Stage 2 — Asset Resolution

### What it does

The specification says *"blacksmith forge"*. The asset library contains files
called `building_smithy.glb` and `shack.glb`. Exact string matching finds
neither. So we match by **meaning**.

Every asset name and every requested object is converted into a vector (a list
of numbers representing meaning) using a sentence-embedding model. Similar
meanings produce similar vectors, so we find the closest match by comparing
vectors rather than spelling.

**Current results on the demo prompt:**

| Requested | Matched asset | Score |
|---|---|---|
| house | house | 0.83 |
| shack | shack | 0.90 |
| bridge | bridge straight pillar | 0.77 |
| tree | tree | 0.90 |
| bush | plant bush | 0.89 |
| stone | cliff cave stone | 0.82 |
| forge | shack | 0.64 ⚠ |

Scores run 0 to 1. Above 0.70 is a confident match. The flagged `forge` is
**correct behaviour, not a bug** — no free asset pack contains a forge, so the
system substitutes the nearest sensible building and tells us it did. Silently
pretending would be worse.

### The asset library

940 models, from two sources, both **CC0 (public domain)** — free for any use,
commercial included, with no attribution required. This means no licensing
section is needed in the final report.

| Source | Provides |
|---|---|
| Kenney.nl (6 packs) | nature, survival, graveyard, space, town, castle |
| Quaternius (Fantasy RTS) | complete medieval buildings |

Categories: 106 buildings, 294 props, 139 vegetation, 131 rocks, 258
structures, 12 light sources.

---

## 5. What each file in the repository does

```
src/
  vocab.py                  the controlled vocabulary — every theme, mood,
                            object category and placement rule the system
                            understands. Single source of truth: adding a new
                            theme means editing this file only.

  schema.py                 the Scene Specification data structure, with
                            validation. Anything the LLM returns is checked
                            and corrected here before it reaches the rest of
                            the pipeline.

  llm_client.py             talks to the local Ollama server.

  prompt_decomposition.py   Stage 1. Builds the LLM instruction, parses the
                            reply, validates it, falls back if anything fails.

  fallback_parser.py        the keyword-only parser. Never fails. Used when
                            the LLM is unavailable, and as the comparison
                            baseline that demonstrates what the LLM adds.

  asset_rules.py            turns messy model filenames into clean searchable
                            names, and sorts each model into a category.

  asset_index.py            Stage 2 search. Converts names to vectors and
                            finds closest matches.

  asset_resolution.py       Stage 2 logic. Binds every object in the spec to
                            a real mesh, handles missing categories, prevents
                            two different objects getting the same model.

  cli.py                    command-line interface for running the pipeline.

scripts/
  ingest_assets.py          scans downloaded asset packs and builds the
                            manifest the search index reads.

  inspect_library.py        diagnostic tool: "what is actually in my library?"
                            Built specifically to debug a bad result (see §6).

tests/                      102 automated tests. Run without a GPU, without
                            the LLM, and without the asset library, so they
                            pass on any machine.

docs/
  SETUP.md                  how to get the project running from scratch.
  ASSETS.md                 how to build the asset library, and why certain
                            packs work and others don't.
```

### The design principle worth highlighting

Every stage communicates through **one shared data structure** — the Scene
Specification. Stage 1 writes it, stage 2 reads and extends it, stage 3 will do
the same. This means each stage can be built and tested in isolation, and a
change to one stage cannot silently break another.

---

## 6. What went wrong, and what we learned

This section matters more than the working code. Four real bugs, each found by
inspecting output rather than by the program crashing.

### Bug 1 — Non-deterministic results

**Symptom:** the same prompt returned different assets on different runs.

**Cause:** we used Python's built-in `hash()` function, which is randomised
per process for security reasons.

**Why it mattered:** results were irreproducible, and a demo could behave
differently from a rehearsal. Reproducibility is a basic requirement of any
system being evaluated.

**Fix:** switched to CRC32, a stable hash. Added a test that runs the same
query through two separate index instances and asserts identical results.

### Bug 2 — A village made of tents

**Symptom:** the pipeline reported success — 7/7 objects resolved, no errors —
but `house` had resolved to a **tent** and `forge` to a **chimney base**.

**Cause:** the asset library contained **zero house models**. Kenney's Castle
Kit and Fantasy Town Kit are *modular* — they provide wall segments, roof
corners and doorways for an artist to assemble by hand, not finished buildings.

**Why it mattered:** this was a **silent failure**. Nothing errored. Only the
confidence scores hinted at it (0.62 for house, versus 0.90 for tree).

**Fixes, in three parts:**
1. Built `inspect_library.py` to answer "does a house exist at all?" — it
   didn't.
2. Added a *modular fragment* detector, so roof corners and wall segments rank
   below complete models during search.
3. Added a source of complete buildings (Quaternius Fantasy RTS).

Result: `house` went from 0.62 (a tent) to 0.83 (an actual house).

**The general lesson:** retrieval quality is capped by library contents. No
amount of algorithm tuning invents a model that isn't there.

### Bug 3 — Keywords matching as prefixes

**Symptom:** cliff faces were filed as *buildings*; roof pieces as
*vegetation*.

**Cause:** our category-matching pattern had a word boundary at the start but
not the end, so `inn` matched "**inn**er", `corn` matched "**corn**er", and
`keep` matched "**keep**er".

**Fix:** required a boundary at both ends, while still allowing plurals
("trees" must still match "tree"). Affected roughly 30 assets.

### Bug 4 — Placing smoke

**Symptom:** the LLM listed "smoke" as a scene object, and the system dutifully
found the closest mesh — a fire basket.

**Cause:** smoke is atmosphere, not geometry. There is no correct model for it.

**Fix:** a list of non-physical concepts (smoke, fog, mist, shadow, wind) that
get dropped with a warning, plus an explicit instruction to the LLM not to
produce them. Care taken not to over-filter: "street lamp" and "lighthouse"
still pass through.

### Bug 5 — Losing model variety

**Symptom:** after cleaning random ID suffixes from filenames, the asset count
dropped by 17.

**Cause:** poly.pizza names files like `House_k6tP5nFuD2.glb`. Stripping the
random ID made several distinct houses all become "house", so they were merged
into one entry.

**Why it mattered:** every house in a generated village would have been the
identical mesh.

**Fix:** distinct models keep separate identities while sharing a searchable
name, so search matches them all equally and the system can vary which mesh it
uses.

---

## 7. Testing

102 automated tests, all passing. Deliberately designed to run **without** a
GPU, the LLM, or the downloaded asset library — using a synthetic mini-library
built in code. This means tests pass on any machine, including a teacher's.

Roughly half are **regression tests**: each one encodes a bug we actually hit,
so the same mistake cannot silently return.

```
python -m pytest tests/ -v
102 passed
```

---

## 8. Answers to likely questions

**"Is the AI actually doing anything, or is this just keyword matching?"**
Two AI components. First, a 7-billion-parameter language model interprets the
prompt and infers unstated content — a village gets houses even though the
prompt never says "houses". Second, a sentence-embedding model matches
requested objects to assets by meaning: "blacksmith forge" finds "smithy"
despite sharing no whole word. We keep a keyword-only parser specifically as a
baseline to show the difference.

**"Why not train your own model?"**
We evaluated this seriously and documented it. Training a 3D generative model
from scratch requires datasets of millions of models and far more compute than
one consumer GPU provides in a semester — even the 2026 papers in our
literature review orchestrate or fine-tune existing models rather than pretrain
them. The approved approach uses pretrained components and puts our original
contribution in the gameplay-flow scoring stage.

**"What is actually novel here?"**
Existing systems generate environments optimised to *look* coherent. None
optimise for whether the space is interesting to *move through*. In Week 4 we
generate several candidate layouts per prompt, send a simulated agent to
explore each one, and score them on path variety, chokepoints and
point-of-interest spacing — then keep the best. That scoring step is the
contribution.

**"What happens if something fails during the demo?"**
Every stage degrades instead of crashing. No LLM: keyword parser. No embedding
model: character matching. Missing asset category: substitution with a warning.
The pipeline always produces a scene.

**"How do you know it's working, rather than just running?"**
Confidence scores. Bug 2 is the example: the system reported complete success
while producing a village of tents. The scores caught it — 0.62 versus the 0.90
of genuine matches. Measuring output quality, not just checking for errors, is
how we found three of our five bugs.

---

## 9. Where we go next

| Week | Milestone |
|---|---|
| 3 | Terrain generation and object placement — objects get real positions, no overlaps, nothing floating |
| 4 | Gameplay-flow scoring — the novelty: a bot playtests candidate layouts |
| 5 | Unity integration — collision, navmesh, a walkable scene |
| 6 | Lighting and atmosphere, scene export |
| 7 | Preview interface and iterative editing |
| 8 | Evaluation metrics and final report |

**Immediate next step:** the resolved scene currently knows *what* objects exist
and how big each one is, but not *where* any of them go. Week 3 gives every
object a position.
