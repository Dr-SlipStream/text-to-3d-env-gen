# Text-to-3D Game Environment Generation

Generate customizable 3D game environments from natural-language prompts.

> *"a foggy medieval village at dusk with a blacksmith forge and a broken stone bridge"*
> → a walkable, engine-importable 3D scene

## What this is

A hybrid generation pipeline that turns a sentence into a playable 3D environment:

1. **Prompt decomposition** — a locally-run LLM parses the prompt into a structured Scene Specification
2. **Asset retrieval** — objects are matched to 3D models from an indexed, license-free asset library
3. **Terrain + layout** — procedural generation places assets on generated terrain without overlaps
4. **Gameplay-flow scoring** *(our novelty)* — a simulated agent explores each candidate layout and scores it on path variety, chokepoints and point-of-interest spacing; the best-scoring layout wins
5. **Assembly + export** — lighting, collision and navmesh are added, then exported to Unity

### Novelty

Existing systems generate environments optimized for *visual* coherence. None optimize for whether the space is actually interesting to move through. We generate several candidate layouts per prompt and use agent-based playtest scoring to pick the one with the best gameplay flow.

## Cost

**Zero.** Everything in the stack is free and runs locally:

| Need | Tool | Cost |
|---|---|---|
| LLM | Ollama (local) | Free |
| Embeddings | sentence-transformers | Free |
| Vector search | ChromaDB | Free |
| 3D assets | CC0 asset packs | Free |
| Engine | Unity Personal | Free |

No API keys. No cloud compute. No per-call charges.

## Quick start

```bash
git clone <your-repo-url>
cd text-to-3d-env-gen

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.cli --check          # verify local LLM setup
python scripts/serve.py            # the app: open http://localhost:8000
```

### The app

`python scripts/serve.py` starts a local web app. Type a place, press
**Generate scene**, and watch it build: the language model reads the
sentence, several candidate layouts are generated, a bot playtests each
one, and the best is shown in 3D next to the bot's report — flow score,
walkability map with the route it learned, how fast it learned, and every
metric behind the score. Earlier scenes stay in the sidebar.

Settings go in the sentence, so there are no flags to remember:

| Say | Does |
|---|---|
| `best of 5` / `try 5 layouts` | generate 5 layouts, keep the one that plays best (default 3) |
| `variation 7` / `seed 7` | a fixed, reproducible variation |
| `small`, `large`, `200m` | map size |
| `dense`, `sparse` | how full the scene is |
| `for unity` | also write a Unity bundle |
| `quick draft` / `offline` | keyword parsing, no language model |
| `simple layout` | rule-based layout instead of the scene graph |
| `no bot` | skip the playtest |

`python scripts/serve.py --offline` runs without Ollama.

### The bot playtest

Every scene is played before you see it (`src/playtest.py`). A Q-learning
agent is trained on a walkability grid of the scene and must learn to cross
it from one end of the road to the other. Alongside it, the route is
measured:

| Metric | What it asks |
|---|---|
| Reachability | can every part of the open ground be reached? |
| Route directness | is the walk sensible, or a long detour? |
| Chokepoints | how many single-file pinches (a bridge is a good one) |
| Open ground | room to move, without nothing to navigate |
| Pacing | the longest stretch of the route with nothing to see |
| Learnability | how fast and reliably the bot learns the route |

These combine into a **flow score** out of 100. With `best of N`, the
pipeline playtests each candidate and keeps the highest — generation and
evaluation in one closed loop.

Full setup instructions, including installing Ollama: **[docs/SETUP.md](docs/SETUP.md)**
Building the 3D asset library: **[docs/ASSETS.md](docs/ASSETS.md)**

The pipeline works without the LLM too (keyword fallback), so you can develop and test on any machine:

```bash
python -m src.cli "a dense forest camp at night" --fallback
```

Once the asset library is built, generate a full 3D scene:

```bash
python scripts/generate.py "a foggy medieval village at dusk" --open
```

### Two layout paths

The default is rule-based: per-theme tables decide what a place contains and
where it goes.

`--graph` uses the relational path instead. The model describes the place as
regions and spatial relations -- *one campfire at the centre of the camp, six
tents ringed around it facing it, trees scattered through the woods avoiding
the camp* -- and a solver satisfies them. It carries no knowledge of camps or
villages, so new kinds of place need no new code.

```bash
python scripts/generate.py "a forest camp around a campfire beside a river" \
    --graph --show-graph --size large
```

`--show-graph` prints the structure before it is solved.

That writes `outputs/<name>/` containing `scene.glb` (open in Blender, Unity or
anything that reads glTF) and `index.html` (a real-time viewer with
time-of-day lighting, weather fog and shadows).

## Project layout

```
src/
  vocab.py                  controlled vocabulary — the single source of truth
  schema.py                 SceneSpec — the data contract between all stages
  llm_client.py             Ollama wrapper
  prompt_decomposition.py   stage 1: prompt → SceneSpec
  fallback_parser.py        keyword parser used when the LLM is unavailable
  asset_rules.py            filename cleaning + category classification
  asset_index.py            stage 2: embedding + vector search over assets
  asset_resolution.py       stage 2: SceneSpec → concrete 3D models
  terrain.py                stage 3: heightmap, height/slope queries, and
                            landscape features (rivers, lakes, hills, valleys)
  scene_graph.py            relational scene description: regions, groups,
                            and the spatial relations between them
  solver.py                 satisfies those relations -- no theme knowledge
  graph_builder.py          gets a graph from the model, or compiles one
                            from the flat spec as a fallback
  graph_pipeline.py         binds groups to real models and produces a
                            PlacedScene the exporter already understands
  graph_dressing.py         adds the things no description mentions:
                            ground cover, variation, consistent density
  crossings.py              bridges where the road meets water, and the
                            road choice that keeps crossings short
  rl/                       navigation MDP built from a scene; Q-learning,
                            SARSA and a random baseline that measure whether
                            a generated place can actually be crossed
  layout.py                 stage 3: object placement, collision avoidance
                            (the original rule-based path, still the default)
  dressing.py               stage 3: fills the world with background detail
  appearance.py             real-world scaling + colour palette
  export_gltf.py            stage 5: bake everything into one .glb (fallback)
  export_scene.py           stage 5: keep each asset's own materials (default)
  viewer.py                 stage 5: three.js viewer with lighting/fog/shadows
  cli.py                    command-line entry point
  pipeline.py               the closed loop: prompt → candidates →
                            playtest → best scene → export
  prompt_options.py         settings read from the sentence ("best of 5")
  playtest.py               the bot playtest and gameplay-flow metrics
web/app.html                the app's page (no build step)
scripts/
  serve.py                  the web app: prompt box, 3D view, bot report
  generate.py               the same pipeline from the command line
  evaluate.py               consistency metrics across a suite of prompts
  ingest_assets.py          scans downloaded packs → manifest.json
  inspect_library.py        what's actually in the library, and what isn't
tests/                      pytest suite (runs without a GPU or LLM)
data/asset_library/         3D assets (downloaded, not committed)
engine/unity/Editor/        Unity editor script that rebuilds a scene
                            from its manifest
docs/
  SETUP.md                  getting the project running
  ASSETS.md                 building the 3D asset library
  WALKTHROUGH.md            full explanation of what's built and why
  DEMO.md                   demo runbook: what to run, what to say
  UNITY.md                  importing a generated scene into Unity
  TECHNICAL_REFERENCE.md    full technical explanation, for exams and vivas
```

### The Scene Specification

Every stage reads and writes this one structure. It's the reason the stages can be built and tested independently.

```json
{
  "theme": "medieval_village",
  "art_style": "low_poly",
  "terrain": { "type": "grassland", "size": "medium" },
  "lighting": { "time_of_day": "dusk", "weather": "fog", "mood": "abandoned" },
  "objects": [
    { "name": "house", "category": "building", "placement": "along_path", "quantity": 7 }
  ]
}
```

To add a new theme, object type or mood, edit `src/vocab.py` — never hardcode strings elsewhere.

## Roadmap

| Week | Milestone | Status |
|---|---|---|
| 1 | Repo scaffold, scene schema, prompt decomposition | ✅ Done |
| 2 | Asset library curation + retrieval index | ✅ Done |
| 3 | Terrain generation + procedural layout engine | ✅ Done |
| 4 | Playtest agent + gameplay-flow scoring *(novelty)* | ⬜ |
| 5 | Unity integration: collision, navmesh, import | ⬜ |
| 6 | Lighting/atmosphere + scene export | ✅ Done (early) |
| 7 | Preview UI + iterative refinement | ⬜ |
| 8 | Evaluation metrics + final report | ⬜ |

## Measured behaviour

```bash
python scripts/evaluate.py --weak
python scripts/evaluate.py --suite breadth           # 16 places it wasn't built for
python scripts/evaluate.py --suite breadth --graph   # same, through the scene graph
```

The breadth suite (snowy villages, harbours, pirate coves, swamps, a volcanic
wasteland, a graveyard…) reports the ground and water it chose, whether the
road can be walked end to end (`conn`), how many bridge segments were laid,
and which requested things were left out because the library has nothing like
them.

26 prompts spanning all four themes, awkward paraphrases, and deliberately
out-of-scope requests:

| Metric | Result |
|---|---|
| Structurally valid | 26/26 |
| Prompt coverage | 100% |
| Retrieval confidence | 0.80 - 0.91 |
| Density spread across themes | 1.6x |

`--weak` lists the queries retrieval handled badly, which doubles as a list of
what the asset library is missing.

## Testing

```bash
python -m pytest tests/ -v
```

102 tests. They never touch the LLM, GPU or asset library, so they pass on any
machine. About half are regression tests encoding bugs we actually hit.
