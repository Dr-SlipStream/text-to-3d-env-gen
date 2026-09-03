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
python -m src.cli "a foggy medieval village at dusk"
```

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
  terrain.py                stage 3: heightmap generation + height/slope queries
  layout.py                 stage 3: object placement, collision avoidance
  dressing.py               stage 3: fills the world with background detail
  appearance.py             real-world scaling + colour palette
  export_gltf.py            stage 5: bake everything into one .glb (fallback)
  export_scene.py           stage 5: keep each asset's own materials (default)
  viewer.py                 stage 5: three.js viewer with lighting/fog/shadows
  cli.py                    command-line entry point
scripts/
  generate.py               end to end: prompt → viewable 3D scene
  evaluate.py               consistency metrics across a suite of prompts
  ingest_assets.py          scans downloaded packs → manifest.json
  inspect_library.py        what's actually in the library, and what isn't
tests/                      pytest suite (runs without a GPU or LLM)
data/asset_library/         3D assets (downloaded, not committed)
engine/unity_project/       Unity scene assembly and export
docs/
  SETUP.md                  getting the project running
  ASSETS.md                 building the 3D asset library
  WALKTHROUGH.md            full explanation of what's built and why
  DEMO.md                   demo runbook: what to run, what to say
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
```

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
