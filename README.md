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

The pipeline works without the LLM too (keyword fallback), so you can develop and test on any machine:

```bash
python -m src.cli "a dense forest camp at night" --fallback
```

## Project layout

```
src/
  vocab.py                  controlled vocabulary — the single source of truth
  schema.py                 SceneSpec — the data contract between all stages
  llm_client.py             Ollama wrapper
  prompt_decomposition.py   stage 1: prompt → SceneSpec
  fallback_parser.py        keyword parser used when the LLM is unavailable
  cli.py                    command-line entry point
tests/                      pytest suite (runs without a GPU or LLM)
data/asset_library/         3D assets (downloaded, not committed)
engine/unity_project/       Unity scene assembly and export
docs/                       setup guide, architecture notes, results
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
| 2 | Asset library curation + retrieval index | ⬜ |
| 3 | Terrain generation + procedural layout engine | ⬜ |
| 4 | Playtest agent + gameplay-flow scoring *(novelty)* | ⬜ |
| 5 | Unity integration: collision, navmesh, import | ⬜ |
| 6 | Lighting/atmosphere + scene export | ⬜ |
| 7 | Preview UI + iterative refinement | ⬜ |
| 8 | Evaluation metrics + final report | ⬜ |

## Testing

```bash
python -m pytest tests/ -v
```

Tests never touch the LLM, so they pass on any machine.
