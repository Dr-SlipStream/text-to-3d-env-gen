"""
End to end from the command line: prompt in, playtested 3D scene out.

    python scripts/generate.py "a foggy medieval village beside a river"
    python scripts/generate.py "a desert outpost at night, best of 5, for unity"
    python scripts/generate.py "a forest camp, variation 7, quick draft"

Settings go in the sentence (see src/prompt_options.py): "best of 5",
"variation 7", "for unity", "large", "dense", "quick draft", "simple
layout", "no bot". The flags below still work and override the sentence.

The web app (python scripts/serve.py) runs exactly the same pipeline.

Writes to outputs/<slug>/:
    index.html      real-time viewer
    scene_manifest.json + assets/ + terrain.glb   the scene
    scene.json      spec, scene graph, placements, validation
    playtest.json   the bot's scores, route and learning curve
    summary.json    one-page summary of the run
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm_client import LLMClient                        # noqa: E402
from src.pipeline import GenerationError, run               # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Generate a 3D scene from text")
    p.add_argument("prompt")
    p.add_argument("--seed", type=int)
    p.add_argument("--fallback", action="store_true",
                   help="skip the LLM, use keyword parsing")
    p.add_argument("--model", help="override the Ollama model")
    p.add_argument("--out", type=Path, default=Path("outputs"))
    p.add_argument("--manifest", type=Path,
                   help="use a specific asset manifest (tests use this so "
                        "they never touch the real library)")
    p.add_argument("--graph", action="store_true",
                   help="scene-graph layout (the default)")
    p.add_argument("--rules", action="store_true",
                   help="rule-based layout instead of the scene graph")
    p.add_argument("--show-graph", action="store_true")
    p.add_argument("--candidates", type=int,
                   help="layouts to generate and playtest (default 3)")
    p.add_argument("--no-playtest", action="store_true",
                   help="skip the bot playtest")
    p.add_argument("--open", action="store_true")
    p.add_argument("--no-terrain", action="store_true")
    p.add_argument("--density", type=float)
    p.add_argument("--size", choices=["small", "medium", "large"])
    p.add_argument("--unity", action="store_true")
    p.add_argument("--unity-project", type=Path)
    p.add_argument("--unity-full", action="store_true")
    p.add_argument("--palette", action="store_true")
    p.add_argument("--baked", action="store_true")
    args = p.parse_args()

    overrides = {
        "seed": args.seed,
        "size": args.size,
        "density": args.density,
        "candidates": args.candidates,
        "offline": True if args.fallback else None,
        "mode": "rules" if args.rules else ("graph" if args.graph else None),
        "playtest": False if args.no_playtest else None,
        "unity": True if args.unity else None,
        "palette": True if args.palette else None,
        "baked": True if args.baked else None,
    }
    client = LLMClient(model=args.model) if args.model else None

    try:
        result = run(args.prompt, manifest=args.manifest, out_root=args.out,
                     overrides=overrides, client=client,
                     log=lambda m: print("  " + m),
                     unity_project=args.unity_project,
                     unity_full=args.unity_full,
                     include_terrain=not args.no_terrain,
                     show_graph=args.show_graph)
    except GenerationError as e:
        print(f"\n{e}")
        return 1

    for w in result["warnings"]:
        print(f"  ! {w}")
    v = result["validation"]
    print(f"\n  valid={v['valid']} connected={v.get('connected')} "
          f"bridges={result['bridges']} objects={result['instances']}")
    if "playtest" in result:
        pt = result["playtest"]
        print(f"  flow score {pt['flow_score']}/100 (grade {pt['grade']})")
    out_dir = Path(result["out_dir"])
    print(f"\n  View it:  python -m http.server -d {out_dir} 8000")
    print("            or run the app:  python scripts/serve.py\n")
    if args.open:
        webbrowser.open((out_dir / "index.html").resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
