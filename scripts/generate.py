"""
End to end: prompt in, viewable 3D scene out.

    python scripts/generate.py "a foggy medieval village at dusk"
    python scripts/generate.py "a forest camp at night" --seed 7 --open
    python scripts/generate.py "a desert outpost" --fallback --candidates 3

Writes to outputs/<slug>/:
    scene.glb     the 3D scene (open in Blender/Unity, or the viewer)
    index.html    a real-time viewer with lighting and atmosphere
    scene.json    the full record: spec, placements, validation
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.asset_index import AssetIndex                      # noqa: E402
from src.asset_resolution import resolve_scene              # noqa: E402
from src.dressing import dress_spec                         # noqa: E402
from src.export_gltf import export_glb                      # noqa: E402
from src.export_scene import export_instanced               # noqa: E402
from src.layout import layout_scene, validate               # noqa: E402
from src.llm_client import LLMClient                        # noqa: E402
from src.prompt_decomposition import decompose              # noqa: E402
from src.terrain import Terrain                             # noqa: E402
from src.viewer import write_viewer                         # noqa: E402


def slugify(text: str, limit: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (s[:limit].rstrip("-")) or "scene"


def main() -> int:
    p = argparse.ArgumentParser(description="Generate a 3D scene from text")
    p.add_argument("prompt")
    p.add_argument("--seed", type=int, default=None,
                   help="fix the random seed for reproducible output")
    p.add_argument("--fallback", action="store_true",
                   help="skip the LLM, use keyword parsing")
    p.add_argument("--model", help="override the Ollama model")
    p.add_argument("--out", type=Path, default=Path("outputs"))
    p.add_argument("--candidates", type=int, default=1,
                   help="generate N layouts and keep the most valid one")
    p.add_argument("--open", action="store_true",
                   help="open the viewer in a browser when done")
    p.add_argument("--no-terrain", action="store_true",
                   help="skip the ground mesh (smaller file, for debugging)")
    p.add_argument("--density", type=float, default=1.0,
                   help="scene detail multiplier (0 = prompt objects only)")
    p.add_argument("--size", choices=["small", "medium", "large"],
                   help="override the scene size the parser chose")
    p.add_argument("--palette", action="store_true",
                   help="recolour assets from our palette instead of using "
                        "their own materials -- gives one coherent look "
                        "across packs whose materials don't survive loading")
    p.add_argument("--baked", action="store_true",
                   help="merge everything into one .glb with flattened "
                        "colours, instead of keeping each asset's own "
                        "materials (smaller, but loses textures)")
    args = p.parse_args()

    seed = args.seed if args.seed is not None else abs(hash(args.prompt)) % 100000
    t0 = time.time()

    # -- 1. prompt -> spec ------------------------------------------------
    print(f'\n  Prompt: "{args.prompt}"')
    client = LLMClient(model=args.model) if args.model else None
    spec = decompose(args.prompt, client=client, force_fallback=args.fallback)
    print(f"  [1/5] Parsed ({spec.parser}): {spec.summary()}")
    for w in spec.warnings:
        print(f"        ! {w}")
    if not spec.theme_recognised:
        print("        ! THEME NOT RECOGNISED -- the scene will be built from "
              "the closest supported theme, not what was asked for.")

    # The parser picks a scene size from the prompt, which is right in
    # general but makes a demo unpredictable: the same sentence can yield a
    # 60m hamlet or a 200m town. This pins it when that matters.
    if args.size and args.size != spec.terrain.size:
        print(f"        size overridden: {spec.terrain.size} -> {args.size}")
        spec.terrain.size = args.size

    # -- 2. spec -> assets ------------------------------------------------
    try:
        index = AssetIndex.from_manifest()
    except FileNotFoundError as e:
        print(f"\n{e}")
        return 1

    # Fill the world with background detail the prompt never mentions but
    # every real place has. Without this a scene is a few models on a field.
    from src import vocab
    size_m = vocab.TERRAIN_SIZE_METRES[spec.terrain.size]
    if args.density > 0:
        spec, added = dress_spec(spec, size_m, density=args.density)
        print(f"        + {added} filler instances "
              f"(density {args.density:g})")

    resolved = resolve_scene(spec, index)
    print(f"  [2/5] Resolved: {resolved.summary()}")
    for w in resolved.warnings:
        print(f"        ! {w}")

    # -- 3. terrain -------------------------------------------------------
    terrain = Terrain.from_spec(spec, seed=seed)
    print(f"  [3/5] Terrain: {terrain.terrain_type}, {terrain.size:.0f}m, "
          f"height {terrain.heights.min():.1f}..{terrain.heights.max():.1f}m")

    # -- 4. layout --------------------------------------------------------
    # Generating several candidates and keeping the best is the hook the
    # gameplay-flow scoring plugs into next -- right now "best" just means
    # most instances successfully placed.
    best, best_report, best_terrain = None, None, None
    for i in range(max(1, args.candidates)):
        t = Terrain.from_spec(spec, seed=seed + i)
        cand = layout_scene(resolved, t, seed=seed + i)
        report = validate(cand)
        score = len(cand.instances) - cand.skipped
        if best is None or score > (len(best.instances) - best.skipped):
            best, best_report, best_terrain = cand, report, t

    placed = best
    terrain = best_terrain
    print(f"  [4/5] Placed: {placed.summary()}")
    if args.candidates > 1:
        print(f"        (best of {args.candidates} candidate layouts)")
    print(f"        valid={best_report['valid']} "
          f"overlaps={best_report['overlaps']} "
          f"floating={best_report['floating']} "
          f"blocking-path={best_report['on_path']}")
    for w in placed.warnings:
        print(f"        ! {w}")

    # -- 5. export --------------------------------------------------------
    out_dir = args.out / slugify(args.prompt)
    if args.baked:
        report = export_glb(placed, out_dir / "scene.glb",
                            include_terrain=not args.no_terrain)
        viewer = write_viewer(placed, out_dir, manifest_file="scene.glb",
                              title=args.prompt[:60])
    else:
        report = export_instanced(placed, out_dir, palette=args.palette)
        viewer = write_viewer(placed, out_dir, title=args.prompt[:60])

    (out_dir / "scene.json").write_text(json.dumps({
        "prompt": args.prompt,
        "seed": seed,
        "spec": spec.model_dump(),
        "scene": placed.to_dict(),
        "validation": {k: v for k, v in best_report.items() if k != "details"},
        "export": report,
    }, indent=2), encoding="utf-8")

    print(f"  [5/5] Exported {report['size_mb']} MB, "
          f"{report['total_triangles']:,} triangles rendered "
          f"({report['unique_meshes']} unique models across "
          f"{report['instances_exported']} instances)")
    if "unique_triangles" in report:
        print(f"        stored geometry: {report['unique_triangles']:,} "
              "triangles (each model kept once, with its own materials)")
    if report["instances_missing"]:
        print(f"        ! {report['instances_missing']} instance(s) had "
              f"unloadable meshes: {report['missing_assets']}")

    print(f"\n  Done in {time.time() - t0:.1f}s -> {out_dir}\n")
    print(f"  View it:  python -m http.server -d {out_dir} 8000")
    print(f"            then open http://localhost:8000\n")

    if args.open:
        webbrowser.open(viewer.resolve().as_uri())

    return 0


if __name__ == "__main__":
    sys.exit(main())
