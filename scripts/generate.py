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
import shutil
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
from src.viewer import lighting_config, write_viewer        # noqa: E402


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
    p.add_argument("--manifest", type=Path,
                   help="use a specific asset manifest instead of the "
                        "project's own (tests use this so they never touch "
                        "the real library)")
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
    p.add_argument("--unity", action="store_true",
                   help="also write a Unity-ready folder: a single baked "
                        ".glb plus the editor importer script")
    p.add_argument("--unity-project", type=Path,
                   help="path to a Unity project; the baked scene is copied "
                        "straight into its Assets folder, so no drag-and-drop "
                        "is needed")
    p.add_argument("--unity-full", action="store_true",
                   help="also copy the per-model assets and manifest, for the "
                        "editor importer. Off by default: a handful of source "
                        "models fail Unity's glTF importer, and the baked "
                        "scene needs none of them")
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
        index = (AssetIndex.from_manifest(args.manifest) if args.manifest
                 else AssetIndex.from_manifest())
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

    # glTF cannot carry fog, ambient light or exposure, so the lighting is
    # written separately for any engine that wants to reproduce the preview.
    (out_dir / "lighting.json").write_text(
        json.dumps(lighting_config(placed), indent=2), encoding="utf-8")

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

    # -- optional Unity bundle -------------------------------------------
    if args.unity or args.unity_project:
        unity_dir = out_dir / "unity"
        unity_dir.mkdir(parents=True, exist_ok=True)

        # A single baked file is the simplest possible engine import: drag it
        # in and it appears. The instanced manifest alongside it is the
        # engine-native route, rebuilt by the editor script.
        baked = export_glb(placed, unity_dir / "scene.glb",
                           include_terrain=not args.no_terrain)

        repo_root = Path(__file__).resolve().parents[1]
        editor_src = repo_root / "engine" / "unity" / "Editor"
        importer = editor_src / "GeneratedSceneImporter.cs"
        lighting_script = editor_src / "SceneLightingApplier.cs"

        editor_dir = unity_dir / "Editor"
        editor_dir.mkdir(exist_ok=True)
        for script in (importer, lighting_script):
            if script.exists():
                shutil.copyfile(script, editor_dir / script.name)

        shutil.copyfile(out_dir / "lighting.json", unity_dir / "lighting.json")

        (unity_dir / "README.txt").write_text(
            "Unity import\n"
            "============\n\n"
            "1. Install glTFast: Window > Package Manager > + > Add package\n"
            "   by name > com.unity.cloud.gltfast\n"
            "   It registers as the default importer for .glb, so models\n"
            "   import automatically once installed.\n\n"
            "2. Simplest route -- drag scene.glb into your Assets folder.\n"
            "   The whole environment appears as one prefab.\n\n"
            "3. Engine-native route -- copy this entire scene folder into\n"
            "   Assets/, then run Tools > Generated Scene > Import from\n"
            "   manifest... and pick scene_manifest.json. This rebuilds the\n"
            "   scene as separate objects sharing meshes, the way a level is\n"
            "   normally authored.\n\n"
            "4. Lighting -- glTF carries geometry only, no fog or ambient.\n"
            "   Run Tools > Generated Scene > Apply Lighting... and pick\n"
            "   lighting.json to reproduce what the browser preview shows.\n\n"
            f"Baked file: {baked['size_mb']} MB, "
            f"{baked['total_triangles']:,} triangles\n",
            encoding="utf-8")

        print(f"        Unity bundle: {unity_dir} "
              f"({baked['size_mb']} MB baked scene + importer script)")

        # Copying into the project directly avoids the commonest import
        # failure: dragging a file from outside the project, which Unity
        # rejects with "Invalid AssetDatabase path". Unity only imports what
        # physically lives under Assets/.
        if args.unity_project:
            project = args.unity_project.expanduser().resolve()
            assets = project / "Assets"
            if not assets.is_dir():
                print(f"        ! {project} is not a Unity project "
                      "(no Assets folder found)")
            else:
                target = assets / "GeneratedScenes" / slugify(args.prompt)
                target.mkdir(parents=True, exist_ok=True)

                # The baked scene is all the drag-and-drop route needs, and
                # it is one file that either imports or doesn't.
                shutil.copyfile(unity_dir / "scene.glb", target / "scene.glb")
                shutil.copyfile(out_dir / "lighting.json",
                                target / "lighting.json")

                # The lighting script is always useful -- glTF carries no
                # lighting at all, so without it an imported night scene comes
                # in lit like an overcast afternoon.
                editor_target = assets / "Editor"
                editor_target.mkdir(exist_ok=True)
                if lighting_script.exists():
                    shutil.copyfile(lighting_script,
                                    editor_target / lighting_script.name)

                # The per-model files are only needed by the editor importer.
                # They are copies of the original pack files, and a few of
                # those fail Unity's glTF importer for reasons of their own --
                # copying them in by default fills the console with errors
                # about models the demo never uses.
                if args.unity_full:
                    for item in ("terrain.glb", "scene_manifest.json"):
                        src = out_dir / item
                        if src.exists():
                            shutil.copyfile(src, target / item)
                    if (out_dir / "assets").is_dir():
                        shutil.copytree(out_dir / "assets", target / "assets",
                                        dirs_exist_ok=True)

                # The editor script belongs in one shared Editor folder, not
                # duplicated per scene.
                if args.unity_full and importer.exists():
                    shutil.copyfile(importer, editor_target / importer.name)

                rel = target.relative_to(project)
                what = ("scene + per-model assets" if args.unity_full
                        else "scene.glb only")
                print(f"        Copied into Unity project: {rel} ({what})")
                print("        Switch to Unity -- it imports automatically.")

    print(f"\n  Done in {time.time() - t0:.1f}s -> {out_dir}\n")
    print(f"  View it:  python -m http.server -d {out_dir} 8000")
    print(f"            then open http://localhost:8000\n")

    if args.open:
        webbrowser.open(viewer.resolve().as_uri())

    return 0


if __name__ == "__main__":
    sys.exit(main())
