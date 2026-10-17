"""
The whole system as one call: prompt in, playtested scene out.

    result = run("a foggy village beside a river, best of 3")

Both front ends use this -- the command line (scripts/generate.py) and the
web app (scripts/serve.py) -- so they cannot drift apart.

The loop it runs:

    1. read settings out of the prompt          prompt_options
    2. prompt -> SceneSpec                      language model (or keywords)
    3. SceneSpec -> scene graph                 language model (or rules)
    4. N candidate layouts                      solver, one seed each
    5. quick playtest of every candidate        reachability, pacing, ...
    6. full bot playtest of the winner          Q-learning on the scene
    7. export: glTF, viewer, scene.json, playtest.json, Unity bundle

Steps 4-6 are the closed loop: the generator proposes, the bot plays, and
the scene that plays best is the one that ships.
"""

from __future__ import annotations

import json
import re
import shutil
import time
import zlib
from pathlib import Path
from typing import Callable, Dict, Optional

from . import vocab
from .asset_index import AssetIndex
from .asset_resolution import resolve_scene
from .crossings import add_bridge_instances
from .dressing import dress_spec
from .export_gltf import export_glb
from .export_scene import export_instanced
from .layout import layout_scene, validate
from .playtest import playtest
from .prompt_decomposition import decompose
from .prompt_options import Options, parse as parse_options
from .terrain import Terrain
from .viewer import lighting_config, write_viewer

REPO = Path(__file__).resolve().parents[1]

# Loading the library and its embedding model takes seconds; the web server
# keeps one per manifest for its whole life.
_INDEX_CACHE: Dict[str, AssetIndex] = {}


def load_index(manifest: Optional[Path] = None) -> AssetIndex:
    key = str(manifest or "")
    if key not in _INDEX_CACHE:
        _INDEX_CACHE[key] = (AssetIndex.from_manifest(manifest) if manifest
                             else AssetIndex.from_manifest())
    return _INDEX_CACHE[key]


def slugify(text: str, limit: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (s[:limit].rstrip("-")) or "scene"


def stable_seed(text: str) -> int:
    """Same prompt, same scene -- across runs and machines.

    Python's own hash() is salted per process, so the old default seed gave
    a different scene every time the same sentence was run.
    """
    return zlib.crc32(text.strip().lower().encode("utf-8")) % 100000


class GenerationError(RuntimeError):
    pass


def run(prompt: str, *, manifest: Optional[Path] = None,
        out_root: Path = Path("outputs"), overrides: Optional[dict] = None,
        client=None, log: Callable[[str], None] = print,
        unity_project: Optional[Path] = None, unity_full: bool = False,
        include_terrain: bool = True, slug: Optional[str] = None,
        show_graph: bool = False) -> dict:
    """Generate, playtest and export one scene. Returns a summary dict."""
    t0 = time.time()
    clean, opts = parse_options(prompt)
    for key, value in (overrides or {}).items():
        if value is not None and hasattr(opts, key):
            setattr(opts, key, value)
    seed = opts.seed if opts.seed is not None else stable_seed(clean)
    opts.seed = seed

    log(f'Prompt: "{clean}"')
    for note in opts.notes:
        log(f"  understood: {note}")

    # -- 1. prompt -> spec -------------------------------------------------
    spec = decompose(clean, client=client, force_fallback=opts.offline)
    if opts.size and opts.size != spec.terrain.size:
        spec.terrain.size = opts.size
    size_m = vocab.TERRAIN_SIZE_METRES[spec.terrain.size]
    log(f"[1/6] Understood ({spec.parser}): {spec.summary()}")
    warnings = list(spec.warnings)
    if not spec.theme_recognised:
        warnings.append(f"no exact theme for this place; built from the "
                        f"closest one ({spec.theme})")

    # -- 2. assets ----------------------------------------------------------
    try:
        index = load_index(manifest)
    except FileNotFoundError as e:
        raise GenerationError(str(e)) from e

    # -- 3. candidates ------------------------------------------------------
    graph, graph_source = None, None
    mode = opts.mode
    if mode == "graph":
        try:
            from .graph_builder import build_graph
            from .graph_dressing import dress as dress_graph
            graph, graph_source = build_graph(
                clean, spec, client=client, force_fallback=opts.offline,
                size_metres=size_m)
            dress_graph(graph, spec, size_m, density=opts.density)
            log(f"[2/6] Scene graph ({graph_source}): {graph.summary()}")
            if show_graph:
                from .graph_pipeline import describe, resolve_groups
                resolved_groups, _ = resolve_groups(graph, index,
                                                    theme=spec.theme)
                log(describe(graph, resolved_groups))
        except Exception as e:                   # fall back, don't fail
            warnings.append(f"scene graph failed ({e}); used rule layout")
            mode = "rules"

    if mode == "rules":
        spec, _ = dress_spec(spec, size_m, density=opts.density)
        resolved = resolve_scene(spec, index)
        warnings.extend(resolved.warnings)
        log(f"[2/6] Resolved assets: {resolved.summary()}")

    n = max(1, opts.candidates if opts.playtest else 1)
    log(f"[3/6] Building {n} candidate layout{'s' if n > 1 else ''}")
    candidates = []
    for i in range(n):
        terrain = Terrain.from_spec(spec, seed=seed + i)
        terrain.apply_features(spec.features, seed=seed + i)
        if mode == "graph":
            from .graph_pipeline import build_scene
            placed = build_scene(graph, terrain, index, spec=spec,
                                 seed=seed + i)
        else:
            placed = layout_scene(resolved, terrain, seed=seed + i)
            add_bridge_instances(placed, index, theme=spec.theme)
        report = validate(placed)
        quick = playtest(placed, train_bot=False) if opts.playtest else None
        score = quick.flow_score if quick else 0.0
        candidates.append({
            "seed": seed + i, "placed": placed, "validation": report,
            "flow": score, "valid": report["valid"],
            "instances": len(placed.instances),
        })
        log(f"      candidate {i + 1}: seed {seed + i}, "
            f"{len(placed.instances)} objects, valid={report['valid']}"
            + (f", flow {score:.0f}" if quick else ""))

    # -- 4. choose: valid first, then how it plays, then how full it is ----
    best = max(candidates, key=lambda c: (c["valid"], c["flow"],
                                          c["instances"]))
    placed, report = best["placed"], best["validation"]
    if n > 1:
        log(f"[4/6] Kept candidate seed {best['seed']} "
            f"(flow {best['flow']:.0f})")

    # -- 5. the bot playtest ---------------------------------------------------
    play = None
    if opts.playtest:
        log("[5/6] Bot playtest: training a Q-learning agent on the scene")
        play = playtest(placed)
        log(f"      {play.summary()}")

    # -- 6. export ------------------------------------------------------------
    slug = slug or slugify(clean)
    out_dir = Path(out_root) / slug
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    if opts.baked:
        export = export_glb(placed, out_dir / "scene.glb",
                            include_terrain=include_terrain)
        write_viewer(placed, out_dir, manifest_file="scene.glb",
                     title=clean[:60])
    else:
        export = export_instanced(placed, out_dir, palette=opts.palette)
        write_viewer(placed, out_dir, title=clean[:60])

    (out_dir / "lighting.json").write_text(
        json.dumps(lighting_config(placed), indent=2), encoding="utf-8")

    bridges = sum(1 for i in placed.instances
                  if getattr(i, "role", "") == "bridge")
    skipped_assets = [w for w in placed.warnings if w.startswith("skipped")]
    warnings.extend(placed.warnings)

    summary = {
        "prompt": prompt,
        "clean_prompt": clean,
        "slug": slug,
        "seed": seed,
        "mode": mode,
        "graph_source": graph_source,
        "options": opts.to_dict(),
        "spec_summary": spec.summary(),
        "theme": spec.theme,
        "theme_recognised": spec.theme_recognised,
        "terrain": spec.terrain.type,
        "size_m": size_m,
        "features": [f.type for f in spec.features],
        "lighting": {"time": spec.lighting.time_of_day,
                     "weather": spec.lighting.weather},
        "instances": len(placed.instances),
        "bridges": bridges,
        "triangles": export.get("total_triangles"),
        "size_mb": export.get("size_mb"),
        "validation": {k: v for k, v in report.items() if k != "details"},
        "candidates": [{k: c[k] for k in ("seed", "flow", "valid",
                                          "instances")}
                       for c in candidates],
        "chosen_seed": best["seed"],
        "skipped_assets": skipped_assets,
        "warnings": warnings,
        "seconds": None,
    }

    if play:
        (out_dir / "playtest.json").write_text(
            json.dumps(play.to_dict(), indent=2), encoding="utf-8")
        summary["playtest"] = {"flow_score": round(play.flow_score, 1),
                               "grade": play.grade,
                               "connected": play.connected}

    (out_dir / "scene.json").write_text(json.dumps({
        "prompt": prompt, "seed": best["seed"], "mode": mode,
        "graph_source": graph_source,
        "options": opts.to_dict(),
        "spec": spec.model_dump(),
        "graph": graph.model_dump() if graph is not None else None,
        "scene": placed.to_dict(),
        "validation": summary["validation"],
        "export": export,
    }, indent=2), encoding="utf-8")

    if opts.unity or unity_project:
        summary["unity"] = write_unity_bundle(
            placed, out_dir, slug, unity_project=unity_project,
            full=unity_full, include_terrain=include_terrain, log=log)

    summary["seconds"] = round(time.time() - t0, 1)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")
    log(f"[6/6] Exported {export.get('size_mb')} MB -> {out_dir} "
        f"in {summary['seconds']}s")
    summary["out_dir"] = str(out_dir)
    summary["export"] = export
    return summary


def write_unity_bundle(placed, out_dir: Path, slug: str,
                       unity_project: Optional[Path] = None,
                       full: bool = False, include_terrain: bool = True,
                       log: Callable[[str], None] = print) -> dict:
    """A Unity-ready folder: one baked .glb, lighting, editor scripts."""
    unity_dir = out_dir / "unity"
    unity_dir.mkdir(parents=True, exist_ok=True)
    baked = export_glb(placed, unity_dir / "scene.glb",
                       include_terrain=include_terrain)

    editor_src = REPO / "engine" / "unity" / "Editor"
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
        "   by name > com.unity.cloud.gltfast\n\n"
        "2. Drag scene.glb into your Assets folder. The whole environment\n"
        "   appears as one prefab.\n\n"
        "3. Lighting: Tools > Generated Scene > Apply Lighting... and pick\n"
        "   lighting.json to reproduce the browser preview.\n\n"
        f"Baked file: {baked['size_mb']} MB, "
        f"{baked['total_triangles']:,} triangles\n",
        encoding="utf-8")
    log(f"      Unity bundle: {unity_dir} ({baked['size_mb']} MB)")
    result = {"dir": str(unity_dir), "size_mb": baked["size_mb"]}

    if unity_project:
        project = Path(unity_project).expanduser().resolve()
        assets = project / "Assets"
        if not assets.is_dir():
            log(f"      ! {project} is not a Unity project "
                "(no Assets folder found)")
            result["copied"] = None
            return result
        target = assets / "GeneratedScenes" / slug
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(unity_dir / "scene.glb", target / "scene.glb")
        shutil.copyfile(out_dir / "lighting.json", target / "lighting.json")
        editor_target = assets / "Editor"
        editor_target.mkdir(exist_ok=True)
        if lighting_script.exists():
            shutil.copyfile(lighting_script,
                            editor_target / lighting_script.name)
        if full:
            for item in ("terrain.glb", "scene_manifest.json"):
                src = out_dir / item
                if src.exists():
                    shutil.copyfile(src, target / item)
            if (out_dir / "assets").is_dir():
                shutil.copytree(out_dir / "assets", target / "assets",
                                dirs_exist_ok=True)
            if importer.exists():
                shutil.copyfile(importer, editor_target / importer.name)
        rel = target.relative_to(project)
        log(f"      Copied into Unity project: {rel}")
        result["copied"] = str(rel)
    return result
