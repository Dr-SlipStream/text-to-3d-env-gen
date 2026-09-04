"""
Stage 5a (alternative): export the scene without destroying its materials.

The baked exporter in `export_gltf` loads every asset, flattens it to a single
vertex colour, and merges everything into one file. That was written to solve
black silhouettes, and it does -- but it throws away the textures the asset
packs ship with. Kenney and Quaternius models carry proper texture atlases;
averaging one down to a single RGB value is why bushes came out cyan and walls
navy.

This exporter keeps the originals intact:

  - each distinct asset file is copied (or converted) into `assets/` untouched
  - the terrain is exported on its own, with our painted vertex colours
  - a manifest records every instance's position, rotation and scale

The viewer then loads each asset once and places copies at those transforms.
Textures survive, the output is smaller (one copy of each model rather than one
per instance), and it mirrors how a game engine actually works -- which makes
the Unity path a straight import rather than a re-export.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import trimesh

from .export_gltf import build_terrain_mesh, ensure_pbr_material
from .layout import PlacedScene

# Formats a browser can load directly. Anything else is converted on the way
# out, accepting whatever material data survives the conversion.
WEB_READY = {".glb"}


def _prepare_asset(src: Path, dest_dir: Path,
                   cache: Dict[str, Optional[dict]],
                   palette: bool = False,
                   name: str = "", category: str = "prop") -> Optional[dict]:
    """Copy or convert one asset, and measure where its base sits.

    Packs disagree about whether a model's origin is at its centre or its
    foot. We record the offset rather than baking it in, so the copied file
    stays byte-identical to the original where possible.
    """
    key = f"{src}|{name}|{category}" if palette else str(src)
    if key in cache:
        return cache[key]

    try:
        if not src.exists():
            cache[key] = None
            return None

        stem = src.stem.replace(" ", "_")
        if palette:
            # Distinct colours per requested object need distinct files.
            stem = f"{stem}__{name.replace(' ', '_')}"[:60]
        suffix = src.suffix.lower()
        out_name = f"{stem}.glb"

        # Avoid collisions between packs that use the same filename.
        n = 1
        while (dest_dir / out_name).exists() and \
                cache.get(f"__name__{out_name}") not in (None, key):
            out_name = f"{stem}_{n}.glb"
            n += 1
        cache[f"__name__{out_name}"] = key

        dest = dest_dir / out_name

        if palette:
            # Recolour from our own palette instead of trusting the source.
            # Packs vary wildly in how their materials survive loading -- some
            # arrive with a working texture, some with nothing, and a mesh
            # with no usable material renders flat white or takes on whatever
            # tint the scene lighting gives it. Overriding gives one coherent
            # look across mismatched packs, at the cost of the original art.
            from .export_gltf import _palette_colours

            mesh = trimesh.load(src, force="mesh", process=False)
            if mesh is None or not hasattr(mesh, "faces"):
                cache[key] = None
                return None
            mesh = mesh.copy()
            _ = mesh.vertex_normals
            mesh.visual = trimesh.visual.ColorVisuals(
                mesh=mesh,
                vertex_colors=_palette_colours(mesh, name, category, str(src)))
            mesh.export(dest)
            ensure_pbr_material(dest)
        elif suffix in WEB_READY:
            shutil.copyfile(src, dest)
            mesh = trimesh.load(src, force="mesh", process=False)
        else:
            mesh = trimesh.load(src, force="mesh", process=False)
            if mesh is None or not hasattr(mesh, "faces"):
                cache[key] = None
                return None
            mesh.export(dest)
            ensure_pbr_material(dest)

        if mesh is None or not hasattr(mesh, "bounds") or mesh.bounds is None:
            base_offset, triangles = 0.0, 0
        else:
            base_offset = float(mesh.bounds[0][1])
            triangles = int(len(mesh.faces))

        info = {
            "file": f"assets/{out_name}",
            "base_offset": round(base_offset, 4),
            "triangles": triangles,
        }
        cache[key] = info
        return info
    except Exception:
        cache[key] = None
        return None


def export_instanced(scene: PlacedScene, out_dir: Path,
                     terrain_subdivisions: int = 128,
                     palette: bool = False) -> dict:
    """Write terrain, original assets, and an instance manifest.

    Returns a report including the triangle budget -- both as drawn (unique
    geometry) and as rendered (every instance), which are very different
    numbers once instancing is in play.
    """
    out_dir = Path(out_dir)
    asset_dir = out_dir / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    # -- terrain ----------------------------------------------------------
    terrain_tris = 0
    if scene.terrain is not None:
        ground = build_terrain_mesh(
            scene.terrain, subdivisions=terrain_subdivisions,
            path=scene.path, centre=scene.centre,
            core_radius=scene.core_radius)
        ground.export(out_dir / "terrain.glb")
        ensure_pbr_material(out_dir / "terrain.glb")
        terrain_tris = len(ground.faces)

    # -- assets -----------------------------------------------------------
    cache: Dict[str, Optional[dict]] = {}
    assets: Dict[str, dict] = {}
    instances: List[dict] = []
    missing: List[str] = []
    rendered_tris = 0

    for inst in scene.instances:
        info = _prepare_asset(Path(inst.asset_file), asset_dir, cache,
                              palette=palette, name=inst.asset_name,
                              category=inst.category)
        if info is None:
            missing.append(inst.asset_name)
            continue

        asset_key = info["file"]
        if asset_key not in assets:
            assets[asset_key] = {
                "file": info["file"],
                "base_offset": info["base_offset"],
                "triangles": info["triangles"],
                "name": inst.asset_name,
                "count": 0,
            }
        assets[asset_key]["count"] += 1
        rendered_tris += info["triangles"]

        instances.append({
            "asset": asset_key,
            "name": inst.name,
            "category": inst.category,
            # Rounded: three decimals is sub-millimetre and keeps the manifest
            # a fraction of the size for a scene of a thousand objects.
            "p": [round(v, 3) for v in inst.position],
            "r": round(inst.rotation_y, 4),
            "s": round(inst.scale, 4),
        })

    spec = scene.spec
    manifest = {
        "version": 1,
        "terrain": "terrain.glb" if scene.terrain is not None else None,
        "size": scene.terrain.size if scene.terrain else 120.0,
        "ground_y": (round(float(scene.terrain.heights.min()), 3)
                     if scene.terrain is not None else 0.0),
        "terrain_max": (round(float(scene.terrain.heights.max()), 3)
                        if scene.terrain is not None else 2.0),
        "centre": ([round(v, 2) for v in scene.centre]
                   if scene.centre else None),
        "path": [[round(x, 2), round(z, 2)] for x, z in scene.path],
        "assets": list(assets.values()),
        "instances": instances,
        "stats": {
            "instance_count": len(instances),
            "unique_assets": len(assets),
            "terrain_triangles": terrain_tris,
            "rendered_triangles": terrain_tris + rendered_tris,
            "unique_triangles": terrain_tris + sum(
                a["triangles"] for a in assets.values()),
        },
    }
    (out_dir / "scene_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8")

    asset_bytes = sum(f.stat().st_size for f in asset_dir.glob("*.glb"))
    terrain_bytes = ((out_dir / "terrain.glb").stat().st_size
                     if (out_dir / "terrain.glb").exists() else 0)

    return {
        "mode": "instanced",
        "size_mb": round((asset_bytes + terrain_bytes) / 1_048_576, 2),
        "instances_exported": len(instances),
        "instances_missing": len(missing),
        "missing_assets": sorted(set(missing))[:10],
        "unique_meshes": len(assets),
        "terrain_triangles": terrain_tris,
        "instance_triangles": rendered_tris,
        "total_triangles": terrain_tris + rendered_tris,
        "unique_triangles": manifest["stats"]["unique_triangles"],
    }
