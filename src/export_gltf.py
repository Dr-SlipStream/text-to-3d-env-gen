"""
Stage 5a: assemble the placed scene into a single 3D file.

Takes a PlacedScene (positions, rotations, scales referencing asset files on
disk) and produces one self-contained `.glb` containing the terrain mesh and
every placed instance.

Why glTF/GLB: it's the standard interchange format for real-time 3D. One file
carries geometry, materials and a scene graph; Unity, Unreal, Blender and
three.js all import it directly. Exporting here rather than straight to a Unity
scene keeps the pipeline engine-agnostic -- the same output drives the web
viewer and, later, the engine import.

Instances share geometry where possible: 40 trees reference one mesh with 40
transforms rather than 40 copies. That keeps the file small and mirrors how a
game engine would actually render them.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import trimesh

from . import appearance
from .layout import PlacedScene
from .terrain import Terrain


def build_terrain_mesh(terrain: Terrain, subdivisions: int = 128,
                       path=None, centre=None,
                       core_radius: float = 0.0) -> trimesh.Trimesh:
    """Turn the heightmap into a mesh with painted ground.

    Beyond elevation colouring, this paints two things that make a scene read
    as inhabited rather than as models dropped on a lawn:

      the road -- a dirt corridor along the path. Keeping the corridor clear
      of objects is not enough; with nothing marking it, buildings appear to
      float in an empty field.

      the village ground -- earth trampled bare around the settlement,
      fading back into grass. Unbroken lawn right up to every doorstep is one
      of the clearest tells of procedural placement.

    Ground colour also gets low-frequency noise so large areas aren't a single
    flat shade.
    """
    n = subdivisions + 1
    xs = np.linspace(0, terrain.size, n)
    zs = np.linspace(0, terrain.size, n)

    # Vectorised bilinear sample of the heightfield. Calling height_at per
    # vertex would be ~16,000 Python calls per scene; this is one array op.
    gx = np.clip(xs / terrain.size * (terrain.res - 1), 0, terrain.res - 1)
    gz = np.clip(zs / terrain.size * (terrain.res - 1), 0, terrain.res - 1)
    x0 = np.floor(gx).astype(int); x1 = np.minimum(x0 + 1, terrain.res - 1)
    z0 = np.floor(gz).astype(int); z1 = np.minimum(z0 + 1, terrain.res - 1)
    fx = (gx - x0)[None, :]; fz = (gz - z0)[:, None]

    h = (terrain.heights[np.ix_(z0, x0)] * (1 - fx) * (1 - fz)
         + terrain.heights[np.ix_(z0, x1)] * fx * (1 - fz)
         + terrain.heights[np.ix_(z1, x0)] * (1 - fx) * fz
         + terrain.heights[np.ix_(z1, x1)] * fx * fz)

    gridx, gridz = np.meshgrid(xs, zs)
    verts = np.column_stack([gridx.ravel(), h.ravel(),
                             gridz.ravel()]).astype(np.float32)

    # -- base colour by elevation ----------------------------------------
    lo, hi = float(terrain.heights.min()), float(terrain.heights.max())
    span = max(hi - lo, 1e-6)
    c_lo = np.array(terrain.preset.colour_low)
    c_hi = np.array(terrain.preset.colour_high)
    dirt = np.array(terrain.preset.colour_dirt)

    t = ((h.ravel() - lo) / span)[:, None]
    rgb = c_lo + (c_hi - c_lo) * t

    px = gridx.ravel()
    pz = gridz.ravel()

    # -- village ground ---------------------------------------------------
    if centre is not None and core_radius > 0:
        d = np.hypot(px - centre[0], pz - centre[1])
        # Strongest at the heart, gone by the edge of the settlement.
        worn = np.clip(1.0 - d / (core_radius * 1.05), 0.0, 1.0) ** 1.4
        rgb = rgb * (1 - worn[:, None] * 0.55) + dirt * (worn[:, None] * 0.55)

    # -- road -------------------------------------------------------------
    if path:
        pts = np.asarray(path, dtype=float)
        # Chunked so a dense path on a fine grid doesn't allocate a huge
        # intermediate array.
        best = np.full(px.shape, np.inf)
        for i in range(0, len(pts), 24):
            chunk = pts[i:i + 24]
            d = np.min(np.hypot(px[:, None] - chunk[:, 0],
                                pz[:, None] - chunk[:, 1]), axis=1)
            best = np.minimum(best, d)

        ROAD_HALF, VERGE = 3.4, 3.0
        road = np.clip(1.0 - (best - ROAD_HALF) / VERGE, 0.0, 1.0)
        road = np.where(best <= ROAD_HALF, 1.0, road) ** 1.2
        rgb = rgb * (1 - road[:, None] * 0.85) + dirt * (road[:, None] * 0.85)

    # -- variation --------------------------------------------------------
    # Low-frequency noise so big areas aren't one flat shade. Deterministic
    # from the terrain seed, so scenes stay reproducible.
    rng = np.random.default_rng(terrain.seed + 991)
    blotch = _terrain_noise_layer(n, 6, rng).ravel()
    rgb = rgb * (0.90 + 0.20 * blotch[:, None])

    rgb = np.clip(rgb, 0, 1)
    colours = np.column_stack([
        (rgb * 255).astype(np.uint8),
        np.full((verts.shape[0], 1), 255, dtype=np.uint8)])

    # Two triangles per quad, built with array arithmetic rather than loops.
    j, i = np.meshgrid(np.arange(subdivisions), np.arange(subdivisions),
                       indexing="ij")
    a = (j * n + i).ravel()
    b, c, d_ = a + 1, a + n, a + n + 1
    faces = np.vstack([np.column_stack([a, c, b]),
                       np.column_stack([b, c, d_])])

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # Force smooth vertex normals to be computed and cached. Without them the
    # exporter writes no NORMAL attribute, so the renderer falls back to flat
    # per-face shading and the ground reads as harsh faceted panels instead of
    # rolling hills. This is the same "poly count around edges" issue: the
    # triangle count is unchanged, but interpolated normals make the surface
    # read as smooth.
    _ = mesh.vertex_normals

    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh,
                                              vertex_colors=colours)
    return mesh


def _terrain_noise_layer(res: int, cells: int, rng) -> np.ndarray:
    """Smooth 0..1 noise on a res x res grid, for ground colour variation."""
    grid = rng.random((cells + 1, cells + 1))
    coords = np.linspace(0, cells, res, endpoint=False)
    idx = np.floor(coords).astype(int)
    frac = coords - idx
    frac = frac * frac * (3.0 - 2.0 * frac)

    iy, ix = np.meshgrid(idx, idx, indexing="ij")
    fy, fx = np.meshgrid(frac, frac, indexing="ij")

    top = grid[iy, ix] * (1 - fx) + grid[iy, ix + 1] * fx
    bot = grid[iy + 1, ix] * (1 - fx) + grid[iy + 1, ix + 1] * fx
    return top * (1 - fy) + bot * fy


def _transform(position, rotation_y: float, scale: float) -> np.ndarray:
    """Build a 4x4 TRS matrix (rotation about Y only -- objects stay upright)."""
    c, s = math.cos(rotation_y), math.sin(rotation_y)
    m = np.eye(4)
    m[:3, :3] = np.array([
        [c * scale, 0.0, s * scale],
        [0.0, scale, 0.0],
        [-s * scale, 0.0, c * scale],
    ])
    m[:3, 3] = position
    return m


def _recover_colour(mesh) -> Optional[tuple]:
    """Try to read a usable base colour off a loaded mesh's material.

    Assets arrive with wildly inconsistent material data -- texture atlases,
    per-part PBR materials, vertex colours, or nothing at all. When a colour
    survives loading we keep it; when it doesn't, the caller falls back to the
    palette.

    The subtlety: trimesh *invents* a neutral grey when a mesh has no colour
    data, and reading `vertex_colors` returns that placeholder rather than
    nothing. Accepting it produced a scene of uniform grey blocks. The
    `visual.kind` property is the reliable signal -- it is None precisely when
    the colour data is trimesh's own invention.
    """
    visual = getattr(mesh, "visual", None)
    if visual is None:
        return None

    kind = getattr(visual, "kind", None)

    if kind in ("vertex", "face"):
        try:
            source = (visual.vertex_colors if kind == "vertex"
                      else visual.face_colors)
            arr = np.asarray(source, dtype=float)[:, :3] / 255.0
            mean = tuple(arr.mean(axis=0))
            if not _is_placeholder_grey(mean):
                return mean
        except Exception:
            pass

    if kind == "texture":
        try:
            material = getattr(visual, "material", None)
            image = getattr(material, "image", None) or getattr(
                material, "baseColorTexture", None)
            if image is not None:
                # Average the texture. Crude, but these are flat-shaded
                # stylised atlases, so an average is representative.
                arr = np.asarray(image.convert("RGB"), dtype=float) / 255.0
                mean = tuple(arr.reshape(-1, 3).mean(axis=0))
                if not _is_placeholder_grey(mean):
                    return mean
        except Exception:
            pass

    try:
        material = getattr(visual, "material", None)
        if material is not None:
            factor = getattr(material, "baseColorFactor", None)
            if factor is None:
                factor = getattr(material, "diffuse", None)
            if factor is not None:
                arr = np.asarray(factor, dtype=float)[:3]
                if arr.max() > 1.5:          # 0-255 rather than 0-1
                    arr = arr / 255.0
                mean = tuple(arr)
                if not _is_placeholder_grey(mean):
                    return mean
    except Exception:
        pass

    return None


def _is_placeholder_grey(rgb) -> bool:
    """True for trimesh's default neutral grey, or anything indistinguishable.

    Its default is (102, 102, 102). A genuinely chosen colour that happens to
    be exactly neutral mid-grey is rare enough that treating it as missing is
    the right trade.
    """
    r, g, b = rgb
    if max(abs(r - g), abs(g - b), abs(r - b)) > 0.02:
        return False                      # has a hue, so it's a real choice
    return 0.34 < (r + g + b) / 3.0 < 0.46


def _load_asset(path: str, name: str, category: str,
                cache: Dict[str, Optional[trimesh.Trimesh]]):
    """Load a model once and reuse it for every instance of that asset.

    Also guarantees the mesh has normals and a visible colour. Source
    materials frequently do not survive being loaded and re-exported, and the
    symptom is an entire scene of black silhouettes -- so rather than trust
    them, we verify and substitute a palette colour when needed.
    """
    # Keyed on name and category as well as path: two different objects can
    # share a mesh file (a variant, or one model matching several queries) and
    # must still get their own palette colour -- otherwise flowers inherit
    # whatever colour the tree that loaded first was given.
    key = (path, name, category)
    if key in cache:
        return cache[key]
    try:
        loaded = trimesh.load(path, force="mesh", process=False)
        if loaded is None or not hasattr(loaded, "faces") or len(loaded.faces) == 0:
            cache[key] = None
            return None

        loaded = loaded.copy()

        # Packs disagree about whether a model's origin is at its centre or
        # its foot; without this, half the library ends up half-buried.
        loaded.apply_translation([0, -loaded.bounds[0][1], 0])

        # Smooth normals: without them the renderer shades flat per face and
        # every curved surface shows harsh facets.
        _ = loaded.vertex_normals

        recovered = _recover_colour(loaded)
        if appearance.is_usable_colour(recovered):
            colours = np.tile(
                np.array([*(np.clip(recovered, 0, 1) * 255).astype(np.uint8),
                          255], dtype=np.uint8),
                (len(loaded.vertices), 1))
        else:
            colours = _palette_colours(loaded, name, category, path)

        loaded.visual = trimesh.visual.ColorVisuals(mesh=loaded,
                                                    vertex_colors=colours)

        cache[key] = loaded
    except Exception:
        cache[key] = None
    return cache[key]


def _palette_colours(mesh, name: str, category: str, path: str) -> np.ndarray:
    """Per-vertex colours from the palette, two-toned by height where it helps.

    A single flat colour per model is what makes generated scenes read as
    untextured blocks -- houses with no roof, trees with no trunk. Splitting
    the model by height and colouring the bands separately costs no geometry
    and no textures.
    """
    key = f"{name}|{path}"
    n_verts = len(mesh.vertices)

    tone = appearance.two_tone(name, category, variation_key=key)
    if tone is None:
        rgb = appearance.varied_colour(name, category, variation_key=key)
        return np.tile(
            np.array([*(np.clip(rgb, 0, 1) * 255).astype(np.uint8), 255],
                     dtype=np.uint8), (n_verts, 1))

    lower_rgb, upper_rgb, split = tone

    ys = mesh.vertices[:, 1]
    lo, hi = float(ys.min()), float(ys.max())
    span = max(hi - lo, 1e-6)
    t = (ys - lo) / span

    lower = np.clip(np.asarray(lower_rgb), 0, 1) * 255
    upper = np.clip(np.asarray(upper_rgb), 0, 1) * 255

    # Narrow blend across the boundary so the seam isn't a hard line, but
    # keep it tight -- a soft gradient reads as a lighting error.
    blend = np.clip((t - split) / 0.06, 0.0, 1.0)[:, None]
    rgb = lower * (1 - blend) + upper * blend

    out = np.empty((n_verts, 4), dtype=np.uint8)
    out[:, :3] = rgb.astype(np.uint8)
    out[:, 3] = 255
    return out


def export_glb(scene: PlacedScene, out_path: Path,
               include_terrain: bool = True) -> dict:
    """Write the whole scene to a single .glb file.

    Returns a report of what was written, including the triangle budget --
    useful both for the poly-count question and for spotting a scene that
    would be too heavy to render smoothly.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gltf_scene = trimesh.Scene()
    cache: Dict[str, Optional[trimesh.Trimesh]] = {}

    terrain_tris = 0
    if include_terrain and scene.terrain is not None:
        ground = build_terrain_mesh(
            scene.terrain, path=scene.path, centre=scene.centre,
            core_radius=scene.core_radius)
        terrain_tris = len(ground.faces)
        gltf_scene.add_geometry(ground, node_name="terrain",
                                geom_name="terrain")

    placed = 0
    missing: List[str] = []
    instance_tris = 0

    for n, inst in enumerate(scene.instances):
        mesh = _load_asset(inst.asset_file, inst.asset_name, inst.category,
                           cache)
        if mesh is None:
            missing.append(inst.asset_name)
            continue

        gltf_scene.add_geometry(
            mesh,
            node_name=f"{inst.name}_{n}",
            geom_name=inst.asset_name,
            transform=_transform(inst.position, inst.rotation_y, inst.scale),
        )
        instance_tris += len(mesh.faces)
        placed += 1

    gltf_scene.export(out_path)
    ensure_pbr_material(out_path)

    return {
        "file": str(out_path),
        "size_mb": round(out_path.stat().st_size / 1_048_576, 2),
        "instances_exported": placed,
        "instances_missing": len(missing),
        "missing_assets": sorted(set(missing))[:10],
        "unique_meshes": sum(1 for v in cache.values() if v is not None),
        "terrain_triangles": terrain_tris,
        "instance_triangles": instance_tris,
        "total_triangles": terrain_tris + instance_tris,
    }


def ensure_pbr_material(glb_path: Path, roughness: float = 0.9) -> bool:
    """Give an exported .glb an explicit, sane material.

    trimesh writes vertex colours but no material block. The glTF default
    material is metallic 1.0 / rough 1.0, which renders as dark metal in any
    engine without an environment map -- our web viewer overrides it in code,
    but Unity has no reason to, so an imported scene would come in black.

    Writing an explicit non-metallic material makes the file look correct
    everywhere, which is the point of exporting a standard format.
    """
    import json as _json
    import struct as _struct

    glb_path = Path(glb_path)
    try:
        data = glb_path.read_bytes()
        if data[:4] != b"glTF":
            return False

        json_len = _struct.unpack("<I", data[12:16])[0]
        json_bytes = data[20:20 + json_len]
        rest = data[20 + json_len:]
        gltf = _json.loads(json_bytes)

        if gltf.get("materials"):
            return False                      # already has one; leave it

        gltf["materials"] = [{
            "name": "generated",
            "doubleSided": False,
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": roughness,
            },
        }]
        for mesh in gltf.get("meshes", []):
            for primitive in mesh.get("primitives", []):
                primitive.setdefault("material", 0)

        new_json = _json.dumps(gltf, separators=(",", ":")).encode("utf-8")
        # Chunks must be 4-byte aligned; JSON pads with spaces.
        pad = (4 - len(new_json) % 4) % 4
        new_json += b" " * pad

        header = bytearray(data[:12])
        chunk = _struct.pack("<I", len(new_json)) + b"JSON" + new_json
        total = 12 + len(chunk) + len(rest)
        header[8:12] = _struct.pack("<I", total)

        glb_path.write_bytes(bytes(header) + chunk + rest)
        return True
    except Exception:
        return False
