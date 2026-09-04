"""Tests for engine-import readiness.

"It imports into Unity without errors" is a claim that gets tested live, so
these check the properties Unity's glTF importer actually depends on: a valid
container, no exotic extensions, and a material that doesn't render as black
metal.
"""

import json
import struct
from pathlib import Path

import pytest

from src import fallback_parser
from src.asset_resolution import resolve_scene
from src.export_gltf import ensure_pbr_material, export_glb
from src.export_scene import export_instanced
from src.layout import layout_scene
from src.terrain import Terrain


def read_glb(path):
    data = path.read_bytes()
    assert data[:4] == b"glTF", "not a GLB container"

    version = struct.unpack("<I", data[4:8])[0]
    declared = struct.unpack("<I", data[8:12])[0]
    json_len = struct.unpack("<I", data[12:16])[0]
    gltf = json.loads(data[20:20 + json_len])

    return {"version": version, "declared": declared,
            "actual": len(data), "gltf": gltf}


@pytest.fixture
def exported(tmp_path):
    """A scene built from real files on disk.

    The shared synthetic library points at paths that don't exist, which is
    fine for testing retrieval but not for testing export -- nothing would be
    written.
    """
    import numpy as np
    import trimesh

    from src.layout import PlacedInstance, PlacedScene

    models = {}
    for name, colour in [("hut", (200, 120, 90)), ("tree", (60, 140, 70))]:
        mesh = trimesh.creation.box(extents=(1, 2, 1))
        mesh.apply_translation([0, 1, 0])
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh=mesh,
            vertex_colors=np.tile([*colour, 255], (len(mesh.vertices), 1)))
        path = tmp_path / f"{name}.glb"
        mesh.export(path)
        models[name] = path

    terrain = Terrain.generate("grassland", 120, seed=7)
    scene = PlacedScene(terrain=terrain, seed=7)
    scene.path = [(20.0, 20.0), (60.0, 60.0), (100.0, 100.0)]
    scene.centre = (60.0, 60.0)
    scene.core_radius = 30.0

    for i in range(12):
        name = "hut" if i % 3 == 0 else "tree"
        scene.instances.append(PlacedInstance(
            name=name, category="building" if name == "hut" else "vegetation",
            asset_name=name, asset_file=str(models[name]),
            position=(20.0 + i * 6, 1.0, 40.0),
            rotation_y=0.2 * i, scale=1.5, radius=1.0, triangles=12))

    baked = tmp_path / "baked" / "scene.glb"
    export_glb(scene, baked)
    inst_dir = tmp_path / "inst"
    export_instanced(scene, inst_dir)
    return baked, inst_dir


def test_glb_container_is_well_formed(exported):
    """A declared length that disagrees with the file size is the classic
    cause of an importer rejecting a file outright."""
    baked, inst_dir = exported
    for path in [baked, inst_dir / "terrain.glb"]:
        info = read_glb(path)
        assert info["version"] == 2
        assert info["declared"] == info["actual"]
        assert info["gltf"]["asset"]["version"] == "2.0"


def test_no_extensions_are_required(exported):
    """Draco, KTX and meshopt each need a separate Unity package. Requiring
    one turns a drag-and-drop into a support call.
    """
    baked, inst_dir = exported
    for path in [baked, inst_dir / "terrain.glb"]:
        gltf = read_glb(path)["gltf"]
        assert not gltf.get("extensionsRequired"), \
            f"{path.name} requires {gltf['extensionsRequired']}"


def test_meshes_have_an_explicit_material(exported):
    """trimesh writes vertex colours but no material, and the glTF default is
    metallic 1.0 -- which renders as dark metal in any engine without an
    environment map. Our web viewer overrides it in code; Unity does not.
    """
    baked, inst_dir = exported
    for path in [baked, inst_dir / "terrain.glb"]:
        gltf = read_glb(path)["gltf"]
        materials = gltf.get("materials", [])
        assert materials, f"{path.name} has no material"

        pbr = materials[0]["pbrMetallicRoughness"]
        assert pbr["metallicFactor"] == 0.0, "would render as metal"
        assert pbr["roughnessFactor"] > 0.5

        for mesh in gltf["meshes"]:
            for primitive in mesh["primitives"]:
                assert "material" in primitive


def test_geometry_carries_normals_and_colour(exported):
    """Without NORMAL the engine falls back to flat shading; without COLOR_0
    the palette is lost."""
    baked, inst_dir = exported
    gltf = read_glb(inst_dir / "terrain.glb")["gltf"]
    attrs = set()
    for mesh in gltf["meshes"]:
        for primitive in mesh["primitives"]:
            attrs |= set(primitive["attributes"])
    assert {"POSITION", "NORMAL", "COLOR_0"} <= attrs


def test_manifest_matches_the_unity_importer_schema(exported):
    """The C# side deserialises these exact field names; a rename here breaks
    the editor script silently."""
    _, inst_dir = exported
    manifest = json.loads(
        (inst_dir / "scene_manifest.json").read_text(encoding="utf-8"))

    assert {"version", "terrain", "assets", "instances", "stats"} \
        <= set(manifest)
    assert {"file", "name", "base_offset", "triangles", "count"} \
        <= set(manifest["assets"][0])
    assert {"asset", "name", "category", "p", "r", "s"} \
        <= set(manifest["instances"][0])
    assert {"instance_count", "unique_assets", "rendered_triangles"} \
        <= set(manifest["stats"])
    assert len(manifest["instances"][0]["p"]) == 3


def test_asset_paths_are_relative(exported):
    """The importer resolves them against the manifest's own folder, so an
    absolute path from the generating machine would break on any other."""
    _, inst_dir = exported
    manifest = json.loads(
        (inst_dir / "scene_manifest.json").read_text(encoding="utf-8"))

    for asset in manifest["assets"]:
        assert not asset["file"].startswith("/")
        assert ":" not in asset["file"]          # no C:\ style paths
        assert asset["file"].startswith("assets/")


def test_material_injection_is_idempotent(exported):
    """Re-running the export must not corrupt an already-patched file."""
    baked, _ = exported
    before = baked.read_bytes()

    assert ensure_pbr_material(baked) is False   # already has one
    assert baked.read_bytes() == before


# --- copying into a Unity project -------------------------------------------

def test_unity_project_copy_lands_under_assets(tmp_path):
    """Unity only imports files that physically live under Assets/.

    Dragging a .glb in from anywhere else fails with "Invalid AssetDatabase
    path", and without glTFast installed Unity refuses the drop outright.
    Copying the files in sidesteps both.

    This test passes its own manifest via --manifest. An earlier version
    overwrote the project's real asset manifest, which meant running the test
    suite could poison the next scene the user generated -- it referenced a
    temporary file that no longer existed, and Unity failed to import it.
    """
    import json
    import subprocess
    import sys

    import numpy as np
    import trimesh

    project = tmp_path / "UnityProject"
    (project / "Assets").mkdir(parents=True)
    (project / "ProjectSettings").mkdir()

    model = tmp_path / "box.glb"
    mesh = trimesh.creation.box(extents=(1, 2, 1))
    mesh.apply_translation([0, 1, 0])
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh,
        vertex_colors=np.tile([120, 140, 90, 255], (len(mesh.vertices), 1)))
    mesh.export(model)

    rows = [("house", "building", 1.0, 1.6), ("tree", "vegetation", 0.4, 3.0),
            ("rock", "rock", 0.3, 0.4), ("barrel", "prop", 0.2, 0.5),
            ("fence", "structure", 0.5, 0.6),
            ("torch", "light_source", 0.1, 1.0)]

    manifest = tmp_path / "test_manifest.json"
    manifest.write_text(json.dumps({
        "version": 1, "generated": "test", "asset_count": len(rows),
        "assets": [{
            "id": f"t/{n}", "name": n, "category": c, "pack": "t",
            "file": str(model), "format": ".glb", "radius": r, "height": h,
            "measured": True, "triangles": 12, "modular": False,
            "variant": None, "theme_hints": [], "tags": [n, c],
        } for n, c, r, h in rows],
    }), encoding="utf-8")

    repo = Path(__file__).resolve().parents[1]
    real_manifest = repo / "data" / "asset_library" / "manifest.json"
    before = real_manifest.read_bytes() if real_manifest.exists() else None

    result = subprocess.run(
        [sys.executable, str(repo / "scripts" / "generate.py"),
         "a medieval village", "--fallback", "--seed", "1",
         "--manifest", str(manifest),
         "--out", str(tmp_path / "out"),
         "--unity-project", str(project)],
        capture_output=True, text=True, cwd=repo, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr

    generated = project / "Assets" / "GeneratedScenes"
    assert generated.is_dir(), "nothing copied into Assets/"

    scene_dir = next(generated.iterdir())
    assert (scene_dir / "scene.glb").exists()

    # By default only the baked scene is copied. The per-model files are
    # copies of the original pack assets, and a few of those fail Unity's
    # glTF importer -- copying them in filled the console with errors about
    # models the drag-and-drop route never touches.
    assert not (scene_dir / "assets").exists(), \
        "per-model assets copied without --unity-full"

    # --unity-full opts back in to the manifest route.
    result = subprocess.run(
        [sys.executable, str(repo / "scripts" / "generate.py"),
         "a forest camp", "--fallback", "--seed", "2",
         "--manifest", str(manifest),
         "--out", str(tmp_path / "out2"),
         "--unity-project", str(project), "--unity-full"],
        capture_output=True, text=True, cwd=repo, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr

    full = next(d for d in generated.iterdir() if "forest" in d.name)
    assert (full / "assets").is_dir()
    assert (full / "scene_manifest.json").exists()
    assert (project / "Assets" / "Editor"
            / "GeneratedSceneImporter.cs").exists()

    # And the real library must be exactly as we found it.
    after = real_manifest.read_bytes() if real_manifest.exists() else None
    assert after == before, "the test modified the project's asset manifest"


# --- lighting export --------------------------------------------------------

def test_lighting_is_exported_as_data(index, tmp_path):
    """glTF carries geometry and materials but has no concept of fog, ambient
    light or exposure, so an imported scene arrives lit by the project's
    defaults -- a night scene comes in looking like an overcast afternoon.
    """
    from src.layout import layout_scene
    from src.viewer import lighting_config

    spec = fallback_parser.parse("a forest camp at night with a campfire")
    scene = layout_scene(resolve_scene(spec, index),
                         Terrain.from_spec(spec, seed=1), seed=1)
    cfg = lighting_config(scene)

    for key in ("sun_colour", "sun_intensity", "sun_elevation_deg",
                "ambient_sky", "ambient_ground", "ambient_intensity",
                "fog_colour", "fog_density", "exposure",
                "lamps_emit", "lamp_positions"):
        assert key in cfg, f"missing {key}"

    assert cfg["time_of_day"] == "night"
    assert cfg["lamps_emit"] is True


def test_lighting_lists_every_lamp_position(index):
    """An engine needs the positions to put real lights at the campfires;
    otherwise they're unlit props."""
    from src.layout import layout_scene
    from src.viewer import lighting_config

    spec = fallback_parser.parse("a village at night with torches")
    scene = layout_scene(resolve_scene(spec, index),
                         Terrain.from_spec(spec, seed=2), seed=2)
    cfg = lighting_config(scene)

    expected = [i for i in scene.instances if i.category == "light_source"]
    assert len(cfg["lamp_positions"]) == len(expected)
    for position in cfg["lamp_positions"]:
        assert len(position) == 3


def test_daylight_scenes_do_not_ask_for_lamps(index):
    from src.layout import layout_scene
    from src.viewer import lighting_config

    spec = fallback_parser.parse("a village on a sunny day")
    scene = layout_scene(resolve_scene(spec, index),
                         Terrain.from_spec(spec, seed=1), seed=1)
    assert lighting_config(scene)["lamps_emit"] is False


def test_viewer_and_engine_lighting_agree(index, tmp_path):
    """Both read from one source, so the browser preview and the engine
    cannot drift apart."""
    import json as _json

    from src.layout import layout_scene
    from src.viewer import lighting_config, write_viewer

    spec = fallback_parser.parse("a forest camp at night")
    scene = layout_scene(resolve_scene(spec, index),
                         Terrain.from_spec(spec, seed=1), seed=1)

    engine = lighting_config(scene)
    html = write_viewer(scene, tmp_path).read_text(encoding="utf-8")
    viewer = _json.loads(html.split("const CFG = ")[1].split(";\n")[0])

    assert engine["sun_colour"] == viewer["sun"]
    assert engine["sun_intensity"] == viewer["sunI"]
    assert engine["fog_colour"] == viewer["fogColour"]
    assert abs(engine["fog_density"] - viewer["fogDensity"]) < 1e-6
    assert engine["exposure"] == viewer["exposure"]
