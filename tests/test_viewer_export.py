"""Tests for stage 5 (glTF export + viewer).

These catch the rendering bugs that made the first generated scene appear as
a blank grey field: fog tuned for the wrong scale, missing surface normals,
and no horizon behind the terrain.
"""

import json
import math

import pytest

from src import fallback_parser
from src.asset_resolution import resolve_scene
from src.export_gltf import build_terrain_mesh
from src.layout import layout_scene
from src.terrain import Terrain
from src.viewer import LIGHTING, WEATHER_FOG, write_viewer


# --- terrain mesh -----------------------------------------------------------

def test_terrain_mesh_has_normals():
    """Without normals the renderer falls back to flat per-face shading and
    the ground reads as harsh faceted panels."""
    t = Terrain.generate("grassland", 120, seed=1)
    mesh = build_terrain_mesh(t, subdivisions=16)
    assert mesh.vertex_normals is not None
    assert len(mesh.vertex_normals) == len(mesh.vertices)


def test_terrain_mesh_has_vertex_colours():
    t = Terrain.generate("grassland", 120, seed=1)
    mesh = build_terrain_mesh(t, subdivisions=16)
    assert mesh.visual.vertex_colors is not None
    assert len(mesh.visual.vertex_colors) == len(mesh.vertices)


def test_terrain_mesh_matches_heightfield():
    """The mesh must follow the terrain, or objects sit off the surface."""
    t = Terrain.generate("rocky", 120, seed=2)
    mesh = build_terrain_mesh(t, subdivisions=32)
    ys = mesh.vertices[:, 1]
    assert abs(ys.min() - t.heights.min()) < 0.5
    assert abs(ys.max() - t.heights.max()) < 0.5


def test_terrain_mesh_export_includes_attributes(tmp_path):
    """glTF must carry POSITION, NORMAL and COLOR_0 or the ground renders
    untextured and flat."""
    import struct

    t = Terrain.generate("grassland", 120, seed=1)
    mesh = build_terrain_mesh(t, subdivisions=16)
    out = tmp_path / "t.glb"
    mesh.export(out)

    data = out.read_bytes()
    length = struct.unpack("<I", data[12:16])[0]
    gltf = json.loads(data[20:20 + length])
    attrs = gltf["meshes"][0]["primitives"][0]["attributes"]

    for required in ("POSITION", "NORMAL", "COLOR_0"):
        assert required in attrs, f"missing {required}"


# --- fog --------------------------------------------------------------------

@pytest.mark.parametrize("size", [60, 120, 200])
def test_fog_stays_readable_at_every_scene_size(size):
    """FogExp2 attenuates by exp(-(density*distance)^2), so a density tuned
    for a small scene whites out a large one.

    The original 0.0135 left a 120m scene 93% fogged -- the generated world
    was invisible. Fog must never exceed roughly half opacity at full scene
    distance.
    """
    for weather, cfg in WEATHER_FOG.items():
        density = cfg["k"] / size
        opacity = 1 - math.exp(-((density * size) ** 2))
        assert opacity < 0.55, f"{weather} at {size}m is {opacity:.0%} fogged"


def test_clear_weather_is_nearly_transparent():
    density = WEATHER_FOG["clear"]["k"] / 120
    opacity = 1 - math.exp(-((density * 120) ** 2))
    assert opacity < 0.10


def test_fog_density_scales_with_scene_size():
    """A 60m and a 200m scene should look equally hazy."""
    small = WEATHER_FOG["fog"]["k"] / 60
    large = WEATHER_FOG["fog"]["k"] / 200
    assert small > large
    o_small = 1 - math.exp(-((small * 60) ** 2))
    o_large = 1 - math.exp(-((large * 200) ** 2))
    assert abs(o_small - o_large) < 0.01


# --- viewer config ----------------------------------------------------------

@pytest.fixture
def viewer_cfg(index, tmp_path):
    spec = fallback_parser.parse("a foggy medieval village at dusk")
    resolved = resolve_scene(spec, index)
    terrain = Terrain.from_spec(spec, seed=1)
    scene = layout_scene(resolved, terrain, seed=1)
    path = write_viewer(scene, tmp_path)
    html = path.read_text(encoding="utf-8")
    return json.loads(html.split("const CFG = ")[1].split(";\n")[0])


def test_viewer_has_horizon_ground_below_terrain(viewer_cfg):
    """Without a horizon plane the terrain renders as a floating island,
    showing its unlit underside against raw sky."""
    assert "groundY" in viewer_cfg
    assert "groundColour" in viewer_cfg
    assert viewer_cfg["groundY"] < viewer_cfg["terrainMax"]


def test_viewer_matches_scene_lighting(viewer_cfg):
    """Light colours are desaturated on the way out, so compare against the
    processed value rather than the raw preset."""
    from src.viewer import _desaturate

    assert viewer_cfg["sun"] == _desaturate(LIGHTING["dusk"]["sun"], 0.25)
    assert viewer_cfg["fogColour"] == WEATHER_FOG["fog"]["colour"]


def test_viewer_html_is_self_contained(index, tmp_path):
    spec = fallback_parser.parse("a medieval village")
    resolved = resolve_scene(spec, index)
    scene = layout_scene(resolved, Terrain.from_spec(spec, seed=1), seed=1)
    html = write_viewer(scene, tmp_path).read_text(encoding="utf-8")

    assert "__CONFIG__" not in html      # every placeholder substituted
    assert "__ROWS__" not in html
    assert "__PROMPT__" not in html
    assert "scene_manifest.json" in html


# --- ground painting --------------------------------------------------------

def test_road_is_painted_into_the_ground():
    """Keeping a corridor clear isn't enough -- with nothing marking it,
    buildings appear to float in an empty field."""
    import numpy as np

    from src.layout import generate_path

    t = Terrain.generate("grassland", 120, seed=1)
    rng = np.random.default_rng(1)
    path = generate_path(120, rng)

    plain = build_terrain_mesh(t, subdivisions=48)
    roaded = build_terrain_mesh(t, subdivisions=48, path=path)

    a = np.asarray(plain.visual.vertex_colors)[:, :3].astype(int)
    b = np.asarray(roaded.visual.vertex_colors)[:, :3].astype(int)

    # A meaningful share of the ground should have turned earth-coloured.
    became_dirt = ((b[:, 0] > b[:, 1]) & (a[:, 1] >= a[:, 0])).sum()
    assert became_dirt > len(b) * 0.01, "no visible road"


def test_village_ground_is_worn_bare():
    """Unbroken lawn up to every doorstep is a clear tell of procedural
    placement."""
    import numpy as np

    t = Terrain.generate("grassland", 120, seed=1)
    plain = build_terrain_mesh(t, subdivisions=48)
    settled = build_terrain_mesh(t, subdivisions=48,
                                 centre=(60.0, 60.0), core_radius=30.0)

    a = np.asarray(plain.visual.vertex_colors)[:, :3].astype(float)
    b = np.asarray(settled.visual.vertex_colors)[:, :3].astype(float)

    verts = settled.vertices
    near = np.hypot(verts[:, 0] - 60, verts[:, 2] - 60) < 12
    far = np.hypot(verts[:, 0] - 60, verts[:, 2] - 60) > 55

    # Ground near the centre should have shifted browner than ground far out.
    shift_near = (b[near, 0] - b[near, 1]).mean()
    shift_far = (b[far, 0] - b[far, 1]).mean()
    assert shift_near > shift_far


def test_ground_colour_varies():
    """A single flat shade over 120m reads as a placeholder plane."""
    import numpy as np

    t = Terrain.generate("grassland", 120, seed=1)
    mesh = build_terrain_mesh(t, subdivisions=48)
    c = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(float)
    assert c.std(axis=0).mean() > 2.0, "ground colour is uniform"


def test_terrain_painting_is_deterministic():
    t = Terrain.generate("grassland", 120, seed=4)
    import numpy as np
    a = np.asarray(build_terrain_mesh(t, 32).visual.vertex_colors)
    b = np.asarray(build_terrain_mesh(t, 32).visual.vertex_colors)
    assert np.array_equal(a, b)


# --- instanced export (materials preserved) ---------------------------------

def _tiny_asset(tmp_path, name, colour):
    """A model with a real material, so we can prove it survives export."""
    import numpy as np
    import trimesh

    m = trimesh.creation.box(extents=(1, 1, 1))
    m.visual = trimesh.visual.ColorVisuals(
        mesh=m, vertex_colors=np.tile([*colour, 255], (len(m.vertices), 1)))
    path = tmp_path / f"{name}.glb"
    m.export(path)
    return path


@pytest.fixture
def instanced(index, tmp_path):
    from src.export_scene import export_instanced
    from src.layout import PlacedInstance, PlacedScene

    src = _tiny_asset(tmp_path, "hut", (200, 120, 90))
    terrain = Terrain.generate("grassland", 120, seed=1)

    scene = PlacedScene(terrain=terrain, seed=1)
    scene.path = [(10.0, 10.0), (60.0, 60.0)]
    scene.centre = (60.0, 60.0)
    scene.core_radius = 30.0
    for i in range(5):
        scene.instances.append(PlacedInstance(
            name="house", category="building", asset_name="hut",
            asset_file=str(src), position=(20.0 + i * 8, 1.0, 40.0),
            rotation_y=0.3 * i, scale=2.0, radius=2.0, triangles=12))

    out = tmp_path / "out"
    report = export_instanced(scene, out)
    return out, report


def test_instanced_export_stores_each_model_once(instanced):
    """Five houses must not mean five copies of the mesh."""
    out, report = instanced
    assert report["unique_meshes"] == 1
    assert report["instances_exported"] == 5
    assert len(list((out / "assets").glob("*.glb"))) == 1


def test_original_materials_survive_export(instanced):
    """The whole point: baking flattens each model to one colour and loses
    the pack's textures. Copying the file keeps them."""
    import json

    import numpy as np
    import trimesh

    out, _ = instanced
    manifest = json.loads((out / "scene_manifest.json").read_text())
    asset = manifest["assets"][0]

    loaded = trimesh.load(out / asset["file"])
    geom = (list(loaded.geometry.values())[0]
            if hasattr(loaded, "geometry") else loaded)
    colour = np.asarray(geom.visual.vertex_colors)[0][:3]

    assert tuple(int(v) for v in colour) == (200, 120, 90)


def test_manifest_carries_every_instance_transform(instanced):
    import json

    out, _ = instanced
    manifest = json.loads((out / "scene_manifest.json").read_text())

    assert len(manifest["instances"]) == 5
    for inst in manifest["instances"]:
        assert len(inst["p"]) == 3
        assert "r" in inst and "s" in inst
        assert inst["asset"] == manifest["assets"][0]["file"]


def test_manifest_records_base_offset(instanced):
    """Packs disagree about whether the origin is at the centre or the foot;
    the viewer needs the offset to seat models on the ground."""
    import json

    out, _ = instanced
    manifest = json.loads((out / "scene_manifest.json").read_text())
    assert "base_offset" in manifest["assets"][0]


def test_terrain_exported_separately(instanced):
    out, _ = instanced
    assert (out / "terrain.glb").exists()


def test_instanced_output_smaller_than_baked(index, tmp_path):
    """Storing one copy of each model should beat merging hundreds."""
    from src.export_gltf import export_glb
    from src.export_scene import export_instanced
    from src.layout import PlacedInstance, PlacedScene

    src = _tiny_asset(tmp_path, "tree", (60, 140, 70))
    terrain = Terrain.generate("grassland", 120, seed=1)

    def make():
        s = PlacedScene(terrain=terrain, seed=1)
        s.path = [(10.0, 10.0), (60.0, 60.0)]
        for i in range(200):
            s.instances.append(PlacedInstance(
                name="tree", category="vegetation", asset_name="tree",
                asset_file=str(src), position=(float(i % 100), 1.0, 30.0),
                rotation_y=0.0, scale=1.0, radius=0.5, triangles=12))
        return s

    inst_report = export_instanced(make(), tmp_path / "inst")
    baked_report = export_glb(make(), tmp_path / "baked" / "scene.glb")

    assert inst_report["size_mb"] <= baked_report["size_mb"]


# --- ambient light colour ---------------------------------------------------

def test_hemisphere_light_is_desaturated():
    """Sky colours are chosen to look right as sky, which means saturated.

    Used directly as the hemisphere light's upward colour they tint every
    surface: strongly blue sky over green foliage renders cyan. Trees came out
    turquoise under the day and night skies, while the dusk scene -- whose sky
    top is nearly grey -- looked correct.
    """
    from src.viewer import LIGHTING, _desaturate

    for tod, cfg in LIGHTING.items():
        raw = cfg["sky_top"].lstrip("#")
        lit = _desaturate(cfg["sky_top"]).lstrip("#")

        raw_rgb = [int(raw[i:i + 2], 16) for i in (0, 2, 4)]
        lit_rgb = [int(lit[i:i + 2], 16) for i in (0, 2, 4)]

        raw_spread = max(raw_rgb) - min(raw_rgb)
        lit_spread = max(lit_rgb) - min(lit_rgb)

        assert lit_spread < raw_spread, f"{tod} not desaturated"
        assert min(lit_rgb) > 100, f"{tod} ambient too dark to light a scene"


def test_viewer_passes_separate_sky_and_light_colours(viewer_cfg):
    """The sky keeps its own saturated colour; only the light is softened."""
    assert viewer_cfg["skyTop"] != viewer_cfg["skyLight"]


def test_night_is_bright_enough_to_see():
    """A physically plausible night render is nearly black, which shows
    nothing. Games solve this with a bright artificial moon."""
    from src.viewer import LIGHTING, TIME_EXPOSURE

    night = LIGHTING["night"]
    day = LIGHTING["day"]

    assert night["ambient_i"] > 0.8
    assert night["intensity"] > 1.0
    assert TIME_EXPOSURE["night"] > TIME_EXPOSURE["day"]
    assert night["intensity"] < day["intensity"]      # still reads as night


def _rgb(hex_colour):
    h = hex_colour.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


@pytest.mark.parametrize("tod", ["dawn", "day", "dusk", "night"])
def test_combined_light_is_close_to_neutral(tod):
    """Every light in the scene pulling towards blue stains every surface.

    A night scene lit by a blue moon, a blue sky and a blue ground bounce
    turned green foliage teal and orange tents pink. The scene should read as
    night from its darkness and sky colour, not from tinting the world.
    """
    from src.viewer import LIGHTING, _desaturate

    cfg = LIGHTING[tod]
    contributions = [
        (_desaturate(cfg["sun"], 0.25), cfg["intensity"]),
        (_desaturate(cfg["sky_top"]), cfg["ambient_i"]),
        (_desaturate(cfg["ambient"], 0.45), cfg["ambient_i"]),
    ]

    total = [0.0, 0.0, 0.0]
    for colour, weight in contributions:
        for i, v in enumerate(_rgb(colour)):
            total[i] += v * weight

    peak = max(total)
    norm = [t / peak for t in total]

    # No channel may dominate another strongly enough to shift hues.
    assert max(norm) - min(norm) < 0.30, f"{tod} light is strongly tinted"

    # Blue specifically must not overpower green, which is what produced
    # teal foliage.
    assert norm[2] - norm[1] < 0.15, f"{tod} is blue-biased over green"


def test_warm_and_cool_times_still_differ():
    """Neutralising the tint must not make every time of day identical."""
    from src.viewer import LIGHTING, _desaturate

    dusk = _rgb(_desaturate(LIGHTING["dusk"]["sun"], 0.25))
    night = _rgb(_desaturate(LIGHTING["night"]["sun"], 0.25))

    assert dusk[0] - dusk[2] > 40, "dusk sun should stay warm"
    assert night[2] >= night[0], "night moon should stay cool"


# --- emissive light sources -------------------------------------------------

def _cfg_for(index, prompt, tmp_path):
    import json

    from src.asset_resolution import resolve_scene
    from src.layout import layout_scene
    from src.viewer import write_viewer

    spec = fallback_parser.parse(prompt)
    scene = layout_scene(resolve_scene(spec, index),
                         Terrain.from_spec(spec, seed=1), seed=1)
    html = write_viewer(scene, tmp_path).read_text(encoding="utf-8")
    return json.loads(html.split("const CFG = ")[1].split(";\n")[0])


def test_dark_scenes_emit_light_from_their_lamps(index, tmp_path):
    """Torches and campfires were placed as geometry that emitted no light,
    so a night scene had lanterns everywhere and no glow."""
    cfg = _cfg_for(index, "a forest camp at night with a campfire", tmp_path)
    assert cfg["lightsEmit"] is True
    assert cfg["maxPointLights"] > 0
    assert cfg["lightIntensity"] > 0


def test_daylight_scenes_skip_point_lights(index, tmp_path):
    """Dynamic lights cost frames and show nothing under a midday sun."""
    cfg = _cfg_for(index, "a village on a sunny day", tmp_path)
    assert cfg["lightsEmit"] is False


def test_point_light_count_is_capped(index, tmp_path):
    """Hundreds of dynamic lights will not render interactively."""
    cfg = _cfg_for(index, "a forest camp at night", tmp_path)
    assert cfg["maxPointLights"] <= 20


def test_lamp_colour_is_warm(index, tmp_path):
    cfg = _cfg_for(index, "a village at night", tmp_path)
    h = cfg["lightColour"].lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    assert r > b, "firelight should be warm"
