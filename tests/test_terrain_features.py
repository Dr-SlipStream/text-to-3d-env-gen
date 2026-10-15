"""Tests for landscape features: rivers, lakes, hills, valleys, clearings.

A prompt naming "a forest with a river" used to parse as a forest and drop
the river in silence -- the theme matched, so nothing even warned.
"""

import numpy as np
import pytest

from src import fallback_parser, vocab
from src.asset_resolution import resolve_scene
from src.export_gltf import build_water_mesh
from src.layout import layout_scene, validate
from src.schema import SceneSpec, TerrainFeature
from src.terrain import Terrain


# --- parsing ----------------------------------------------------------------

@pytest.mark.parametrize("prompt,expected", [
    ("a forest with a river", "river"),
    ("a camp beside a stream", "river"),
    ("a village next to a lake", "lake"),
    ("an outpost around an oasis", "lake"),
    ("a village on a hill", "hill"),
    ("a settlement in a valley", "valley"),
    ("a clearing in the woods", "clearing"),
])
def test_features_are_detected(prompt, expected):
    spec = fallback_parser.parse(prompt)
    assert expected in [f.type for f in spec.features], \
        f"{prompt!r} did not yield a {expected}"


def test_prompts_without_features_get_none():
    spec = fallback_parser.parse("a medieval village on a sunny day")
    assert spec.features == []


def test_feature_size_follows_adjectives():
    assert fallback_parser.parse(
        "a camp by a narrow stream").features[0].size == "small"
    assert fallback_parser.parse(
        "a village beside a large lake").features[0].size == "large"


def test_duplicate_features_are_merged():
    """Two rivers in one scene would leave nowhere to build."""
    spec = SceneSpec(features=[TerrainFeature(type="river"),
                               TerrainFeature(type="stream"),
                               TerrainFeature(type="creek")])
    assert len(spec.features) == 1


def test_feature_count_is_capped():
    spec = SceneSpec(features=[TerrainFeature(type=t)
                               for t in vocab.TERRAIN_FEATURES])
    assert len(spec.features) <= 3


# --- carving ----------------------------------------------------------------

@pytest.mark.parametrize("feature", vocab.TERRAIN_FEATURES)
def test_every_feature_changes_the_terrain(feature):
    a = Terrain.generate("grassland", 120, seed=5)
    b = Terrain.generate("grassland", 120, seed=5)
    b.apply_features([TerrainFeature(type=feature)], seed=5)

    assert not np.array_equal(a.heights, b.heights), \
        f"{feature} left the terrain untouched"
    assert np.isfinite(b.heights).all()


def test_rivers_and_lakes_hold_water():
    for feature in ("river", "lake"):
        t = Terrain.generate("grassland", 120, seed=5)
        t.apply_features([TerrainFeature(type=feature)], seed=5)
        assert t.water_level is not None
        assert t.features.has_water


def test_dry_features_hold_no_water():
    for feature in ("hill", "valley", "clearing"):
        t = Terrain.generate("grassland", 120, seed=5)
        t.apply_features([TerrainFeature(type=feature)], seed=5)
        assert t.water_level is None, f"{feature} should be dry"


def test_water_covers_a_plausible_share_of_the_scene():
    """Deriving water from "anything below the water level" marked 42% of an
    untouched grassland as submerged, because the level is a percentile of
    terrain height and much natural ground sits below it.
    """
    for feature, ceiling in (("river", 0.30), ("lake", 0.20)):
        t = Terrain.generate("grassland", 120, seed=5)
        t.apply_features([TerrainFeature(type=feature)], seed=5)
        fraction = t.features.water_fraction
        assert 0.01 < fraction < ceiling, \
            f"{feature} covers {fraction:.0%} of the terrain"


def test_untouched_terrain_reports_no_water():
    t = Terrain.generate("grassland", 120, seed=5)
    assert t.water_level is None
    assert not t.is_water(60, 60)


def test_hill_raises_and_valley_lowers():
    base = Terrain.generate("grassland", 120, seed=5)

    hill = Terrain.generate("grassland", 120, seed=5)
    hill.apply_features([TerrainFeature(type="hill")], seed=5)
    assert hill.heights.max() > base.heights.max()

    valley = Terrain.generate("grassland", 120, seed=5)
    valley.apply_features([TerrainFeature(type="valley")], seed=5)
    assert valley.heights.min() < base.heights.min()


def test_carving_is_deterministic():
    def build():
        t = Terrain.generate("grassland", 120, seed=9)
        t.apply_features([TerrainFeature(type="river"),
                          TerrainFeature(type="hill")], seed=9)
        return t

    a, b = build(), build()
    assert np.array_equal(a.heights, b.heights)
    assert np.array_equal(a.features.water_mask, b.features.water_mask)


def test_hills_are_carved_before_water():
    """Cutting a river first and then raising a hill through it dams the
    river, so the order is fixed rather than whatever the prompt listed."""
    t = Terrain.generate("grassland", 120, seed=3)
    t.apply_features([TerrainFeature(type="river"),
                      TerrainFeature(type="hill")], seed=3)

    # The river bed must still be the lowest ground.
    mask = t.features.water_mask
    assert t.heights[mask].min() <= t.heights[~mask].min() + 0.5


# --- placement respects water -----------------------------------------------

@pytest.fixture
def watery(index):
    from src.dressing import dress_spec

    spec = fallback_parser.parse("a forest camp with a river")
    size = vocab.TERRAIN_SIZE_METRES[spec.terrain.size]
    spec, _ = dress_spec(spec, size)

    terrain = Terrain.from_spec(spec, seed=5)
    terrain.apply_features(spec.features, seed=5)
    scene = layout_scene(resolve_scene(spec, index), terrain, seed=5)
    return scene, terrain


def test_nothing_is_placed_in_water(watery):
    scene, terrain = watery
    drowned = [i.name for i in scene.instances
               if terrain.is_water(i.position[0], i.position[2])]
    assert not drowned, f"{len(drowned)} objects in the water"


def test_buildings_stand_back_from_the_bank(watery):
    """A building on the very edge of a river looks precarious."""
    scene, terrain = watery
    for inst in scene.instances:
        if inst.category != "building":
            continue
        assert not terrain.is_water(inst.position[0], inst.position[2],
                                    margin=4.0)


def test_scene_with_water_is_still_valid(watery):
    scene, _ = watery
    assert validate(scene)["valid"]
    assert len(scene.instances) > 100, "water left too little usable ground"


def test_the_road_avoids_running_down_the_riverbed(watery):
    scene, terrain = watery
    wet = sum(1 for x, z in scene.path if terrain.is_water(x, z))
    assert wet < len(scene.path) * 0.25, \
        "the road spends too long in the water"


def test_settlement_is_not_founded_in_a_lake(index):
    from src.dressing import dress_spec

    spec = fallback_parser.parse("a village beside a large lake")
    size = vocab.TERRAIN_SIZE_METRES[spec.terrain.size]
    spec, _ = dress_spec(spec, size)

    terrain = Terrain.from_spec(spec, seed=11)
    terrain.apply_features(spec.features, seed=11)
    scene = layout_scene(resolve_scene(spec, index), terrain, seed=11)

    assert scene.centre is not None
    assert not terrain.is_water(scene.centre[0], scene.centre[1])


# --- water mesh -------------------------------------------------------------

def test_water_mesh_is_built_for_wet_scenes():
    t = Terrain.generate("grassland", 120, seed=5)
    t.apply_features([TerrainFeature(type="river")], seed=5)

    mesh = build_water_mesh(t)
    assert mesh is not None
    assert len(mesh.faces) > 0
    # A flat surface at one level.
    ys = mesh.vertices[:, 1]
    assert abs(ys.max() - ys.min()) < 1e-4
    assert abs(float(ys[0]) - t.water_level) < 1e-3


def test_no_water_mesh_for_dry_scenes():
    t = Terrain.generate("grassland", 120, seed=5)
    assert build_water_mesh(t) is None

    t.apply_features([TerrainFeature(type="hill")], seed=5)
    assert build_water_mesh(t) is None


def test_water_mesh_stays_inside_the_terrain():
    t = Terrain.generate("grassland", 120, seed=5)
    t.apply_features([TerrainFeature(type="lake")], seed=5)
    mesh = build_water_mesh(t)

    lo, hi = mesh.bounds
    assert lo[0] >= -0.01 and hi[0] <= t.size + 0.01
    assert lo[2] >= -0.01 and hi[2] <= t.size + 0.01


# --- water surface quality --------------------------------------------------

def test_water_surface_is_built_from_shapes_not_grid_cells():
    """A cell-based surface has a staircase edge -- at 128 resolution on a
    200m scene that is a 1.6m step, and it is the first thing the eye catches
    along a bank. Building from the centre-line gives a smooth edge and far
    fewer triangles.
    """
    t = Terrain.generate("grassland", 200, seed=5)
    t.apply_features([TerrainFeature(type="river")], seed=5)
    mesh = build_water_mesh(t)

    # A ribbon following a centre-line needs two vertices per sample; a
    # cell-based mesh over the same area needs thousands.
    assert len(mesh.faces) < 800, \
        f"{len(mesh.faces)} triangles suggests a cell-based surface"
    assert len(mesh.faces) > 20


def test_water_surface_spans_the_river():
    """The ribbon must run the length of the channel, not a fragment of it."""
    t = Terrain.generate("grassland", 200, seed=5)
    t.apply_features([TerrainFeature(type="river")], seed=5)
    mesh = build_water_mesh(t)

    lo, hi = mesh.bounds
    span = max(hi[0] - lo[0], hi[2] - lo[2])
    assert span > t.size * 0.7, "the river does not cross the scene"


def test_water_surface_has_no_degenerate_triangles():
    t = Terrain.generate("grassland", 200, seed=5)
    t.apply_features([TerrainFeature(type="river"),
                      TerrainFeature(type="lake")], seed=5)
    mesh = build_water_mesh(t)

    areas = mesh.area_faces
    assert (areas > 1e-6).all(), "degenerate triangles in the water surface"


def test_lake_surface_is_round():
    t = Terrain.generate("grassland", 200, seed=5)
    t.apply_features([TerrainFeature(type="lake")], seed=5)
    mesh = build_water_mesh(t)

    lo, hi = mesh.bounds
    width, depth = hi[0] - lo[0], hi[2] - lo[2]
    assert abs(width - depth) < width * 0.1, "lake is not circular"


def test_manifest_records_river_geometry():
    """The centre-line and width travel with the scene, so any consumer can
    rebuild the surface rather than reverse-engineering it from the mask."""
    t = Terrain.generate("grassland", 200, seed=5)
    t.apply_features([TerrainFeature(type="river")], seed=5)

    data = t.features.to_dict()
    assert data["rivers"], "no river recorded"
    river = data["rivers"][0]
    assert "centre_line" in river and "half_width" in river
    assert len(river["centre_line"]) > 10
    assert river["half_width"] > 0


def _fraction_covered(terrain, mesh, step: int = 23) -> float:
    """Share of flooded cells that actually have water drawn over them."""
    mask = terrain.features.water_mask
    res = mask.shape[0]
    cell = terrain.size / (res - 1)

    zs, xs = np.nonzero(mask)
    samples = np.column_stack([xs * cell, zs * cell])[::step]
    if len(samples) == 0:
        return 0.0

    tri = mesh.triangles[:, :, [0, 2]]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]

    def side(p, q, r):
        return ((p[..., 0] - r[..., 0]) * (q[..., 1] - r[..., 1])
                - (q[..., 0] - r[..., 0]) * (p[..., 1] - r[..., 1]))

    covered = 0
    for px, pz in samples:
        p = np.array([px, pz])
        d1, d2, d3 = side(p, a, b), side(p, b, c), side(p, c, a)
        neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
        pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
        if (~(neg & pos)).any():
            covered += 1
    return covered / len(samples)


@pytest.mark.parametrize("size,feature", [
    (200, "river"), (120, "river"), (60, "river"), (200, "lake"),
])
def test_water_surface_covers_the_carved_channel(size, feature):
    """The surface must actually sit over the water, not beside it.

    An earlier version built the ribbon as two continuous rails sharing
    vertices. The centre-line is piecewise linear, so its direction changes
    abruptly at each joint, and with an 11m half-width and points 2m apart
    the rails crossed at every bend -- the river rendered as a few thin
    slivers and X-shaped folds. Every vertex was correctly above the ground,
    so a position check passed; only coverage catches it.
    """
    t = Terrain.generate("forest_floor", size, seed=5)
    t.apply_features([TerrainFeature(type=feature, size="large")], seed=5)
    mesh = build_water_mesh(t)

    assert mesh is not None
    covered = _fraction_covered(t, mesh)
    assert covered > 0.9, f"only {covered:.0%} of the channel has water on it"


def test_water_surface_sits_above_the_riverbed():
    """Most of the surface must be above the ground, or the river renders as
    nothing.

    Not all of it: each segment's quad is extended past its ends so the joints
    between segments are covered, and those extensions can reach onto the
    bank. Where they do the surface is simply hidden inside the terrain, which
    is harmless -- so this checks the bulk, not every vertex.
    """
    t = Terrain.generate("forest_floor", 200, seed=5)
    t.apply_features([TerrainFeature(type="river")], seed=5)
    mesh = build_water_mesh(t)

    level = t.water_level
    ground = np.array([t.height_at(float(x), float(z))
                       for x, _, z in mesh.vertices])
    visible = (ground <= level).mean()
    assert visible > 0.75, \
        f"only {visible:.0%} of the water surface is above ground"


def test_water_surface_faces_upward():
    """Every triangle must wind so its normal points up.

    Getting this backwards renders the whole river invisible: the geometry is
    correct, correctly positioned, above the riverbed and covering the
    channel -- and culled as back-facing. Coverage and position tests all
    passed while nothing appeared on screen.
    """
    for feature in ("river", "lake"):
        for size in (60, 120, 200):
            t = Terrain.generate("forest_floor", size, seed=5)
            t.apply_features([TerrainFeature(type=feature)], seed=5)
            mesh = build_water_mesh(t)

            normals = mesh.face_normals
            assert (normals[:, 1] > 0).all(), (
                f"{feature} at {size}m has "
                f"{(normals[:, 1] <= 0).sum()} downward-facing triangles")


def test_water_material_is_double_sided(tmp_path):
    """Belt and braces: a viewer that culls differently still shows water."""
    import json
    import struct

    from src.export_gltf import ensure_pbr_material

    t = Terrain.generate("forest_floor", 120, seed=5)
    t.apply_features([TerrainFeature(type="river")], seed=5)
    mesh = build_water_mesh(t)

    path = tmp_path / "water.glb"
    mesh.export(path)
    ensure_pbr_material(path, roughness=0.15, double_sided=True)

    data = path.read_bytes()
    json_len = struct.unpack("<I", data[12:16])[0]
    gltf = json.loads(data[20:20 + json_len])
    assert gltf["materials"][0]["doubleSided"] is True


def test_ground_material_stays_single_sided(tmp_path):
    """Only water needs both sides; doubling the terrain wastes fill rate."""
    import json
    import struct

    from src.export_gltf import build_terrain_mesh, ensure_pbr_material

    t = Terrain.generate("grassland", 120, seed=5)
    path = tmp_path / "terrain.glb"
    build_terrain_mesh(t, subdivisions=32).export(path)
    ensure_pbr_material(path)

    data = path.read_bytes()
    json_len = struct.unpack("<I", data[12:16])[0]
    gltf = json.loads(data[20:20 + json_len])
    assert gltf["materials"][0]["doubleSided"] is False
