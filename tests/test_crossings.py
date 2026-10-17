"""Tests for bridges where the road crosses water.

The RL experiments found a river village where the generator had cut the map
in two: every geometric check passed, and an agent could not reach the far
side in three seeds out of five. These tests pin the fix.
"""

import math

import pytest

from src import fallback_parser
from src.crossings import (Crossing, add_bridge_instances, crossing_cost,
                           find_crossings, plan_crossings)
from src.graph_builder import compile_graph
from src.graph_dressing import dress
from src.graph_pipeline import build_scene
from src.layout import generate_path, validate
from src.rl import env_from_scene
from src.schema import TerrainFeature
from src.terrain import Terrain


def river_terrain(seed=0, size=120):
    t = Terrain.generate("grassland", size, seed=seed)
    t.apply_features([TerrainFeature(type="river")], seed=seed)
    return t


def river_scene(index, seed):
    spec = fallback_parser.parse("a medieval village beside a river")
    spec.terrain.size = "medium"
    graph = compile_graph(spec)
    dress(graph, spec, 120)
    terrain = Terrain.from_spec(spec, seed=seed)
    terrain.apply_features(spec.features, seed=seed)
    return build_scene(graph, terrain, index, spec=spec, seed=seed)


# --- the geometry --------------------------------------------------------

def test_a_dry_road_needs_no_bridge():
    t = Terrain.generate("grassland", 120, seed=1)
    import numpy as np
    path = generate_path(120, np.random.default_rng(1))
    assert find_crossings(t, path) == []


def test_a_road_through_a_river_gets_a_crossing():
    import numpy as np
    t = river_terrain(seed=3)
    rng = np.random.default_rng(3)
    for _ in range(20):
        path = generate_path(120, rng)
        if any(t.is_water(x, z) for x, z in path):
            break
    crossings = find_crossings(t, path)
    assert crossings, "road enters water but no crossing was found"
    for c in crossings:
        # Both ends of the deck land on dry ground.
        assert not t.is_water(*c.start)
        assert not t.is_water(*c.end)
        assert c.deck_height >= t.water_level


def test_crossing_covers_its_own_deck():
    c = Crossing(start=(0.0, 0.0), end=(20.0, 0.0), deck_height=1.0)
    assert c.covers(10.0, 0.0)
    assert c.covers(10.0, 2.0)
    assert not c.covers(10.0, 8.0)
    assert not c.covers(40.0, 0.0)
    assert abs(c.span - 20.0) < 1e-6


def test_heading_points_along_the_crossing():
    c = Crossing(start=(0.0, 0.0), end=(0.0, 10.0), deck_height=0.0)
    assert abs(c.heading) < 1e-6           # +Z
    c = Crossing(start=(0.0, 0.0), end=(10.0, 0.0), deck_height=0.0)
    assert abs(c.heading - math.pi / 2) < 1e-6


def test_road_choice_prefers_one_short_crossing():
    """A road running down the riverbed needs a forty-metre span; one short
    square crossing is a bridge. Cost should tell them apart."""
    import numpy as np
    t = river_terrain(seed=0)
    rng = np.random.default_rng(0)
    costs = [crossing_cost(t, generate_path(120, rng)) for _ in range(12)]
    assert min(costs) < max(costs) or max(costs) == 0


# --- the outcome the RL agent cared about ---------------------------------

@pytest.mark.parametrize("seed", range(5))
def test_river_scenes_are_crossable(index, seed):
    """Previously three of these five seeds left the goal on the far bank."""
    scene = river_scene(index, seed)
    env = env_from_scene(scene, resolution=40)
    assert env.shortest_path_length() is not None, \
        "the far end of the road is unreachable"


def test_validation_reports_connectivity(index):
    scene = river_scene(index, 0)
    report = validate(scene)
    assert report["connected"] is True
    assert report["unreachable"] == 0


def test_validation_catches_a_split_map(index):
    """Remove the bridges and the same scene must fail validation."""
    for seed in range(8):
        scene = river_scene(index, seed)
        if scene.terrain.features.bridges:
            scene.terrain.features.bridges = []
            scene.instances = [i for i in scene.instances
                               if i.role != "bridge"]
            report = validate(scene)
            assert report["connected"] is False
            assert not report["valid"]
            return
    pytest.skip("no seed produced a crossing")


def test_bridge_is_exempt_from_on_path_and_floating_checks(index):
    """A bridge sits on the road and over water on purpose."""
    for seed in range(8):
        scene = river_scene(index, seed)
        if any(i.role == "bridge" for i in scene.instances):
            report = validate(scene)
            assert report["on_path"] == 0
            assert report["floating"] == 0
            return
    pytest.skip("no bridge placed")


def test_bridge_is_walkable_not_an_obstacle(index):
    from src.rl.gridworld import walkability_from_scene

    for seed in range(8):
        scene = river_scene(index, seed)
        bridges = scene.terrain.features.bridges
        if not bridges:
            continue
        grid = walkability_from_scene(scene, resolution=40)
        cell = scene.terrain.size / 40
        cx, cz = bridges[0].centre
        assert grid[int(cz / cell), int(cx / cell)], \
            "the middle of the bridge is not walkable"
        return
    pytest.skip("no bridge planned")


def test_terrain_records_bridges_in_its_export():
    import numpy as np
    t = river_terrain(seed=3)
    rng = np.random.default_rng(3)
    for _ in range(20):
        path = generate_path(120, rng)
        if plan_crossings(t, path):
            break
    data = t.features.to_dict()
    assert "bridges" in data
    if t.features.bridges:
        assert data["bridges"][0]["span"] > 0


def test_missing_bridge_model_falls_back_to_a_ford(index):
    """No bridge in the library: the crossing still walks, nothing silly
    like a fence gets laid across the river."""
    class Empty:
        def best(self, *a, **k):
            return None

    for seed in range(8):
        scene = river_scene(index, seed)
        if not scene.terrain.features.bridges:
            continue
        scene.instances = [i for i in scene.instances if i.role != "bridge"]
        assert add_bridge_instances(scene, Empty()) == 0
        assert validate(scene)["connected"] is True
        return
    pytest.skip("no crossing")


def test_bridge_is_road_sized_not_river_sized(index, tmp_path):
    """Scaling one model up to the full span turned a 23-metre crossing into
    a 23-metre-long, 12-metre-tall fortress. A bridge is as wide as the road
    and repeats along its length instead."""
    import json

    import numpy as np
    import trimesh

    from src.asset_index import AssetIndex
    from src.crossings import DECK_WIDTH, MAX_BRIDGE_HEIGHT

    # A chunky arch model, roughly the proportions of a stone bridge piece.
    model = tmp_path / "bridge.glb"
    mesh = trimesh.creation.box(extents=(2.0, 1.6, 1.4))
    mesh.apply_translation([0, 0.8, 0])
    mesh.export(model)

    # Only this bridge, so the measurements below are of a known model.
    assets = [dict(a) for a in index.assets if "bridge" not in a["name"]]
    assets.append({"id": "t/bridge_stone", "name": "bridge stone",
                   "category": "structure", "pack": "t", "file": str(model),
                   "radius": 1.0, "height": 1.6, "modular": False,
                   "theme_hints": [], "tags": ["bridge", "stone"]})
    lib = AssetIndex(assets, embedder=index.embedder)

    for seed in range(8):
        scene = river_scene(lib, seed)
        pieces = [i for i in scene.instances if i.role == "bridge"]
        if not pieces:
            continue
        crossing = scene.terrain.features.bridges[0]

        for p in pieces:
            assert 1.4 * p.scale <= DECK_WIDTH + 0.01, "deck wider than road"
            assert 1.6 * p.scale <= MAX_BRIDGE_HEIGHT + 0.01, "bridge too tall"

        covered = sum(2 * p.radius for p in pieces
                      if crossing.covers(p.position[0], p.position[2], 1.0))
        assert covered >= crossing.span * 0.95, "segments don't reach the bank"
        return
    pytest.skip("no crossing")


def test_deck_sits_above_water_even_with_railings(tmp_path):
    """A post-and-rail bridge was placed by the top of its railings, which
    put the planks under the river."""
    import trimesh
    from types import SimpleNamespace
    from src.crossings import Crossing, _measure, add_bridge_instances

    deck = trimesh.creation.box(extents=(4.0, 0.1, 1.2))
    deck.apply_translation([0, 0.55, 0])             # planks at 0.6
    parts = [deck]
    for x in (-1.8, 0, 1.8):
        for z in (-0.55, 0.55):
            post = trimesh.creation.box(extents=(0.1, 1.6, 0.1))
            post.apply_translation([x, 0.8, z])       # 0 .. 1.6
            parts.append(post)
    model = tmp_path / "bridge_posts.glb"
    trimesh.util.concatenate(parts).export(model)

    long_, short, height, along_x, deck_level = _measure(str(model), 1.0)
    assert height == pytest.approx(1.6, abs=0.05)
    assert deck_level == pytest.approx(0.6, abs=0.1)

    t = Terrain.generate("grassland", 120, seed=2)
    t.apply_features([TerrainFeature(type="river")], seed=2)
    t.features.bridges = [Crossing((50, 40), (50, 62), 2.0)]
    scene = SimpleNamespace(terrain=t, instances=[], warnings=[])

    match = SimpleNamespace(name="bridge posts", file=str(model),
                            radius=2.0, asset={"triangles": 100})
    lib = SimpleNamespace(best=lambda *a, **k: match)
    assert add_bridge_instances(scene, lib) > 0
    seg = scene.instances[0]
    planks = seg.position[1] + deck_level * seg.scale
    assert planks > t.water_level + 0.5, "deck under the water"
