"""Tests for prompts outside the settings the system was built around.

Before this work "a snowy village", "a pirate cove" and "a swamp hut" all came
out as the same green medieval village: the theme matched, the ground and the
water did not, and nothing warned. These tests pin down the pieces that fixed
it -- new ground types, the coast, the road stopping at the shore, and
leaving out what the library cannot supply rather than faking it.
"""

import math

import numpy as np
import pytest

from src import fallback_parser, vocab
from src.crossings import choose_road, trim_to_land, add_bridge_instances
from src.export_gltf import build_water_mesh
from src.layout import generate_path
from src.schema import SceneSpec, TerrainFeature
from src.terrain import Terrain


# --- the ground follows the prompt ------------------------------------------

@pytest.mark.parametrize("prompt,terrain", [
    ("a snowy village in winter", "snow"),
    ("a witch's hut in a swamp", "swamp"),
    ("a camp on a sandy beach", "beach_sand"),
    ("a volcanic wasteland", "barren_rock"),
])
def test_biome_comes_from_the_prompt(prompt, terrain):
    assert fallback_parser.parse(prompt).terrain.type == terrain


@pytest.mark.parametrize("prompt,feature", [
    ("a fishing harbour town", "coast"),
    ("a pirate cove", "coast"),
    ("a village by the sea", "coast"),
])
def test_sea_words_make_a_coast(prompt, feature):
    assert feature in [f.type for f in fallback_parser.parse(prompt).features]


def test_snow_weather_turns_grass_to_snow():
    spec = SceneSpec.model_validate({
        "terrain": {"type": "grassland"},
        "lighting": {"weather": "snow"},
    })
    assert spec.terrain.type == "snow"


def test_weather_does_not_override_a_chosen_ground():
    """Snow falling on a desert is the prompt's call, not ours to undo."""
    spec = SceneSpec.model_validate({
        "terrain": {"type": "desert_sand"},
        "lighting": {"weather": "snow"},
    })
    assert spec.terrain.type == "desert_sand"


@pytest.mark.parametrize("ground,feature", list(vocab.BIOME_FEATURES.items()))
def test_wet_biomes_bring_their_water(ground, feature):
    spec = SceneSpec.model_validate({"terrain": {"type": ground}})
    assert feature in [f.type for f in spec.features]


def test_biome_water_is_not_added_twice():
    spec = SceneSpec.model_validate({
        "terrain": {"type": "swamp"},
        "features": [{"type": "river"}],
    })
    assert sum(f.holds_water for f in spec.features) == 1


@pytest.mark.parametrize("ground", ["snow", "swamp", "beach_sand"])
def test_new_grounds_generate(ground):
    t = Terrain.generate(ground, 120, seed=3)
    assert np.isfinite(t.heights).all()
    assert t.heights.max() > t.heights.min()


# --- the coast ---------------------------------------------------------------

def _coast(seed=4):
    t = Terrain.generate("beach_sand", 120, seed=seed)
    t.apply_features([TerrainFeature(type="coast")], seed=seed)
    return t


@pytest.mark.parametrize("seed", range(4))
def test_coast_floods_one_side_only(seed):
    t = _coast(seed)
    assert t.features.coasts, "no coast recorded"
    fraction = t.features.water_fraction
    assert 0.10 < fraction < 0.45, f"sea covers {fraction:.0%}"


@pytest.mark.parametrize("seed", range(4))
def test_beach_behind_the_shore_is_dry(seed):
    t = _coast(seed)
    edge, line, _depth = t.features.coasts[0]
    # Step inland from each shoreline point and check the ground is above
    # the water there.
    inward = {"north": (0, 1), "south": (0, -1),
              "west": (1, 0), "east": (-1, 0)}[edge]
    dry = 0
    for x, z in line[4:-4]:
        px, pz = x + inward[0] * 4, z + inward[1] * 4
        if 0 < px < t.size and 0 < pz < t.size:
            dry += t.height_at(px, pz) > t.water_level
    assert dry >= 0.9 * len(line[4:-4])


def test_coast_has_a_water_surface():
    mesh = build_water_mesh(_coast())
    assert mesh is not None and len(mesh.faces) > 0
    # Water is seen from above: every face normal must point up.
    assert (mesh.face_normals[:, 1] > 0).all()


# --- roads and the sea --------------------------------------------------------

def test_trim_drops_only_the_wet_ends():
    t = _coast()
    rng = np.random.default_rng(0)
    path = generate_path(t.size, rng)
    trimmed = trim_to_land(t, path)
    assert trimmed
    assert not t.is_water(*trimmed[0], margin=1.0)
    assert not t.is_water(*trimmed[-1], margin=1.0)
    # Contiguous slice of the original, nothing reordered.
    first = path.index(trimmed[0])
    assert path[first:first + len(trimmed)] == trimmed


def test_trim_leaves_a_dry_road_alone():
    t = Terrain.generate("grassland", 120, seed=1)
    path = [(10.0, 10.0), (60.0, 60.0), (110.0, 110.0)]
    assert trim_to_land(t, path) == path


@pytest.mark.parametrize("seed", range(4))
def test_coastal_road_does_not_bridge_the_sea(seed):
    t = _coast(seed)
    rng = np.random.default_rng(seed)
    road = choose_road(t, t.size, rng, generate_path)
    assert len(road) >= 12
    wet = sum(t.is_water(x, z, margin=0.5) for x, z in road)
    assert wet <= 2, f"{wet} road samples out at sea"


# --- leaving things out -------------------------------------------------------

def test_hopeless_matches_are_skipped(index):
    """A request the library has nothing for is reported, not faked."""
    from src.graph_pipeline import resolve_groups
    from src.scene_graph import Group, SceneGraph

    graph = SceneGraph(groups=[
        Group(id="thing", asset="zzqx frobnicator", count=3),
    ])
    resolved, warnings = resolve_groups(graph, index)
    if "thing" in resolved:
        pytest.skip("fixture library happens to match the nonsense query")
    assert any("skipped" in w or "no model" in w for w in warnings)


def test_no_bridge_model_means_a_ford(index):
    """A crossing with no bridge in the library stays walkable, unmodelled."""
    from types import SimpleNamespace
    from src.crossings import Crossing

    t = Terrain.generate("grassland", 120, seed=2)
    t.apply_features([TerrainFeature(type="river")], seed=2)
    t.features.bridges = [Crossing((50, 40), (50, 70), 2.0)]
    scene = SimpleNamespace(terrain=t, instances=[], warnings=[])

    class NoBridges:
        def best(self, *a, **k):
            return None

    assert add_bridge_instances(scene, NoBridges()) == 0
    assert scene.warnings and "ford" in scene.warnings[0]
    assert t.on_bridge(50, 55)


def test_short_span_gets_one_fitted_segment(index):
    """A long model over a narrow stream is shrunk to the span."""
    from types import SimpleNamespace
    from src.crossings import Crossing

    t = Terrain.generate("grassland", 120, seed=2)
    t.apply_features([TerrainFeature(type="river")], seed=2)
    t.features.bridges = [Crossing((50, 50), (50, 56), 2.0)]
    scene = SimpleNamespace(terrain=t, instances=[], warnings=[])
    n = add_bridge_instances(scene, index)
    if n == 0:
        pytest.skip("fixture library has no bridge")
    assert n == 1
    seg = scene.instances[0]
    assert seg.radius * 2 <= 6 * 1.3
