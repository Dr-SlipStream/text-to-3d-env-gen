"""Tests for filling out a scene graph.

A graph says what a place *is*. It never mentions ground cover, undergrowth or
scattered stones, because no description does -- and without them a scene is
models on a field.
"""

import math

import pytest

from src import fallback_parser, vocab
from src.graph_builder import compile_graph
from src.graph_dressing import (DEFAULT_TARGET, TARGET_DENSITY, add_ground_cover,
                                add_variation, dress, normalise_density)
from src.graph_pipeline import build_scene
from src.layout import validate
from src.scene_graph import Group, Region, Relation, SceneGraph
from src.terrain import Terrain


def simple_graph():
    return SceneGraph(
        regions=[Region(id="camp", kind="settlement", size="medium"),
                 Region(id="woods", kind="wilderness", size="large")],
        groups=[
            Group(id="fire", asset="campfire", count=1,
                  relations=[Relation(kind="at_center_of", target="camp")]),
            Group(id="trees", asset="pine tree", density=1.5,
                  relations=[Relation(kind="scatter_in", target="woods")]),
        ])


# --- ground cover -----------------------------------------------------------

def test_ground_cover_is_added():
    graph = simple_graph()
    before = len(graph.groups)
    assert add_ground_cover(graph, "forest_floor") > 0
    assert len(graph.groups) > before


def test_ground_cover_goes_in_an_open_region():
    graph = simple_graph()
    add_ground_cover(graph, "grassland")

    cover = [g for g in graph.groups if g.id.startswith("cover_")]
    assert cover
    for group in cover:
        assert group.placement.kind == "scatter_in"
        assert group.placement.target == "woods"


def test_ground_cover_invents_a_region_if_there_is_none():
    """It has to go somewhere."""
    graph = SceneGraph(
        regions=[Region(id="camp", kind="settlement")],
        groups=[Group(id="fire", asset="campfire", count=1)])
    add_ground_cover(graph, "grassland")
    assert any(r.kind == "wilderness" for r in graph.regions)


def test_ground_cover_does_not_duplicate_what_is_already_there():
    graph = SceneGraph(
        regions=[Region(id="woods", kind="wilderness")],
        groups=[Group(id="g", asset="grass", density=1.0,
                      relations=[Relation(kind="scatter_in",
                                          target="woods")])])
    add_ground_cover(graph, "grassland")
    grass = [g for g in graph.groups if g.asset == "grass"]
    assert len(grass) == 1


# --- variation --------------------------------------------------------------

def test_scattered_populations_get_siblings():
    """A hundred copies of one tree is the clearest sign of procedural work."""
    graph = simple_graph()
    assert add_variation(graph) > 0

    trees = [g for g in graph.groups if "tree" in g.asset]
    assert len(trees) > 1


def test_variation_splits_density_rather_than_adding_to_it():
    """Variety should not also mean more objects."""
    graph = simple_graph()
    before = sum(g.density for g in graph.groups if g.density)
    add_variation(graph)
    after = sum(g.density for g in graph.groups if g.density)
    assert abs(before - after) < 0.01


def test_fixed_counts_are_not_varied():
    """One campfire stays one campfire."""
    graph = simple_graph()
    add_variation(graph)
    fires = [g for g in graph.groups if "campfire" in g.asset]
    assert len(fires) == 1


def test_variants_are_not_themselves_varied():
    graph = simple_graph()
    add_variation(graph)
    n = len(graph.groups)
    add_variation(graph)
    assert len(graph.groups) == n, "variants of variants"


def test_generated_ids_are_unique_and_readable():
    """Truncating to a fixed width produced the same id twice, and the graph
    then dropped one as a duplicate."""
    graph = simple_graph()
    dress(graph, fallback_parser.parse("a forest camp"), 200.0)

    ids = [g.id for g in graph.groups]
    assert len(ids) == len(set(ids)), "duplicate group ids"
    for gid in ids:
        assert gid and not gid.endswith("_")


# --- density normalisation --------------------------------------------------

def test_density_is_scaled_towards_a_target():
    graph = simple_graph()
    before = sum(g.density for g in graph.groups if g.density)
    normalise_density(graph, "forest_camp", 200.0)
    after = sum(g.density for g in graph.groups if g.density)
    assert after != before


def test_fixed_counts_survive_normalisation():
    """The prompt's own content is not up for rescaling."""
    graph = simple_graph()
    normalise_density(graph, "forest_camp", 200.0)
    assert graph.group("fire").count == 1


def test_normalisation_is_a_correction_not_a_rewrite():
    """A deliberately sparse graph stays sparser than a dense one."""
    sparse = simple_graph()
    sparse.groups[1].density = 0.2
    dense = simple_graph()
    dense.groups[1].density = 6.0

    normalise_density(sparse, "forest_camp", 200.0)
    normalise_density(dense, "forest_camp", 200.0)

    assert dense.groups[1].density > sparse.groups[1].density


@pytest.mark.parametrize("theme", list(TARGET_DENSITY))
def test_every_theme_has_a_target(theme):
    assert 20 < TARGET_DENSITY[theme] < 100


# --- the whole thing --------------------------------------------------------

@pytest.mark.parametrize("prompt", [
    "a forest camp around a campfire beside a river",
    "a medieval village on a sunny day",
    "a desert outpost around an oasis",
    "a sci-fi base on a hill",
])
def test_dressed_graphs_fill_the_scene(index, prompt):
    """Undressed graphs came out at a third of the density of the rule-based
    path, because nothing asked for the small stuff."""
    spec = fallback_parser.parse(prompt)
    spec.terrain.size = "large"
    size = vocab.TERRAIN_SIZE_METRES["large"]

    graph = compile_graph(spec)
    dress(graph, spec, size)

    terrain = Terrain.from_spec(spec, seed=5)
    terrain.apply_features(spec.features, seed=5)
    scene = build_scene(graph, terrain, index, spec=spec, seed=5)

    per_1000 = 1000 * len(scene.instances) / (size * size)
    assert per_1000 > 20, f"{prompt}: only {per_1000:.0f} per 1000m2"
    assert validate(scene)["valid"]


def test_dressing_keeps_density_consistent_across_themes(index):
    """One prompt producing a crowd and the next an empty field is the thing
    normalisation exists to prevent."""
    densities = []
    for prompt in ["a forest camp with a river", "a medieval village",
                   "a village beside a lake"]:
        spec = fallback_parser.parse(prompt)
        spec.terrain.size = "large"
        size = vocab.TERRAIN_SIZE_METRES["large"]

        graph = compile_graph(spec)
        dress(graph, spec, size)
        terrain = Terrain.from_spec(spec, seed=5)
        terrain.apply_features(spec.features, seed=5)
        scene = build_scene(graph, terrain, index, spec=spec, seed=5)
        densities.append(1000 * len(scene.instances) / (size * size))

    assert max(densities) / min(densities) < 2.5


def test_disabling_dressing_leaves_the_graph_alone():
    graph = simple_graph()
    before = len(graph.groups)
    result = dress(graph, fallback_parser.parse("a forest camp"), 200.0,
                   density=0)
    assert len(graph.groups) == before
    assert result["cover"] == 0


# --- soft edges -------------------------------------------------------------

def test_scatter_thins_towards_a_settlement_rather_than_stopping(index):
    """A forest that ends on a hard circle reads as a cut-out."""
    spec = fallback_parser.parse("a forest camp around a campfire")
    spec.terrain.size = "large"
    size = vocab.TERRAIN_SIZE_METRES["large"]

    graph = compile_graph(spec)
    dress(graph, spec, size)
    terrain = Terrain.from_spec(spec, seed=5)
    terrain.apply_features(spec.features, seed=5)
    scene = build_scene(graph, terrain, index, spec=spec, seed=5)

    cx, cz = scene.centre
    radius = scene.core_radius

    inner = outer = 0        # just inside the boundary, and just outside
    for inst in scene.instances:
        if inst.category != "vegetation":
            continue
        d = math.hypot(inst.position[0] - cx, inst.position[2] - cz) / radius
        if 0.75 <= d < 1.0:
            inner += 1
        elif 1.0 <= d < 1.25:
            outer += 1

    assert inner > 0, "nothing at all inside the boundary: a hard edge"
    assert outer > inner, "no thinning towards the settlement"


# --- natural distribution ---------------------------------------------------

def test_density_field_has_stands_and_clearings():
    """An even sprinkle of trees is the clearest sign of a generated forest."""
    import numpy as np

    from src.solver import DensityField

    field = DensityField(200, seed=42).field.ravel()
    assert field.min() >= 0.2, "nowhere should be completely bare"
    assert (field < 0.4).mean() > 0.15, "no clearings"
    assert (field > 0.7).mean() > 0.05, "no dense stands"


def test_each_population_gets_its_own_pattern():
    """Trees and undergrowth thickening in exactly the same places would look
    just as artificial as an even spread."""
    import numpy as np

    from src.solver import DensityField

    a = DensityField(200, seed=42).field.ravel()
    b = DensityField(200, seed=99).field.ravel()
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.5


def test_density_field_is_stable_for_a_group():
    import numpy as np

    from src.solver import DensityField

    a = DensityField(200, seed=7).field
    b = DensityField(200, seed=7).field
    assert np.array_equal(a, b)


def test_scattered_vegetation_actually_clusters(index):
    """Clump anchors alone were not enough: a rejected proposal re-picks a
    random anchor, so over sixty attempts the result averages back out to
    uniform. The density field is what makes the clustering survive.
    """
    import collections

    import numpy as np

    from src.graph_builder import compile_graph

    spec = fallback_parser.parse("a forest camp with many trees")
    spec.terrain.size = "large"
    size = vocab.TERRAIN_SIZE_METRES["large"]

    graph = compile_graph(spec)
    dress(graph, spec, size)
    terrain = Terrain.from_spec(spec, seed=5)
    terrain.apply_features(spec.features, seed=5)
    scene = build_scene(graph, terrain, index, spec=spec, seed=5)

    cells = collections.Counter()
    n = 10
    for inst in scene.instances:
        if inst.category != "vegetation":
            continue
        cells[(int(inst.position[0] / size * n),
               int(inst.position[2] / size * n))] += 1

    counts = np.array([cells.get((i, j), 0)
                       for i in range(n) for j in range(n)])
    assert counts.sum() > 200, "not enough vegetation to judge"

    # Uniform scatter lands near 0.2; clumped distributions run well above.
    variation = counts.std() / max(counts.mean(), 1)
    assert variation > 0.45, f"vegetation is evenly spread ({variation:.2f})"


def test_ground_cover_uses_small_things(index):
    """"small rock" resolves to a 1.4m boulder, and a field of those reads as
    litter rather than as ground."""
    from src.graph_dressing import DEFAULT_COVER, GROUND_COVER

    for cover in list(GROUND_COVER.values()) + [DEFAULT_COVER]:
        for asset, _density in cover:
            assert "small rock" not in asset, \
                f"{asset!r} is a boulder, not ground cover"
