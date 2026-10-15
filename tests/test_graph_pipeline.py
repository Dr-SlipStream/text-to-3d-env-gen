"""Tests for getting a scene graph from a prompt and running it end to end."""

import math

import pytest

from src import fallback_parser, vocab
from src.graph_builder import compile_graph, _ensure_surroundings
from src.graph_pipeline import build_scene, describe, resolve_groups
from src.layout import validate
from src.scene_graph import Group, Region, Relation, SceneGraph
from src.terrain import Terrain


# --- compiling a graph from the flat spec -----------------------------------

def test_compiled_graph_has_regions_and_groups():
    spec = fallback_parser.parse("a medieval village on a sunny day")
    graph = compile_graph(spec)

    assert any(r.kind == "settlement" for r in graph.regions)
    assert any(r.kind == "wilderness" for r in graph.regions)
    assert len(graph.groups) >= 3


def test_water_features_become_water_regions():
    spec = fallback_parser.parse("a forest camp beside a river")
    graph = compile_graph(spec)

    water = [r for r in graph.regions if r.kind == "water"]
    assert water, "the river did not become a region"
    # and something relates to it
    targets = {rel.target for g in graph.groups for rel in g.relations}
    assert water[0].id in targets, "nothing is positioned by the river"


def test_a_single_light_becomes_the_focal_point():
    """A lone campfire is what a camp arranges itself around."""
    from src.schema import SceneObject, SceneSpec

    spec = SceneSpec(theme="forest_camp", objects=[
        SceneObject(name="campfire", category="light_source", quantity=1),
        SceneObject(name="tent", category="building", quantity=5),
    ])
    graph = compile_graph(spec)

    fire = next(g for g in graph.groups if "campfire" in g.asset)
    assert fire.count == 1
    assert fire.placement.kind == "at_center_of"

    tents = next(g for g in graph.groups if "tent" in g.asset)
    assert tents.placement.kind == "ring_around"
    assert tents.placement.target == fire.id
    assert tents.orientation.kind == "faces"


def test_vegetation_becomes_density_not_a_count():
    """Trees fill space; their number should follow the size of the scene."""
    spec = fallback_parser.parse("a forest camp")
    graph = compile_graph(spec)

    for group in graph.groups:
        if "tree" in group.asset or "bush" in group.asset:
            assert group.density is not None, f"{group.asset} has a fixed count"
            assert group.count is None


def test_every_relation_target_exists_after_compiling():
    for prompt in ["a medieval village", "a forest camp with a river",
                   "a desert outpost", "a sci-fi base"]:
        spec = fallback_parser.parse(prompt)
        graph = compile_graph(spec)
        known = {r.id for r in graph.regions} | {g.id for g in graph.groups}
        for group in graph.groups:
            for rel in group.relations:
                assert rel.target in known or rel.kind == "inside"


def test_groups_without_a_position_are_given_one():
    """A group has to go somewhere, and "anywhere" is the sensible reading."""
    graph = SceneGraph(
        regions=[Region(id="camp", kind="settlement")],
        groups=[Group(id="stuff", asset="crate", count=4)])
    _ensure_surroundings(graph)

    assert graph.groups[0].placement is not None
    assert any("no position" in w for w in graph.warnings)


# --- resolving groups to models ---------------------------------------------

def test_every_group_resolves_to_a_model(index):
    spec = fallback_parser.parse("a medieval village on a sunny day")
    graph = compile_graph(spec)
    resolved, _ = resolve_groups(graph, index, theme=spec.theme)

    assert len(resolved) >= len(graph.groups) - 1
    for rg in resolved.values():
        assert rg.match is not None
        assert rg.radius > 0


def test_groups_are_categorised_by_the_asset_keyword_table(index):
    """The same table that files the assets decides what a group is, so the
    search looks on the right shelf."""
    graph = SceneGraph(
        regions=[Region(id="camp", kind="settlement")],
        groups=[
            Group(id="a", asset="campfire", count=1,
                  relations=[Relation(kind="inside", target="camp")]),
            Group(id="b", asset="pine tree", count=3,
                  relations=[Relation(kind="inside", target="camp")]),
        ])
    resolved, _ = resolve_groups(graph, index)
    assert resolved["a"].category == "light_source"
    assert resolved["b"].category == "vegetation"


# --- the whole path ---------------------------------------------------------

@pytest.fixture
def graph_scene(index):
    spec = fallback_parser.parse("a forest camp around a campfire by a river")
    spec.terrain.size = "large"
    graph = compile_graph(spec)

    terrain = Terrain.from_spec(spec, seed=5)
    terrain.apply_features(spec.features, seed=5)
    return graph, terrain, build_scene(graph, terrain, index, spec=spec, seed=5)


def test_graph_scene_is_structurally_valid(graph_scene):
    _, _, scene = graph_scene
    report = validate(scene)
    assert report["valid"], report["details"]
    assert len(scene.instances) > 50


def test_graph_scene_exports_like_the_rule_based_one(graph_scene, tmp_path):
    """Same output type, so export, viewer and Unity all work unchanged.

    The shared test library points at files that do not exist on disk, which
    is fine for exercising retrieval but means no models are copied. Export
    with real files is covered in test_unity_export.py; what matters here is
    that a graph-built scene goes through the exporter at all.
    """
    from src.export_scene import export_instanced
    from src.viewer import write_viewer

    _, _, scene = graph_scene
    report = export_instanced(scene, tmp_path / "out")

    assert (tmp_path / "out" / "terrain.glb").exists()
    assert (tmp_path / "out" / "scene_manifest.json").exists()
    assert report["terrain_triangles"] > 0

    # and the viewer accepts it unchanged
    assert write_viewer(scene, tmp_path / "out").exists()


def test_graph_scene_records_the_settlement(graph_scene):
    """The ground painting and the viewer both need to know where it is."""
    _, _, scene = graph_scene
    assert scene.centre is not None
    assert scene.core_radius > 0


def test_nothing_in_the_water_on_the_graph_path(graph_scene):
    _, terrain, scene = graph_scene
    for inst in scene.instances:
        assert not terrain.is_water(inst.position[0], inst.position[2])


def test_graph_path_is_deterministic(index):
    spec = fallback_parser.parse("a medieval village")

    def run():
        graph = compile_graph(spec)
        terrain = Terrain.from_spec(spec, seed=3)
        terrain.apply_features(spec.features, seed=3)
        return build_scene(graph, terrain, index, spec=spec, seed=3)

    a, b = run(), run()
    assert [i.position for i in a.instances] == [i.position for i in b.instances]


def test_describe_renders(index):
    spec = fallback_parser.parse("a forest camp with a river")
    graph = compile_graph(spec)
    resolved, _ = resolve_groups(graph, index)

    text = describe(graph, resolved)
    assert "region" in text and "group" in text
    assert len(text.splitlines()) > 6


@pytest.mark.parametrize("prompt", [
    "a medieval village on a sunny day",
    "a forest camp with a river",
    "a desert outpost around an oasis",
    "a sci-fi base on a hill",
])
def test_every_theme_solves_through_the_graph(index, prompt):
    spec = fallback_parser.parse(prompt)
    spec.terrain.size = "large"
    graph = compile_graph(spec)

    terrain = Terrain.from_spec(spec, seed=7)
    terrain.apply_features(spec.features, seed=7)
    scene = build_scene(graph, terrain, index, spec=spec, seed=7)

    assert len(scene.instances) > 50, f"{prompt} produced an empty scene"
    assert validate(scene)["valid"]


# --- scale awareness --------------------------------------------------------

def test_population_hint_grows_with_the_scene():
    """Without a sense of scale the model writes the same four-tent camp
    whether the scene is sixty metres across or two hundred -- and at two
    hundred that camp is lost in forty thousand square metres of forest.
    """
    from src.graph_builder import _population_hint

    small = _population_hint(60)
    medium = _population_hint(120)
    large = _population_hint(200)

    assert small < medium < large
    assert large >= small * 2


def test_the_prompt_tells_the_model_how_big_the_scene_is():
    from src.graph_builder import GRAPH_USER, _population_hint

    text = GRAPH_USER.format(prompt="a camp", size=200, area="40,000",
                             buildings=_population_hint(200))
    assert "200 metres" in text
    assert "40,000" in text


def test_a_settlement_has_a_minimum_size(index):
    """Below a floor the groups inside cannot spread out, however few."""
    from src.solver import MIN_SETTLEMENT_RADIUS
    from src.scene_graph import Region, SceneGraph

    graph = SceneGraph(
        regions=[Region(id="camp", kind="settlement", size="small")],
        groups=[Group(id="tents", asset="tent", count=4,
                      relations=[Relation(kind="inside", target="camp")])])

    terrain = Terrain.generate("grassland", 60, seed=1)
    scene = build_scene(graph, terrain, index, seed=1)
    assert scene.core_radius >= min(MIN_SETTLEMENT_RADIUS, 60 * 0.42) - 0.01


def test_a_settlement_cannot_swallow_the_whole_scene(index):
    from src.scene_graph import Region, SceneGraph

    graph = SceneGraph(
        regions=[Region(id="town", kind="settlement", size="large")],
        groups=[Group(id="houses", asset="tent", count=4,
                      relations=[Relation(kind="inside", target="town")])])

    terrain = Terrain.generate("grassland", 200, seed=1)
    scene = build_scene(graph, terrain, index, seed=1)
    assert scene.core_radius <= 200 * 0.45


def test_a_large_camp_fits_in_its_settlement(index):
    """Twelve tents ringed around a fire must not pile into one clearing."""
    from src.scene_graph import Region, SceneGraph

    graph = SceneGraph(
        regions=[Region(id="camp", kind="settlement", size="small"),
                 Region(id="woods", kind="wilderness", size="large")],
        groups=[
            Group(id="fire", asset="campfire", count=1,
                  relations=[Relation(kind="at_center_of", target="camp")]),
            Group(id="tents", asset="tent", count=12,
                  relations=[Relation(kind="ring_around", target="fire",
                                      distance=12),
                             Relation(kind="faces", target="fire")]),
        ])

    terrain = Terrain.generate("forest_floor", 200, seed=5)
    scene = build_scene(graph, terrain, index, seed=5)

    tents = [i for i in scene.instances if i.name == "tent"]
    assert len(tents) >= 10, f"only {len(tents)} of 12 tents fitted"


def test_a_ring_widens_to_fit_what_goes_on_it(index):
    """The model asks for "twelve tents at nine metres" without checking
    whether twelve tents fit on a 56-metre circle. Dropping the surplus
    leaves the camp half built; widening keeps the arrangement asked for.
    """
    from src.scene_graph import Region, SceneGraph

    graph = SceneGraph(
        regions=[Region(id="camp", kind="settlement", size="medium")],
        groups=[
            Group(id="fire", asset="campfire", count=1,
                  relations=[Relation(kind="at_center_of", target="camp")]),
            Group(id="tents", asset="tent", count=20,
                  relations=[Relation(kind="ring_around", target="fire",
                                      distance=10),
                             Relation(kind="faces", target="fire")]),
        ])

    terrain = Terrain.generate("forest_floor", 200, seed=5)
    scene = build_scene(graph, terrain, index, seed=5)

    tents = [i for i in scene.instances if i.name == "tent"]
    assert len(tents) >= 17, f"only {len(tents)} of 20 placed"
    assert any("widened" in w for w in scene.warnings)


def test_the_settlement_centre_is_beside_the_road_not_on_it(index):
    """Anchoring the centre to the path midpoint puts the focal point in the
    middle of the road, and everything ringed around it then fails the
    don't-block-the-path check -- half a camp simply wasn't built.
    """
    from src.layout import PATH_HALF_WIDTH, _distance_to_path
    from src.scene_graph import Region, SceneGraph

    graph = SceneGraph(
        regions=[Region(id="camp", kind="settlement", size="medium")],
        groups=[Group(id="fire", asset="campfire", count=1,
                      relations=[Relation(kind="at_center_of",
                                          target="camp")])])

    for seed in (1, 5, 9):
        terrain = Terrain.generate("grassland", 200, seed=seed)
        scene = build_scene(graph, terrain, index, seed=seed)
        cx, cz = scene.centre
        assert _distance_to_path(cx, cz, scene.path) > PATH_HALF_WIDTH, \
            "the settlement centre sits on the road"


def test_the_road_still_runs_through_the_settlement(index):
    """Stepping aside must not push the village off the map's main route."""
    from src.layout import _distance_to_path
    from src.scene_graph import Region, SceneGraph

    graph = SceneGraph(
        regions=[Region(id="village", kind="settlement", size="medium")],
        groups=[Group(id="well", asset="well", count=1,
                      relations=[Relation(kind="at_center_of",
                                          target="village")])])

    terrain = Terrain.generate("grassland", 200, seed=3)
    scene = build_scene(graph, terrain, index, seed=3)
    cx, cz = scene.centre
    assert _distance_to_path(cx, cz, scene.path) < scene.core_radius, \
        "the road no longer passes through the settlement"
