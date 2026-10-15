"""Tests for the scene graph and the relation solver.

These cover the thing the flat object list could not express. A prompt saying
"a campfire" means one fire, at the heart of the camp, with tents facing it.
A list of objects with counts can only say "134 light sources", which is what
the system used to produce.
"""

import math

import numpy as np
import pytest

from src.scene_graph import Group, Region, Relation, SceneGraph
from src.schema import TerrainFeature
from src.solver import solve
from src.terrain import Terrain


def camp_graph():
    """The example the whole design exists to express."""
    return SceneGraph(
        regions=[
            Region(id="camp", kind="settlement", size="medium"),
            Region(id="woods", kind="wilderness", size="large"),
            Region(id="river", kind="water"),
        ],
        groups=[
            Group(id="fire", asset="campfire", count=1,
                  relations=[Relation(kind="at_center_of", target="camp")]),
            Group(id="tents", asset="tent", count=6,
                  relations=[Relation(kind="ring_around", target="fire",
                                      distance=9),
                             Relation(kind="faces", target="fire")]),
            Group(id="seats", asset="log", count=8,
                  relations=[Relation(kind="cluster_near", target="fire",
                                      distance=4)]),
            Group(id="trees", asset="pine tree", density=1.2,
                  relations=[Relation(kind="scatter_in", target="woods"),
                             Relation(kind="avoid", target="camp",
                                      distance=4)]),
        ])


RADII = {"fire": 0.8, "tents": 2.0, "seats": 0.9, "trees": 1.6, "reeds": 0.3,
         "rocks": 0.7}
CATS = {"fire": "light_source", "tents": "building", "seats": "prop",
        "trees": "vegetation", "reeds": "vegetation", "rocks": "rock"}


@pytest.fixture
def solved():
    graph = camp_graph()
    terrain = Terrain.generate("forest_floor", 200, seed=5)
    terrain.apply_features([TerrainFeature(type="river")], seed=5)
    return graph, terrain, solve(graph, terrain, RADII, CATS, seed=5)


# --- the graph itself -------------------------------------------------------

def test_a_count_of_one_means_one():
    """The whole point. "A campfire" is one campfire."""
    graph = camp_graph()
    assert graph.group("fire").count == 1


def test_group_needs_exactly_one_amount():
    both = Group(id="g", asset="x", count=5, density=2.0)
    assert (both.count is None) != (both.density is None)

    neither = Group(id="g", asset="x")
    assert neither.count == 1


def test_only_one_positional_relation_survives():
    """An object can only be in one place."""
    g = Group(id="g", asset="x", relations=[
        Relation(kind="at_center_of", target="a"),
        Relation(kind="ring_around", target="b"),
        Relation(kind="faces", target="b"),
    ])
    positional = [r for r in g.relations if r.is_positional]
    assert len(positional) == 1
    assert g.orientation is not None


def test_relations_naming_nothing_are_dropped():
    """A dangling target is the commonest way a generated graph goes wrong,
    and placing the group anywhere would hide it."""
    graph = SceneGraph(
        regions=[Region(id="camp", kind="settlement")],
        groups=[Group(id="tents", asset="tent", count=3, relations=[
            Relation(kind="ring_around", target="nonexistent"),
        ])])
    assert graph.groups[0].relations == []
    assert any("nothing in the graph" in w for w in graph.warnings)


def test_dependencies_are_placed_first():
    """Tents are positioned around a campfire, so the fire needs a position
    before the tents are solved."""
    graph = camp_graph()
    order = [g.id for g in graph.resolution_order()]
    assert order.index("fire") < order.index("tents")
    assert order.index("fire") < order.index("seats")


def test_circular_dependencies_do_not_hang():
    graph = SceneGraph(
        regions=[Region(id="r", kind="settlement")],
        groups=[
            Group(id="a", asset="x", count=1,
                  relations=[Relation(kind="cluster_near", target="b")]),
            Group(id="b", asset="y", count=1,
                  relations=[Relation(kind="cluster_near", target="a")]),
        ])
    order = graph.resolution_order()
    assert len(order) == 2
    assert any("circular" in w for w in graph.warnings)


def test_duplicate_ids_are_dropped():
    graph = SceneGraph(
        regions=[Region(id="camp"), Region(id="camp")],
        groups=[Group(id="a", asset="x"), Group(id="a", asset="y")])
    assert len(graph.regions) == 1
    assert len(graph.groups) == 1


# --- the solver -------------------------------------------------------------

def test_exactly_one_anchor_is_placed(solved):
    _, _, scene = solved
    assert len(scene.positions_of("fire")) == 1


def test_ring_places_at_the_requested_radius(solved):
    _, _, scene = solved
    fire = scene.positions_of("fire")[0]
    tents = scene.positions_of("tents")

    assert len(tents) >= 5
    for x, z in tents:
        d = math.hypot(x - fire[0], z - fire[1])
        assert 6.0 < d < 14.0, f"tent {d:.1f}m from a 9m ring"


def test_ring_spreads_around_the_anchor(solved):
    """All six on one side would satisfy the radius but not the intent."""
    _, _, scene = solved
    fire = scene.positions_of("fire")[0]
    angles = sorted(math.atan2(z - fire[1], x - fire[0])
                    for x, z in scene.positions_of("tents"))

    gaps = [(b - a) for a, b in zip(angles, angles[1:])]
    gaps.append(2 * math.pi - (angles[-1] - angles[0]))
    assert max(gaps) < math.pi, "the ring has a hole larger than a semicircle"


def test_faces_turns_objects_towards_the_anchor(solved):
    _, _, scene = solved
    fire = scene.positions_of("fire")[0]

    for inst in scene.instances:
        if inst.group_id != "tents":
            continue
        want = math.atan2(fire[0] - inst.position[0],
                          fire[1] - inst.position[2])
        error = abs((inst.rotation_y - want + math.pi) % (2 * math.pi)
                    - math.pi)
        assert error < 0.6, f"tent is {math.degrees(error):.0f} deg off"


def test_cluster_near_stays_close(solved):
    _, _, scene = solved
    fire = scene.positions_of("fire")[0]
    for x, z in scene.positions_of("seats"):
        assert math.hypot(x - fire[0], z - fire[1]) < 16.0


def test_avoid_keeps_scatter_out_of_the_settlement(solved):
    """Trees keep out of the camp, thinning at its edge rather than stopping
    on a hard circle.

    An earlier version demanded that nothing at all crossed the boundary,
    which produced a forest with a perfectly circular bite taken out of it --
    correct by the letter of the constraint and obviously artificial. What
    matters is that the middle of the camp is clear and density falls off
    towards it.
    """
    graph, _, scene = solved
    camp = scene.regions["camp"]

    inner = outer = 0
    for x, z in scene.positions_of("trees"):
        d = math.hypot(x - camp.centre[0], z - camp.centre[1]) / camp.radius
        assert d > 0.55, "a tree grew in the middle of the camp"
        if d < 1.0:
            inner += 1
        elif d < 1.4:
            outer += 1

    assert outer > inner, "no thinning towards the camp"


def test_nothing_is_placed_in_water(solved):
    _, terrain, scene = solved
    for inst in scene.instances:
        assert not terrain.is_water(inst.position[0], inst.position[2])


def test_everything_sits_on_the_terrain(solved):
    _, terrain, scene = solved
    for inst in scene.instances:
        x, y, z = inst.position
        assert abs(y - terrain.height_at(x, z)) < 0.75


def test_nothing_overlaps(solved):
    _, _, scene = solved
    items = [(i.position[0], i.position[2], i.radius) for i in scene.instances]
    for i, (ax, az, ar) in enumerate(items):
        for bx, bz, br in items[i + 1:]:
            gap = math.hypot(ax - bx, az - bz)
            assert gap >= (ar + br) * 0.9, "two objects overlap"


def test_solving_is_deterministic():
    graph = camp_graph()
    terrain = Terrain.generate("forest_floor", 200, seed=5)
    terrain.apply_features([TerrainFeature(type="river")], seed=5)

    def run():
        t = Terrain.generate("forest_floor", 200, seed=5)
        t.apply_features([TerrainFeature(type="river")], seed=5)
        return solve(camp_graph(), t, RADII, CATS, seed=5)

    a, b = run(), run()
    assert [i.position for i in a.instances] == [i.position for i in b.instances]


# --- relations that reference the terrain ------------------------------------

def test_along_a_river_follows_the_bank():
    graph = SceneGraph(
        regions=[Region(id="river", kind="water")],
        groups=[Group(id="reeds", asset="grass", density=0.8,
                      relations=[Relation(kind="along", target="river")])])

    terrain = Terrain.generate("forest_floor", 200, seed=5)
    terrain.apply_features([TerrainFeature(type="river")], seed=5)
    scene = solve(graph, terrain, RADII, CATS, seed=5)

    reeds = scene.positions_of("reeds")
    assert len(reeds) > 8, "a linear region is a strip, not a disc"
    for x, z in reeds:
        assert terrain.is_water(x, z, margin=12.0), "a reed far from the water"


def test_on_shore_stays_on_the_land():
    graph = SceneGraph(
        regions=[Region(id="lake", kind="water")],
        groups=[Group(id="rocks", asset="rock", density=0.5,
                      relations=[Relation(kind="on_shore_of", target="lake",
                                          distance=5)])])

    terrain = Terrain.generate("grassland", 200, seed=5)
    terrain.apply_features([TerrainFeature(type="lake")], seed=5)
    scene = solve(graph, terrain, RADII, CATS, seed=5)

    rocks = scene.positions_of("rocks")
    assert rocks
    for x, z in rocks:
        assert not terrain.is_water(x, z)
        assert terrain.is_water(x, z, margin=14.0), "not actually on a shore"


def test_a_graph_with_no_terrain_features_still_solves():
    """Regions naming features that don't exist must degrade, not crash."""
    graph = SceneGraph(
        regions=[Region(id="river", kind="water"),
                 Region(id="camp", kind="settlement")],
        groups=[Group(id="tents", asset="tent", count=4,
                      relations=[Relation(kind="inside", target="camp")])])

    terrain = Terrain.generate("grassland", 120, seed=1)   # no features
    scene = solve(graph, terrain, RADII, CATS, seed=1)
    assert len(scene.positions_of("tents")) >= 3
