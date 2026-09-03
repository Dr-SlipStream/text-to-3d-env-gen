"""Tests for stage 3 (terrain + placement) and stage 5 (export)."""

import numpy as np
import pytest

from src import fallback_parser
from src.asset_index import AssetIndex, HashingEmbedder
from src.asset_resolution import resolve_scene
from src.layout import PATH_HALF_WIDTH, layout_scene, validate
from src.terrain import PRESETS, Terrain


# --- terrain ----------------------------------------------------------------

@pytest.mark.parametrize("ttype", list(PRESETS))
def test_terrain_generates_for_every_type(ttype):
    t = Terrain.generate(ttype, 120, seed=1)
    assert t.heights.shape == (128, 128)
    assert np.isfinite(t.heights).all()
    assert t.heights.max() > t.heights.min()


def test_terrain_is_deterministic():
    """Same seed must give the same ground, or results aren't reproducible."""
    a = Terrain.generate("grassland", 120, seed=7)
    b = Terrain.generate("grassland", 120, seed=7)
    assert np.array_equal(a.heights, b.heights)

    c = Terrain.generate("grassland", 120, seed=8)
    assert not np.array_equal(a.heights, c.heights)


def test_height_query_is_continuous():
    """Interpolated, not nearest-cell -- objects on slopes must not step."""
    t = Terrain.generate("rocky", 120, seed=3)
    xs = np.linspace(10, 110, 400)
    hs = [t.height_at(float(x), 60.0) for x in xs]
    jumps = np.abs(np.diff(hs))
    assert jumps.max() < 1.0, "height field has discontinuities"


def test_height_query_clamps_outside_bounds():
    t = Terrain.generate("grassland", 120, seed=1)
    for x, z in [(-50, 60), (500, 60), (60, -50), (60, 500)]:
        assert np.isfinite(t.height_at(x, z))


def test_flatten_disc_levels_ground():
    """Buildings need a level pad or they straddle slopes."""
    t = Terrain.generate("rocky", 120, seed=5)
    x, z = 60.0, 60.0
    before = t.slope_at(x, z)
    t.flatten_disc(x, z, radius=6.0)
    after = t.slope_at(x, z)
    assert after <= before + 1e-6


# --- layout -----------------------------------------------------------------

@pytest.fixture
def placed(index):
    spec = fallback_parser.parse("a medieval village at dusk with houses")
    resolved = resolve_scene(spec, index)
    terrain = Terrain.from_spec(spec, seed=42)
    return layout_scene(resolved, terrain, seed=42), terrain


def test_nothing_overlaps(placed):
    scene, _ = placed
    assert validate(scene)["overlaps"] == 0


def test_nothing_floats_or_sinks(placed):
    """Every instance must sit on the terrain surface."""
    scene, terrain = placed
    for inst in scene.instances:
        x, y, z = inst.position
        assert abs(y - terrain.height_at(x, z)) < 0.75, f"{inst.name} floats"


def test_everything_inside_bounds(placed):
    scene, terrain = placed
    for inst in scene.instances:
        x, _, z = inst.position
        assert 0 <= x <= terrain.size
        assert 0 <= z <= terrain.size


def test_path_stays_clear(placed):
    """The walkable corridor is what makes the scene traversable."""
    from src.layout import _distance_to_path
    scene, _ = placed
    for inst in scene.instances:
        x, _, z = inst.position
        assert _distance_to_path(x, z, scene.path) >= PATH_HALF_WIDTH * 0.6


def test_layout_is_deterministic(index):
    spec = fallback_parser.parse("a medieval village")
    resolved = resolve_scene(spec, index)

    def build(seed):
        return layout_scene(resolved, Terrain.from_spec(spec, seed=seed),
                            seed=seed)

    a, b = build(11), build(11)
    assert [i.to_dict() for i in a.instances] == [i.to_dict() for i in b.instances]
    assert [i.to_dict() for i in build(12).instances] != [i.to_dict() for i in a.instances]


@pytest.mark.parametrize("prompt", [
    "a foggy medieval village at dusk",
    "a dense forest camp at night",
    "a small desert outpost",
    "a large sci-fi base",
])
def test_all_themes_produce_valid_scenes(index, prompt):
    spec = fallback_parser.parse(prompt)
    resolved = resolve_scene(spec, index)
    terrain = Terrain.from_spec(spec, seed=3)
    scene = layout_scene(resolved, terrain, seed=3)

    assert len(scene.instances) > 0
    assert validate(scene)["valid"], validate(scene)["details"]


def test_instances_vary_in_rotation_and_scale(index):
    """Identical repeated meshes are the giveaway of procedural placement."""
    spec = fallback_parser.parse("a forest camp with many trees")
    resolved = resolve_scene(spec, index)
    scene = layout_scene(resolved, Terrain.from_spec(spec, seed=4), seed=4)

    trees = [i for i in scene.instances if i.category == "vegetation"]
    assert len(trees) > 3
    assert len({round(t.rotation_y, 3) for t in trees}) > 1
    assert len({round(t.scale, 3) for t in trees}) > 1


def test_buildings_placed_before_scatter(index):
    """Big objects must claim ground before hundreds of trees fill it."""
    from src.layout import _order_objects
    spec = fallback_parser.parse("a medieval village with houses and trees")
    resolved = resolve_scene(spec, index)
    ordered = [o.category for o in _order_objects(resolved.objects)]
    if "building" in ordered and "vegetation" in ordered:
        assert ordered.index("building") < ordered.index("vegetation")


# --- scene dressing ---------------------------------------------------------

def test_dressing_adds_substantial_detail(index):
    """A prompt names a handful of objects; a believable scene needs hundreds.

    35 instances across a 120m scene reads as an empty field.
    """
    from src import vocab
    from src.dressing import dress_spec

    spec = fallback_parser.parse("a medieval village at dusk")
    bare = sum(o.quantity for o in spec.objects)

    dressed, added = dress_spec(
        spec, vocab.TERRAIN_SIZE_METRES[spec.terrain.size])
    assert added > bare * 3, "dressing barely changed the scene"


def test_dressing_respects_quantity_cap_by_chunking(index):
    """Filler needs >40 instances per type, but the cap protects the LLM path."""
    from src import vocab
    from src.dressing import plan_filler

    spec = fallback_parser.parse("a medieval village")
    filler = plan_filler(spec, 200.0)
    for obj in filler:
        assert obj.quantity <= vocab.MAX_QUANTITY
    assert sum(f.quantity for f in filler) > vocab.MAX_QUANTITY


def test_filler_accounts_for_what_the_prompt_already_asked_for(index):
    """Filler tops up towards a target instead of adding blindly on top.

    It used to skip a category entirely when the prompt named it, which gutted
    scenes -- a desert prompt mentioning one rock lost all fifty rock filler
    instances. Topping up keeps the density target while still respecting the
    prompt's own request.
    """
    from src import vocab
    from src.dressing import plan_filler
    from src.schema import SceneObject, SceneSpec

    size = vocab.TERRAIN_SIZE_METRES["medium"]

    bare = SceneSpec(theme="forest_camp", objects=[])
    stocked = SceneSpec(theme="forest_camp", objects=[
        SceneObject(name="tree", category="vegetation", quantity=40)])

    def trees(spec):
        return sum(f.quantity for f in plan_filler(spec, size)
                   if f.name.split()[0] == "tree")

    # Some filler survives, but less than for an empty prompt.
    assert trees(stocked) < trees(bare)
    assert trees(bare) > 0


def test_settlement_buildings_scale_down_when_prompt_has_some(index):
    """A prompt already specifying many houses shouldn't get a second village."""
    from src import vocab
    from src.dressing import plan_filler
    from src.schema import SceneObject, SceneSpec

    size = vocab.TERRAIN_SIZE_METRES["medium"]

    bare = SceneSpec(theme="medieval_village", objects=[])
    many = SceneSpec(theme="medieval_village", objects=[
        SceneObject(name="house", category="building", quantity=20)])

    def buildings(spec):
        return sum(f.quantity for f in plan_filler(spec, size)
                   if f.category == "building")

    assert buildings(many) < buildings(bare)


def test_village_has_enough_buildings_to_read_as_one(index):
    """Seven scattered houses is a hamlet; a village needs a real cluster."""
    from src import vocab
    from src.dressing import dress_spec

    spec = fallback_parser.parse("a medieval village on a sunny day")
    dressed, _ = dress_spec(
        spec, vocab.TERRAIN_SIZE_METRES[spec.terrain.size])
    total = sum(o.quantity for o in dressed.objects
                if o.category == "building")
    assert total >= 12, f"only {total} buildings"


def test_dark_scenes_get_light_sources(index):
    """A dusk scene with no lights looks unlit, not atmospheric.

    The lights may come from the prompt itself or from dressing -- what
    matters is that the finished scene has them.
    """
    from src import vocab
    from src.dressing import dress_spec

    for tod in ("dusk", "night"):
        spec = fallback_parser.parse(f"a medieval village at {tod}")
        dressed, _ = dress_spec(
            spec, vocab.TERRAIN_SIZE_METRES[spec.terrain.size])
        assert any(o.category == "light_source" for o in dressed.objects), tod


def test_dressing_adds_lights_when_prompt_has_none(index):
    """When nothing in the prompt lights the scene, dressing must."""
    from src import vocab
    from src.dressing import plan_filler
    from src.schema import Lighting, SceneObject, SceneSpec

    spec = SceneSpec(
        theme="medieval_village",
        lighting=Lighting(time_of_day="night"),
        objects=[SceneObject(name="house", category="building", quantity=5)],
    )
    filler = plan_filler(spec, vocab.TERRAIN_SIZE_METRES["medium"])
    assert any(f.category == "light_source" for f in filler)


def test_dense_scene_stays_valid_and_fast(index):
    """Hundreds of objects must still place without overlaps, quickly.

    Linear collision checking would be tens of millions of comparisons here;
    this is what the spatial grid exists for.
    """
    import time
    from src import vocab
    from src.dressing import dress_spec

    spec = fallback_parser.parse("a medieval village at dusk")
    spec, _ = dress_spec(spec, vocab.TERRAIN_SIZE_METRES[spec.terrain.size])
    resolved = resolve_scene(spec, index)

    t0 = time.time()
    scene = layout_scene(resolved, Terrain.from_spec(spec, seed=9), seed=9)
    elapsed = time.time() - t0

    assert len(scene.instances) > 150, "dressing did not densify the scene"
    assert validate(scene)["overlaps"] == 0
    assert elapsed < 10.0, f"placement too slow: {elapsed:.1f}s"


def test_occupancy_grid_matches_bruteforce():
    """The spatial grid must not miss collisions the naive check would catch."""
    import numpy as np
    from src.layout import _OccupancyGrid

    rng = np.random.default_rng(0)
    grid = _OccupancyGrid(120.0)
    placed = []

    for _ in range(300):
        x, z = rng.uniform(0, 120, 2)
        r = float(rng.uniform(0.4, 5.0))
        brute = any((x - ox) ** 2 + (z - oz) ** 2 < (r + orad) ** 2
                    for ox, oz, orad in placed)
        assert grid.collides(x, z, r) == brute
        if not brute:
            grid.add(x, z, r)
            placed.append((x, z, r))


def test_grid_finds_large_distant_neighbours():
    """The search must reach far enough to find big objects, not just near ones.

    Widening by only the query's own radius missed a large object stored
    several cells away whose footprint still reached the proposal -- trees
    ended up overlapping a building.
    """
    from src.layout import _OccupancyGrid

    grid = _OccupancyGrid(200.0, cell=6.0)
    grid.add(100.0, 100.0, 20.0)          # one very large object

    # A small object 15m away is inside that footprint and must be rejected.
    assert grid.collides(115.0, 100.0, 1.0)
    # Well outside it, and must not be.
    assert not grid.collides(140.0, 100.0, 1.0)


# --- settlement -------------------------------------------------------------

def test_buildings_cluster_into_a_settlement(index):
    """Buildings spread evenly across the terrain read as scattered sheds.

    Real settlements are dense in the middle and thin out into farmland.
    """
    import numpy as np

    spec = fallback_parser.parse("a medieval village with houses")
    resolved = resolve_scene(spec, index)
    scene = layout_scene(resolved, Terrain.from_spec(spec, seed=6), seed=6)

    buildings = [i for i in scene.instances if i.category == "building"]
    assert len(buildings) >= 3
    assert scene.centre is not None

    cx, cz = scene.centre
    dists = [np.hypot(b.position[0] - cx, b.position[2] - cz)
             for b in buildings]
    assert max(dists) <= scene.core_radius * 1.35, "buildings escaped the core"


def test_big_vegetation_kept_out_of_the_village(index):
    """You don't get mature trees in the middle of a village square."""
    import numpy as np

    spec = fallback_parser.parse("a medieval village with trees")
    resolved = resolve_scene(spec, index)
    scene = layout_scene(resolved, Terrain.from_spec(spec, seed=6), seed=6)

    cx, cz = scene.centre
    trees = [i for i in scene.instances if i.category == "vegetation"]
    if trees:
        closest = min(np.hypot(t.position[0] - cx, t.position[2] - cz)
                      for t in trees)
        assert closest > scene.core_radius * 0.5


def test_nothing_hangs_over_the_terrain_edge(index):
    """Position inside bounds isn't enough -- the footprint must fit too."""
    spec = fallback_parser.parse("a medieval village")
    resolved = resolve_scene(spec, index)
    terrain = Terrain.from_spec(spec, seed=6)
    scene = layout_scene(resolved, terrain, seed=6)

    for inst in scene.instances:
        x, _, z = inst.position
        assert x - inst.radius >= -0.01, f"{inst.name} overhangs the west edge"
        assert x + inst.radius <= terrain.size + 0.01
        assert z - inst.radius >= -0.01
        assert z + inst.radius <= terrain.size + 0.01


def test_vegetation_grows_in_clumps_not_confetti(index):
    """Uniform scatter of hundreds of tufts reads as noise, not meadow."""
    import numpy as np

    from src import vocab
    from src.dressing import dress_spec

    spec = fallback_parser.parse("a medieval village")
    spec, _ = dress_spec(spec, vocab.TERRAIN_SIZE_METRES[spec.terrain.size])
    resolved = resolve_scene(spec, index)
    scene = layout_scene(resolved, Terrain.from_spec(spec, seed=8), seed=8)

    veg = np.array([[i.position[0], i.position[2]]
                    for i in scene.instances if i.category == "vegetation"])
    assert len(veg) > 50

    # Mean nearest-neighbour distance well below the uniform expectation
    # indicates clustering rather than an even spread.
    d = np.hypot(veg[:, None, 0] - veg[None, :, 0],
                 veg[:, None, 1] - veg[None, :, 1])
    np.fill_diagonal(d, np.inf)
    observed = d.min(axis=1).mean()

    size = 120.0
    uniform_expectation = 0.5 * np.sqrt((size * size) / len(veg))
    assert observed < uniform_expectation, "vegetation is evenly scattered"


def test_road_is_graded_flat(index):
    """A road that rolls over every bump doesn't read as built."""
    import numpy as np

    from src.layout import generate_path

    terrain = Terrain.generate("rocky", 120, seed=3)
    rng = np.random.default_rng(3)
    path = generate_path(120, rng)

    before = np.mean([terrain.slope_at(x, z) for x, z in path[::4]])
    terrain.flatten_path(path, half_width=5.0)
    after = np.mean([terrain.slope_at(x, z) for x, z in path[::4]])

    assert after < before


# --- consistency across prompts ---------------------------------------------

def test_filler_tops_up_rather_than_skipping(index):
    """A prompt naming 'rock' must not lose the entire rock filler.

    Skipping outright gutted desert scenes: rocks were their largest filler
    category, and naming one rock in the prompt removed all fifty.
    """
    from src import vocab
    from src.dressing import plan_filler
    from src.schema import SceneObject, SceneSpec

    size = vocab.TERRAIN_SIZE_METRES["medium"]

    bare = SceneSpec(theme="desert_outpost", objects=[])
    named = SceneSpec(theme="desert_outpost", objects=[
        SceneObject(name="rock", category="rock", quantity=3)])

    def rocks(spec):
        return sum(f.quantity for f in plan_filler(spec, size)
                   if f.category == "rock")

    assert rocks(named) > 0, "naming a rock removed all rock filler"
    assert rocks(named) >= rocks(bare) - 10


@pytest.mark.parametrize("prompt", [
    "a medieval village on a sunny day",
    "a dense forest camp at night",
    "a small desert outpost in a sandstorm",
    "a sci-fi base on an alien planet",
])
def test_scene_density_is_consistent_across_themes(index, prompt):
    """A system tuned on one prompt looks finished until someone types
    another. Desert scenes came out at a fifth of a village's density.
    """
    from src import vocab
    from src.dressing import dress_spec

    spec = fallback_parser.parse(prompt)
    size = vocab.TERRAIN_SIZE_METRES[spec.terrain.size]
    spec, _ = dress_spec(spec, size)
    resolved = resolve_scene(spec, index)
    scene = layout_scene(resolved, Terrain.from_spec(spec, seed=42), seed=42)

    per_1000 = 1000 * len(scene.instances) / (size * size)
    assert 25 <= per_1000 <= 110, f"{prompt}: {per_1000:.0f} per 1000m2"


# --- honest handling of unsupported prompts ---------------------------------

@pytest.mark.parametrize("prompt", [
    "a snowy mountain pass in a blizzard",
    "an underwater city of coral towers",
    "a Japanese garden with cherry blossom",
    "",
    "asdfgh qwerty zxcvb",
])
def test_unsupported_prompts_are_flagged(prompt):
    """Silently returning a medieval village for 'an underwater city' is
    worse than saying the theme wasn't understood."""
    spec = fallback_parser.parse(prompt)
    assert not spec.theme_recognised
    assert any("theme" in w.lower() for w in spec.warnings)


@pytest.mark.parametrize("prompt", [
    "a medieval village on a sunny day",
    "a dense forest camp at night",
    "a cyberpunk alley with neon signs",
    "a desert outpost in a sandstorm",
])
def test_supported_prompts_are_not_flagged(prompt):
    spec = fallback_parser.parse(prompt)
    assert spec.theme_recognised


def test_unsupported_prompts_still_produce_valid_scenes(index):
    """Flagged or not, the pipeline must never fail outright."""
    from src import vocab
    from src.dressing import dress_spec

    for prompt in ["an underwater city", "", "zzzz qqqq"]:
        spec = fallback_parser.parse(prompt)
        spec, _ = dress_spec(
            spec, vocab.TERRAIN_SIZE_METRES[spec.terrain.size])
        resolved = resolve_scene(spec, index)
        scene = layout_scene(resolved, Terrain.from_spec(spec, seed=1), seed=1)
        assert len(scene.instances) > 50
        assert validate(scene)["valid"]


def test_filler_queries_match_their_declared_category():
    """Every dressing query must classify into the category it asks for.

    Retrieval searches within the requested category, so a mismatch means the
    literal-name filter looks in the wrong place and silently finds nothing.
    This cost us sci-fi quality for days: 'crystal' was requested as a prop
    but classifies as a rock, and 'platform' as a building but classifies as
    a structure, so both fell through to semantic search and matched an
    'archery building' and an 'astronaut'.
    """
    from src import asset_rules
    from src.dressing import FILLER_PLANS, NIGHT_LIGHTS, SETTLEMENT_BUILDINGS

    problems = []

    for theme, plans in FILLER_PLANS.items():
        for f in plans:
            actual = asset_rules.classify(f.query)
            if actual != f.category:
                problems.append(f"{theme}: {f.query!r} asks for "
                                f"{f.category}, classifies as {actual}")

    for theme, items in SETTLEMENT_BUILDINGS.items():
        for query, _ in items:
            actual = asset_rules.classify(query)
            if actual != "building":
                problems.append(f"{theme}: {query!r} asks for building, "
                                f"classifies as {actual}")

    for theme, f in NIGHT_LIGHTS.items():
        actual = asset_rules.classify(f.query)
        if actual != f.category:
            problems.append(f"{theme}: {f.query!r} asks for {f.category}, "
                            f"classifies as {actual}")

    assert not problems, "category mismatches:\n  " + "\n  ".join(problems)
