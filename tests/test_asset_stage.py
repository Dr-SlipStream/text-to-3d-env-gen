"""Tests for stage 2 (SceneSpec -> concrete assets).

Uses the synthetic asset library from conftest.py, so these run anywhere
without downloading models.
"""

import pytest

from src import asset_rules, fallback_parser, vocab
from src.asset_index import AssetIndex, HashingEmbedder
from src.asset_resolution import resolve_scene


# --- filename cleaning ------------------------------------------------------

@pytest.mark.parametrize("filename,expected", [
    ("tree_default.glb", "tree"),
    ("tree_pineDefaultA.glb", "tree pine a"),
    ("rock_smallA.glb", "rock small a"),
    ("building_smithy.glb", "building smithy"),
    ("Barrel_01.obj", "barrel"),
    ("structure-dome.fbx", "structure dome"),
])
def test_clean_name(filename, expected):
    assert asset_rules.clean_name(filename) == expected


def test_clean_name_never_empty():
    """Even pathological filenames must yield something searchable."""
    for fn in ["default.glb", "01.obj", "_.glb", "model.glb"]:
        assert asset_rules.clean_name(fn).strip()


# --- categorisation ---------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("tree pine", "vegetation"),
    ("bush small", "vegetation"),
    ("cactus short", "vegetation"),
    ("rock large", "rock"),
    ("boulder round", "rock"),
    ("house small", "building"),
    ("cottage thatched", "building"),
    ("building smithy", "building"),
    ("tent small open", "building"),
    ("bridge wood", "structure"),
    ("fence simple", "structure"),
    ("well stone", "structure"),
    ("torch lit", "light_source"),
    ("campfire stones", "light_source"),
    ("lantern hanging", "light_source"),
    ("crate", "prop"),
    ("barrel open", "prop"),
    ("cart wooden", "prop"),
])
def test_classification(name, expected):
    assert asset_rules.classify(name) == expected


def test_wall_is_structure_not_rock():
    """'wall stone' is a wall, not a stone -- rule ordering matters."""
    assert asset_rules.classify("wall stone a") == "structure"


def test_every_asset_gets_a_valid_category(fake_assets):
    for a in fake_assets:
        assert a["category"] in vocab.OBJECT_CATEGORIES


def test_all_categories_populated(index):
    """Our fake library should cover every category, like a real one must."""
    cats = index.categories()
    for c in vocab.OBJECT_CATEGORIES:
        assert cats.get(c, 0) > 0, f"no assets for {c}"


# --- retrieval --------------------------------------------------------------

def test_search_respects_category(index):
    """A 'house' query restricted to buildings must not return a bush."""
    for hit in index.search("house", category="building", top_k=5):
        assert hit.category == "building"


def test_search_finds_obvious_matches(index):
    assert "tree" in index.best("tree", category="vegetation").name
    assert "barrel" in index.best("barrel", category="prop").name
    assert "bridge" in index.best("bridge", category="structure").name


def test_theme_boost_prefers_matching_pack(index):
    """A sci-fi scene should prefer space-kit assets over medieval ones."""
    hit = index.best("container", category="prop", theme="sci_fi_base")
    assert "space" in hit.asset["pack"]


def test_search_never_returns_empty_for_known_category(index):
    for cat in vocab.OBJECT_CATEGORIES:
        assert index.search("anything at all", category=cat, top_k=1)


def test_index_rejects_empty_asset_list():
    with pytest.raises(ValueError):
        AssetIndex([], embedder=HashingEmbedder())


def test_embeddings_are_deterministic(fake_assets):
    """Python's hash() is randomised per process; we must not depend on it.

    Without a stable hash, the same prompt would retrieve different assets on
    every run, making results irreproducible and demos unreliable.
    """
    a = AssetIndex(fake_assets, embedder=HashingEmbedder())
    b = AssetIndex(fake_assets, embedder=HashingEmbedder())

    for query in ["blacksmith forge", "pine tree", "wooden bridge", "torch"]:
        ra = [(m.name, round(m.score, 6)) for m in a.search(query, top_k=5)]
        rb = [(m.name, round(m.score, 6)) for m in b.search(query, top_k=5)]
        assert ra == rb, f"non-deterministic results for {query!r}"


def test_semantic_substring_match(index):
    """'blacksmith forge' should find the smithy via the shared 'smith'."""
    hit = index.best("blacksmith forge", category="building")
    assert "smithy" in hit.name


# --- full resolution --------------------------------------------------------

def test_resolve_scene_binds_every_object(index):
    spec = fallback_parser.parse("a medieval village with houses and trees")
    scene = resolve_scene(spec, index)

    assert len(scene.objects) == len(spec.objects)
    for obj in scene.objects:
        assert obj.resolved, f"{obj.name} did not resolve"
        assert obj.file
        assert obj.radius > 0


def test_resolve_preserves_quantities(index):
    spec = fallback_parser.parse("a village with 3 houses")
    scene = resolve_scene(spec, index)
    house = next(o for o in scene.objects if o.name == "house")
    assert house.quantity == 3


@pytest.mark.parametrize("prompt", [
    "a foggy medieval village at dusk with a blacksmith forge",
    "a dense dark forest camp at night with a campfire",
    "a small abandoned desert outpost in a sandstorm",
    "a sci-fi base on an alien planet with neon lights",
])
def test_all_themes_resolve_completely(index, prompt):
    """Every theme must produce a fully resolved, non-empty scene."""
    spec = fallback_parser.parse(prompt)
    scene = resolve_scene(spec, index)
    assert scene.total_instances > 0
    unresolved = [o.name for o in scene.objects if not o.resolved]
    assert not unresolved, f"unresolved: {unresolved}"


def test_missing_category_substitutes(fake_assets):
    """If the library has no light sources, we substitute rather than drop."""
    stripped = [a for a in fake_assets if a["category"] != "light_source"]
    index = AssetIndex(stripped, embedder=HashingEmbedder())

    spec = fallback_parser.parse("a village with torches")
    scene = resolve_scene(spec, index)

    for obj in scene.objects:
        assert obj.resolved
    assert any("light_source" in w for w in scene.warnings)


def test_report_renders(index):
    spec = fallback_parser.parse("a medieval village at dusk")
    scene = resolve_scene(spec, index)
    text = scene.report()
    assert "requested" in text
    assert len(text.splitlines()) > 3


# --- regression tests from real-library findings ----------------------------

def test_category_overrides_correct_llm_mistakes():
    """The LLM called a forge a 'structure'; a forge is a building you enter.

    Caught when a real 614-asset library resolved 'forge' to a stone path.
    """
    from src.schema import SceneObject

    assert SceneObject(name="forge", category="structure").category == "building"
    assert SceneObject(name="blacksmith forge", category="structure").category == "building"
    assert SceneObject(name="bridge", category="building").category == "structure"
    assert SceneObject(name="torch", category="prop").category == "light_source"


def test_override_uses_longest_match():
    """'market stall' is a building; the longer key must win over 'stall'."""
    from src import vocab
    assert vocab.override_category("market stall", "prop") == "building"


def test_placement_rederived_after_category_override():
    """A corrected category must get the placement rule for that category."""
    from src.schema import SceneObject
    obj = SceneObject(name="forge", category="structure")
    assert obj.category == "building"
    assert obj.placement == "along_path"      # building default, not 'center'


def test_distinct_objects_prefer_distinct_assets(index):
    """'house' and 'cottage' must not both resolve to the same model.

    Caught when a real library with no houses mapped both to the same tent.
    """
    from src.schema import SceneObject, SceneSpec

    spec = SceneSpec(
        theme="medieval_village",
        objects=[
            SceneObject(name="house", category="building", quantity=5),
            SceneObject(name="cottage", category="building", quantity=3),
        ],
    )
    scene = resolve_scene(spec, index)
    ids = [o.match.asset["id"] for o in scene.objects if o.resolved]
    assert len(set(ids)) == len(ids), "distinct objects reused the same asset"


def test_thresholds_differ_by_embedder(fake_assets):
    """Semantic scores run higher than hashing scores, so the bar must differ."""
    from src.asset_resolution import LOW_CONFIDENCE_BY_EMBEDDER
    assert (LOW_CONFIDENCE_BY_EMBEDDER["sentence-transformers"]
            > LOW_CONFIDENCE_BY_EMBEDDER["hashing"])

    index = AssetIndex(fake_assets, embedder=HashingEmbedder())
    spec = fallback_parser.parse("a medieval village")
    scene = resolve_scene(spec, index)
    assert scene.low_confidence_threshold == LOW_CONFIDENCE_BY_EMBEDDER["hashing"]


# --- regression: keyword prefix matching ------------------------------------

@pytest.mark.parametrize("name,expected", [
    # Each of these was misclassified by prefix matching against a keyword.
    ("cliff corner inner large rock", "rock"),      # "inn" matched "inner"
    ("roof corner", "prop"),                        # "corn" matched "corner"
    ("character keeper", "prop"),                   # "keep" matched "keeper"
    ("terrain side corner inner", "prop"),          # "inn" matched "inner"
])
def test_keywords_match_whole_words_only(name, expected):
    """Keywords must not match as prefixes of longer words.

    Found when a real 862-asset library filed cliff faces as buildings and
    roof pieces as vegetation.
    """
    assert asset_rules.classify(name) == expected


@pytest.mark.parametrize("name,expected", [
    ("trees", "vegetation"),
    ("bushes", "vegetation"),
    ("hedges", "vegetation"),
    ("rocks", "rock"),
    ("barrels", "prop"),
])
def test_plurals_still_match(name, expected):
    """Fixing prefix matching must not break simple plurals."""
    assert asset_rules.classify(name) == expected


# --- regression: modular fragments ------------------------------------------

@pytest.mark.parametrize("name,modular", [
    ("roof corner inner", True),
    ("chimney base", True),
    ("tower square mid", True),
    ("wall corner", True),
    ("house large", False),
    ("windmill", False),
    ("tree", False),
    ("corner", False),          # a single positional word is not a fragment
])
def test_modular_detection(name, modular):
    assert asset_rules.is_modular(name) is modular


def test_complete_models_outrank_fragments():
    """A whole house must beat a chimney base when both could match.

    Found when the library's only 'buildings' were modular pieces and a
    village prompt resolved 'forge' to a chimney base.
    """
    assets = [
        {
            "id": "p/chimney_base", "name": "chimney base", "category": "building",
            "pack": "p", "file": "f", "radius": 1.0, "height": 2.0,
            "modular": True, "theme_hints": [], "tags": ["chimney", "base"],
        },
        {
            "id": "p/house", "name": "house", "category": "building",
            "pack": "p", "file": "f", "radius": 1.0, "height": 2.0,
            "modular": False, "theme_hints": [], "tags": ["house"],
        },
    ]
    index = AssetIndex(assets, embedder=HashingEmbedder())
    assert index.best("house", category="building").name == "house"


def test_fragment_still_used_when_nothing_better():
    """Fragments are demoted, not excluded -- something must still resolve."""
    assets = [{
        "id": "p/roof_corner", "name": "roof corner", "category": "building",
        "pack": "p", "file": "f", "radius": 1.0, "height": 2.0,
        "modular": True, "theme_hints": [], "tags": ["roof", "corner"],
    }]
    index = AssetIndex(assets, embedder=HashingEmbedder())
    assert index.best("house", category="building") is not None


# --- regression: random ID fragments in filenames ---------------------------

@pytest.mark.parametrize("raw,expected_head", [
    ("house k6t p5n fud2.glb", "house"),
    ("house v z1 clb wm sx.glb", "house"),
    ("house yl adp cju8 u.glb", "house"),
])
def test_random_id_fragments_stripped(raw, expected_head):
    """Model hosts append random IDs to filenames; they dilute the embedding.

    Found when poly.pizza models arrived named 'House_k6tP5nFuD2'.
    """
    cleaned = asset_rules.clean_name(raw)
    assert cleaned.split()[0] == expected_head
    assert not any(ch.isdigit() for ch in cleaned)


@pytest.mark.parametrize("filename,expected", [
    ("rock_smallA.glb", "rock small a"),
    ("hangar_roundB.glb", "hangar round b"),      # variant letter must survive
    ("tree_pineDefaultA.glb", "tree pine a"),
    ("bridge_stone.glb", "bridge stone"),
])
def test_legitimate_names_unaffected(filename, expected):
    """ID stripping must not damage normal pack filenames."""
    assert asset_rules.clean_name(filename) == expected


def test_clean_name_never_strips_everything():
    """If a name is entirely ID-like we keep it rather than return nothing."""
    assert asset_rules.clean_name("k6t_p5n.glb").strip()


# --- regression: non-physical "objects" -------------------------------------

@pytest.mark.parametrize("name", ["smoke", "fog", "mist", "shadows", "wind"])
def test_non_physical_objects_dropped(name):
    """Atmosphere is not a placeable model.

    Found when the LLM listed 'smoke' as an object and it resolved to a
    fire basket.
    """
    from src.schema import SceneObject, SceneSpec

    spec = SceneSpec(objects=[
        SceneObject(name=name, category="prop", quantity=2),
        SceneObject(name="house", category="building", quantity=3),
    ])
    names = [o.name for o in spec.objects]
    assert name not in names
    assert "house" in names
    assert any("atmosphere/effect" in w for w in spec.warnings)


def test_physical_objects_with_similar_names_kept():
    """'lamp' and 'light_source' assets are real; don't over-filter."""
    from src.schema import SceneObject, SceneSpec

    spec = SceneSpec(objects=[
        SceneObject(name="street lamp", category="light_source", quantity=4),
        SceneObject(name="lighthouse", category="building", quantity=1),
    ])
    assert len(spec.objects) == 2


# --- literal name matching --------------------------------------------------

def _asset(name, category, pack, hints):
    return {"id": f"{pack}/{name}".replace(" ", "_"), "name": name,
            "category": category, "pack": pack, "file": "f",
            "radius": 1.0, "height": 2.0, "modular": False,
            "theme_hints": hints, "tags": name.split() + [category]}


@pytest.fixture
def lopsided_index():
    """A library where one theme heavily outnumbers another, as real ones do.

    The real library had ~100 medieval buildings against ~10 sci-fi ones.
    """
    assets = [_asset(n, "building", "medieval", ["medieval_village"])
              for n in ["archery building", "stone tower", "tower house",
                        "blacksmith shop", "market hall", "granary",
                        "chapel", "manor", "barracks", "stable"]]
    assets += [_asset(n, "building", "space", ["sci_fi_base"])
               for n in ["structure dome", "hangar large a", "platform large"]]
    assets += [_asset(n, "prop", "medieval", ["medieval_village"])
               for n in ["astronaut b", "weapon rifle", "lantern candle"]]
    assets += [_asset(n, "prop", "space", ["sci_fi_base"])
               for n in ["crystal glow a", "antenna tall", "container metal"]]
    return AssetIndex(assets, embedder=HashingEmbedder())


@pytest.mark.parametrize("query,category,expect", [
    ("platform", "building", "platform"),
    ("dome structure", "building", "dome"),
    ("hangar building", "building", "hangar"),
    ("crystal", "prop", "crystal"),
    ("antenna", "prop", "antenna"),
    ("container crate", "prop", "container"),
])
def test_literal_name_match_beats_semantic_drift(lopsided_index, query,
                                                 category, expect):
    """When a library is dominated by one theme, embedding similarity alone
    talks itself out of the obvious answer.

    Measured on the real 1001-asset library: 'platform' matched an 'archery
    building' at 0.43, 'crystal' matched 'astronaut', 'antenna' matched
    'weapon rifle'. Every one of those had a correctly-named asset available.
    """
    hit = lopsided_index.best(query, category=category, theme="sci_fi_base")
    assert expect in hit.name, f"{query} matched {hit.name}"
    assert hit.score > 0.6


def test_literal_match_does_not_reintroduce_the_barrel_bug():
    """Barrels exist only in off-theme packs; the canoe is on-theme.

    An earlier version let the theme bonus override the semantic score and a
    medieval prompt asking for a barrel got a canoe.
    """
    assets = [
        _asset("canoe", "prop", "nature", ["forest_camp", "medieval_village"]),
        _asset("barrel", "prop", "survival", ["forest_camp", "desert_outpost"]),
        _asset("barrel open", "prop", "survival", ["forest_camp"]),
        _asset("bucket", "prop", "nature", ["medieval_village"]),
    ]
    index = AssetIndex(assets, embedder=HashingEmbedder())
    hit = index.best("barrel", category="prop", theme="medieval_village")
    assert "barrel" in hit.name


def test_falls_back_to_semantic_when_no_literal_match(lopsided_index):
    """The filter must narrow the search, never empty it."""
    hit = lopsided_index.best("somethingnobodynamed", category="prop",
                              theme="sci_fi_base")
    assert hit is not None


def test_literal_match_respects_category(lopsided_index):
    """A literal match in the wrong category must not escape the filter."""
    for hit in lopsided_index.search("crystal", category="building", top_k=5):
        assert hit.category == "building"


@pytest.mark.parametrize("name,modular", [
    ("cliff stone", True),
    ("cliff cave stone", True),
    ("terrain road corner", True),
    ("corridor wall", True),
    ("rock", False),
    ("rock large a", False),
    ("stone tall g", False),
])
def test_terrain_sections_are_flagged_modular(name, modular):
    """Cliff and terrain sections tile into a landscape; dropped in
    individually as boulders they read as grey archways littering the map,
    which is how a forest scene ended up covered in them.
    """
    assert asset_rules.is_modular(name) is modular
