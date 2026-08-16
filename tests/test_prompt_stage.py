"""Tests for stage 1 (prompt -> SceneSpec).

Run with:  python -m pytest tests/ -v
These tests never touch the LLM, so they pass on any machine.
"""

import pytest

from src import fallback_parser, vocab
from src.schema import SceneObject, SceneSpec


# --- vocabulary conformance -------------------------------------------------

@pytest.mark.parametrize("prompt,expected_theme", [
    ("a foggy medieval village at dusk", "medieval_village"),
    ("a dense forest camp at night", "forest_camp"),
    ("an abandoned desert outpost", "desert_outpost"),
    ("a futuristic sci-fi base with neon", "sci_fi_base"),
])
def test_theme_detection(prompt, expected_theme):
    spec = fallback_parser.parse(prompt)
    assert spec.theme == expected_theme


def test_all_values_inside_vocabulary():
    """Nothing may escape the controlled vocabulary -- downstream stages
    index into dicts keyed by these values and would KeyError otherwise."""
    prompts = [
        "a foggy medieval village at dusk with a blacksmith forge",
        "a dense dark forest camp at night with a campfire",
        "a small abandoned desert outpost in a sandstorm",
        "a sci-fi base on an alien planet with neon lights",
        "",                       # empty prompt must still produce a valid spec
        "asdfghjkl qwertyuiop",   # nonsense must still produce a valid spec
    ]
    for p in prompts:
        spec = fallback_parser.parse(p)
        assert spec.theme in vocab.THEMES
        assert spec.art_style in vocab.ART_STYLES
        assert spec.terrain.type in vocab.TERRAIN_TYPES
        assert spec.terrain.size in vocab.TERRAIN_SIZES
        assert spec.lighting.time_of_day in vocab.TIMES_OF_DAY
        assert spec.lighting.weather in vocab.WEATHER
        assert spec.lighting.mood in vocab.MOODS
        for obj in spec.objects:
            assert obj.category in vocab.OBJECT_CATEGORIES
            assert obj.placement in vocab.PLACEMENT_RULES
            assert 1 <= obj.quantity <= vocab.MAX_QUANTITY


def test_never_returns_empty_scene():
    """An empty scene is useless. Every prompt must yield objects."""
    for p in ["", "   ", "a place", "xyzzy"]:
        spec = fallback_parser.parse(p)
        assert len(spec.objects) >= 3, f"empty scene for prompt {p!r}"


# --- specific parsing behaviour --------------------------------------------

def test_atmosphere_detection():
    spec = fallback_parser.parse("a foggy village at dusk")
    assert spec.lighting.weather == "fog"
    assert spec.lighting.time_of_day == "dusk"


def test_material_adjective_does_not_set_terrain():
    """'stone bridge' describes a bridge, not rocky ground."""
    spec = fallback_parser.parse("a medieval village with a stone bridge")
    assert spec.terrain.type == "grassland"


def test_synonyms_collapse_to_one_object():
    """'blacksmith forge' is one building, not two."""
    spec = fallback_parser.parse("a village with a blacksmith forge")
    names = [o.name for o in spec.objects]
    assert not ("blacksmith" in names and "forge" in names)


def test_essential_categories_present():
    """A village must have buildings; a forest must have vegetation."""
    spec = fallback_parser.parse("a medieval village with a stone bridge")
    assert "building" in {o.category for o in spec.objects}

    spec = fallback_parser.parse("a forest camp with a campfire")
    assert "vegetation" in {o.category for o in spec.objects}


def test_explicit_quantity_is_respected():
    spec = fallback_parser.parse("a village with 3 houses")
    house = next(o for o in spec.objects if o.name == "house")
    assert house.quantity == 3


def test_size_adjective_changes_scene_size():
    assert fallback_parser.parse("a small village").terrain.size == "small"
    assert fallback_parser.parse("a huge village").terrain.size == "large"


# --- schema robustness ------------------------------------------------------

def test_schema_snaps_near_miss_values():
    """LLMs return 'sunset' or 'lowpoly'; we snap rather than crash."""
    spec = SceneSpec.model_validate({
        "theme": "medieval village",       # space instead of underscore
        "art_style": "lowpoly",            # missing underscore
        "terrain": {"type": "grassland", "size": "medium"},
        "lighting": {"time_of_day": "dusk", "weather": "foggy", "mood": "peaceful"},
        "objects": [{"name": "house", "category": "buildings", "quantity": 5}],
    })
    assert spec.theme == "medieval_village"
    assert spec.art_style == "low_poly"
    assert spec.lighting.weather == "fog"
    assert spec.objects[0].category == "building"


def test_schema_rejects_garbage_but_uses_defaults():
    spec = SceneSpec.model_validate({
        "theme": "underwater_atlantis",     # not in vocabulary
        "objects": [{"name": "fish", "category": "sea_creature", "quantity": 999}],
    })
    assert spec.theme in vocab.THEMES
    assert spec.objects[0].category in vocab.OBJECT_CATEGORIES
    assert spec.objects[0].quantity <= vocab.MAX_QUANTITY


def test_duplicate_objects_merge():
    spec = SceneSpec.model_validate({
        "objects": [
            {"name": "house", "category": "building", "quantity": 3},
            {"name": "house", "category": "building", "quantity": 4},
        ],
    })
    assert len(spec.objects) == 1
    assert spec.objects[0].quantity == 7


def test_placement_defaults_by_category():
    obj = SceneObject(name="tree", category="vegetation")
    assert obj.placement == "scatter"
    obj = SceneObject(name="house", category="building")
    assert obj.placement == "along_path"


def test_json_roundtrip():
    spec = fallback_parser.parse("a foggy medieval village at dusk")
    restored = SceneSpec.from_json(spec.to_json())
    assert restored.theme == spec.theme
    assert len(restored.objects) == len(spec.objects)
