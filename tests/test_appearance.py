"""Tests for scale normalisation and colour assignment.

Both fix problems where placement was correct but the result looked wrong:
2m "houses" scattered across a 120m field, and a scene of black silhouettes.
"""

import numpy as np
import pytest

from src import appearance
from src.export_gltf import _is_placeholder_grey, _recover_colour


# --- scale ------------------------------------------------------------------

@pytest.mark.parametrize("name,category,radius,height,expect_m,dim", [
    ("house", "building", 1.0, 1.2, 8.0, "width"),
    ("house large", "building", 2.0, 2.4, 8.0, "width"),
    ("windmill", "building", 1.5, 4.0, 12.0, "height"),
    ("tree", "vegetation", 0.8, 3.0, 6.5, "height"),
    ("grass tuft", "vegetation", 0.2, 0.4, 0.55, "height"),
    ("barrel", "prop", 0.3, 0.5, 1.0, "height"),
])
def test_assets_normalise_to_real_world_size(name, category, radius, height,
                                             expect_m, dim):
    """Packs model at arbitrary scale; a 2m house reads as debris, not a village."""
    scale = appearance.normalisation_scale(name, category, radius, height)
    current = height if dim == "height" else radius * 2
    assert abs(current * scale - expect_m) < 0.01


def test_same_object_different_model_scales_agree():
    """A big and a small source model of a house should end up the same size."""
    small = appearance.normalisation_scale("house", "building", 0.5, 0.6)
    large = appearance.normalisation_scale("house", "building", 4.0, 5.0)
    assert abs(0.5 * 2 * small - 4.0 * 2 * large) < 0.01


def test_scale_is_clamped_for_broken_models():
    """A degenerate model must not produce a 500x monster."""
    scale = appearance.normalisation_scale("house", "building", 1e-6, 1e-6)
    assert scale <= appearance.MAX_SCALE


def test_longest_keyword_wins():
    """A more specific keyword must beat a shorter one it contains.

    'market stall' is a market (4m), not just a stall (3.5m). Equal-length
    keywords ('pine' vs 'tree') tie-break on dict order, which is arbitrary
    but harmless -- both give a plausible tree height.
    """
    assert appearance.target_size("watchtower", "building")[1] == 11.0
    assert appearance.target_size("market stall", "building")[1] == 4.0


# --- colour -----------------------------------------------------------------

@pytest.mark.parametrize("name,category,channel", [
    ("tree", "vegetation", "green"),
    ("bush", "vegetation", "green"),
    ("rock", "rock", "grey"),
    ("barrel", "prop", "brown"),
])
def test_palette_is_semantically_sensible(name, category, channel):
    r, g, b = appearance.base_colour(name, category)
    if channel == "green":
        assert g > r and g > b, f"{name} should be green-dominant"
    elif channel == "grey":
        assert max(abs(r - g), abs(g - b)) < 0.08, f"{name} should be neutral"
    elif channel == "brown":
        assert r > g > b, f"{name} should be brown"


def test_every_category_has_a_colour():
    from src import vocab
    for cat in vocab.OBJECT_CATEGORIES:
        rgb = appearance.base_colour("something unrecognised", cat)
        assert appearance.is_usable_colour(rgb)


def test_colour_variation_is_deterministic():
    a = appearance.varied_colour("tree", "vegetation", "key-1")
    b = appearance.varied_colour("tree", "vegetation", "key-1")
    c = appearance.varied_colour("tree", "vegetation", "key-2")
    assert a == b
    assert a != c


def test_variation_stays_close_to_base():
    """Variation should look like a natural population, not random confetti."""
    base = appearance.base_colour("tree", "vegetation")
    for i in range(30):
        v = appearance.varied_colour("tree", "vegetation", f"k{i}")
        assert max(abs(v[j] - base[j]) for j in range(3)) < 0.22


def test_black_and_white_rejected_as_lost_materials():
    assert not appearance.is_usable_colour((0.0, 0.0, 0.0))
    assert not appearance.is_usable_colour((1.0, 1.0, 1.0))
    assert appearance.is_usable_colour((0.3, 0.5, 0.25))


# --- placeholder detection --------------------------------------------------

def test_trimesh_default_grey_is_rejected():
    """trimesh invents a neutral grey when a mesh has no colour data.

    Accepting it produced an entire scene of uniform grey blocks.
    """
    assert _is_placeholder_grey((102 / 255, 102 / 255, 102 / 255))
    assert _is_placeholder_grey((0.4, 0.4, 0.4))


def test_real_colours_are_not_rejected():
    assert not _is_placeholder_grey((0.22, 0.42, 0.21))   # foliage green
    assert not _is_placeholder_grey((0.72, 0.62, 0.48))   # timber
    assert not _is_placeholder_grey((0.85, 0.85, 0.85))   # pale stone


def test_material_free_mesh_falls_back_to_palette(tmp_path):
    """The end-to-end guarantee: a mesh with no material must still get colour."""
    import trimesh

    from src.export_gltf import _load_asset

    box = trimesh.creation.box(extents=(1, 1, 1))
    path = tmp_path / "plain.glb"
    box.export(path)

    mesh = _load_asset(str(path), "tree", "vegetation", {})
    assert mesh is not None
    colours = np.asarray(mesh.visual.vertex_colors)[:, :3] / 255.0
    r, g, b = colours.mean(axis=0)
    assert g > r and g > b, "a tree with no material should still be green"


def test_shared_mesh_gets_per_object_colour(tmp_path):
    """Two objects sharing a model file must not share a colour.

    Caught when flowers came out foliage-green because a tree using the same
    mesh was cached first.
    """
    import trimesh

    from src.export_gltf import _load_asset

    trimesh.creation.box(extents=(1, 1, 1)).export(tmp_path / "shared.glb")
    path = str(tmp_path / "shared.glb")
    cache = {}

    tree = _load_asset(path, "tree", "vegetation", cache)
    flower = _load_asset(path, "flower", "vegetation", cache)

    tc = np.asarray(tree.visual.vertex_colors)[0, :3]
    fc = np.asarray(flower.visual.vertex_colors)[0, :3]
    assert not np.array_equal(tc, fc)


# --- two-tone shading -------------------------------------------------------

def test_buildings_get_a_roof():
    """A single flat colour makes a house read as a box."""
    tone = appearance.two_tone("cottage", "building")
    assert tone is not None
    walls, roof, split = tone
    assert walls != roof
    assert 0.4 < split < 0.85


def test_trees_get_a_trunk():
    tone = appearance.two_tone("pine tree", "vegetation")
    assert tone is not None
    trunk, foliage, split = tone
    assert trunk[0] > trunk[1] > trunk[2], "trunk should be brown"
    assert foliage[1] > foliage[0], "foliage should be green"
    assert split < 0.5, "trunk is the lower part"


def test_small_plants_have_no_trunk():
    """A grass tuft has no trunk to show."""
    assert appearance.two_tone("grass tuft", "vegetation") is None
    assert appearance.two_tone("flower", "vegetation") is None


def test_roof_material_suits_the_building():
    """A church gets slate; a hut gets thatch."""
    _, church_roof, _ = appearance.two_tone("church", "building")
    _, hut_roof, _ = appearance.two_tone("hut", "building")
    assert church_roof == appearance.ROOF_COLOURS["slate"]
    assert hut_roof == appearance.ROOF_COLOURS["thatch"]


def test_two_tone_applied_by_vertex_height(tmp_path):
    """The roof band must actually differ from the walls in the mesh."""
    import trimesh

    from src.export_gltf import _load_asset

    box = trimesh.creation.box(extents=(2, 3, 2))
    box.apply_translation([0, 1.5, 0])
    path = tmp_path / "house.glb"
    box.export(path)

    mesh = _load_asset(str(path), "cottage", "building", {})
    colours = np.asarray(mesh.visual.vertex_colors)[:, :3]
    ys = mesh.vertices[:, 1]

    low = colours[ys < ys.mean()].mean(axis=0)
    high = colours[ys > ys.mean()].mean(axis=0)
    assert not np.allclose(low, high, atol=6), "roof and walls are the same"


# --- height ceiling ---------------------------------------------------------

def test_tall_narrow_models_do_not_become_giants():
    """Fitting width alone turns a 1x3 model into a 24m tower.

    Buildings dwarfed the 6.5m trees before the height ceiling existed.
    """
    scale = appearance.normalisation_scale("house", "building",
                                           radius=0.5, height=3.0)
    assert 3.0 * scale <= appearance.MAX_HEIGHTS["building"] + 0.01


@pytest.mark.parametrize("category", list(appearance.MAX_HEIGHTS))
def test_height_ceiling_respected_for_every_category(category):
    scale = appearance.normalisation_scale("thing", category,
                                           radius=0.2, height=4.0)
    assert 4.0 * scale <= appearance.MAX_HEIGHTS[category] + 0.01


def test_long_flat_models_do_not_become_giants():
    """A fallen log or corridor section can pass the height check while being
    twenty metres wide.

    A forest camp filled with enormous grey logs because only height was
    capped -- the models were flat enough that the ceiling never triggered.
    """
    for name, category, radius, height in [
        ("log", "vegetation", 6.0, 0.5),
        ("corridor", "prop", 5.0, 1.0),
        ("wall", "structure", 4.0, 1.5),
    ]:
        scale = appearance.normalisation_scale(name, category, radius, height)
        world_width = radius * 2 * scale
        assert world_width <= appearance.MAX_WIDTHS[category] + 0.01, \
            f"{name} is {world_width:.1f}m wide"


@pytest.mark.parametrize("category", list(appearance.MAX_WIDTHS))
def test_width_ceiling_respected_for_every_category(category):
    scale = appearance.normalisation_scale("thing", category,
                                           radius=25.0, height=0.5)
    assert 50.0 * scale <= appearance.MAX_WIDTHS[category] + 0.01


def test_normal_models_are_not_shrunk_by_the_width_cap():
    """The cap must only catch outliers, not squash ordinary assets."""
    scale = appearance.normalisation_scale("house", "building", 1.0, 1.6)
    assert abs(2.0 * scale - 8.0) < 0.01        # still hits its 8m target

    scale = appearance.normalisation_scale("tree", "vegetation", 0.8, 3.0)
    assert abs(3.0 * scale - 6.5) < 0.01        # still hits its 6.5m target


def test_jitter_cannot_breach_the_size_caps():
    """Placement multiplies the normalised scale by random jitter for variety.

    An 8m log at 1.35x vegetation jitter renders 10.8m wide -- which is
    exactly what a forest scene showed after the caps were added, because
    the cap was applied before the jitter rather than after.
    """
    for name, category, radius, height in [
        ("log", "vegetation", 0.33, 0.05),
        ("tree", "vegetation", 0.8, 3.0),
        ("rock", "rock", 2.0, 0.4),
    ]:
        norm = appearance.normalisation_scale(name, category, radius, height)
        cap = appearance.cap_scale(category, radius, height)
        worst = min(norm * 1.35, cap)     # widest jitter any category uses

        assert radius * 2 * worst <= appearance.MAX_WIDTHS[category] + 0.01
        assert height * worst <= appearance.MAX_HEIGHTS[category] + 0.01


def test_cap_scale_does_not_shrink_reasonable_models():
    """The cap is a ceiling, not a target -- a normal tree must be unaffected."""
    norm = appearance.normalisation_scale("tree", "vegetation", 0.8, 3.0)
    cap = appearance.cap_scale("vegetation", 0.8, 3.0)
    assert cap >= norm, "cap is below the model's own target size"


def test_log_is_not_treated_as_a_stump():
    """'log stump branch' matched 'stump' -- the longest keyword -- and took
    its 0.8m *height* target. For a flat log that meant scaling up 16x."""
    dim, target = appearance.target_size("log", "vegetation")
    assert dim == "width"
    assert target > 1.0
