"""
Make the scene readable: real-world scale, and colour.

Two problems this solves, both of which made early scenes look wrong in ways
that had nothing to do with placement being correct.

SCALE. Free asset packs model at whatever size suited their author -- a Kenney
house is about 2 units wide, a Quaternius tree about 3 units tall. Used raw,
a "village" on a 120m terrain becomes 2m huts scattered across a field, which
reads as debris rather than architecture. We normalise every asset to a
plausible real-world size for what it represents: houses about 8m, trees about
6m, barrels about 1m.

COLOUR. Materials frequently do not survive being loaded and re-exported --
packs use texture atlases, per-part materials, or formats whose material data
trimesh drops, and the result is a scene of uniformly black silhouettes. Rather
than depend on source materials being intact, we assign each asset a colour
from a palette keyed on what the object *is*. Trees come out green, stone grey,
timber brown.

A side benefit: assets from different packs stop looking like a collage,
because they all draw from one coherent palette.
"""

from __future__ import annotations

import colorsys
import re
import zlib
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Real-world target sizes, in metres.
#
# (dimension, size) -- "height" for things defined by how tall they are,
# "width" for things defined by their footprint.
# Checked longest-keyword-first, falling back to the category default.
# ---------------------------------------------------------------------------
TARGET_SIZES = {
    # buildings
    "windmill": ("height", 12.0),
    "watchtower": ("height", 11.0),
    "tower": ("height", 10.0),
    "church": ("height", 12.0),
    "castle": ("width", 18.0),
    "barracks": ("width", 11.0),
    "warehouse": ("width", 11.0),
    "hangar": ("width", 14.0),
    "barn": ("width", 10.0),
    "tavern": ("width", 9.0),
    "inn": ("width", 9.0),
    "house": ("width", 8.0),
    "cottage": ("width", 7.0),
    "cabin": ("width", 6.5),
    "farm": ("width", 8.0),
    "hut": ("width", 5.0),
    "shack": ("width", 5.0),
    "smithy": ("width", 7.0),
    "forge": ("width", 7.0),
    "stall": ("width", 3.5),
    "market": ("width", 4.0),
    "tent": ("width", 4.0),
    "dome": ("width", 9.0),

    # structures
    "bridge": ("width", 9.0),
    "fountain": ("width", 4.0),
    "gate": ("height", 5.0),
    "wall": ("width", 4.0),
    "fence": ("width", 3.0),
    "well": ("height", 2.5),
    "platform": ("width", 6.0),
    "stairs": ("width", 3.0),

    # vegetation
    "tree": ("height", 6.5),
    "pine": ("height", 8.0),
    "oak": ("height", 7.0),
    "palm": ("height", 7.0),
    "cactus": ("height", 3.0),
    "hedge": ("height", 1.6),
    "bush": ("height", 1.3),
    "shrub": ("height", 1.2),
    "log": ("width", 2.5),
    "stump": ("height", 0.8),
    "mushroom": ("height", 0.4),
    "grass": ("height", 0.55),
    "flower": ("height", 0.45),
    "plant": ("height", 0.8),

    # rocks
    "boulder": ("width", 2.6),
    "cliff": ("width", 8.0),
    "rock": ("width", 1.4),
    "stone": ("width", 1.1),
    "pebble": ("width", 0.4),
    "crystal": ("height", 1.8),

    # props
    "cart": ("width", 2.6),
    "wagon": ("width", 3.2),
    "anvil": ("height", 0.9),
    "barrel": ("height", 1.0),
    "crate": ("width", 1.0),
    "box": ("width", 0.9),
    "container": ("width", 2.6),
    "chest": ("width", 1.1),
    "sign": ("height", 2.0),
    "banner": ("height", 2.6),
    "bench": ("width", 1.8),
    "sack": ("height", 0.8),
    "bucket": ("height", 0.5),
    "antenna": ("height", 5.0),
    "pipe": ("width", 2.0),

    # lights
    "campfire": ("width", 1.6),
    "brazier": ("height", 1.2),
    "lantern": ("height", 0.5),
    "lamp": ("height", 3.0),
    "torch": ("height", 1.8),
    "candle": ("height", 0.3),
}

CATEGORY_TARGETS = {
    "building": ("width", 7.0),
    "structure": ("width", 4.0),
    "vegetation": ("height", 2.0),
    "rock": ("width", 1.4),
    "prop": ("width", 1.0),
    "light_source": ("height", 1.8),
}

# Normalising by width alone produces giants: a model 1 unit wide and 3 units
# tall, scaled to an 8m width, ends up 24m high -- taller than any building in
# a village and utterly out of proportion with 6.5m trees. After the width fit
# we check the resulting height and shrink further if it exceeds what the
# object plausibly is.
MAX_HEIGHTS = {
    "building": 14.0,
    "structure": 8.0,
    "vegetation": 9.0,
    "rock": 4.0,
    "prop": 3.0,
    "light_source": 4.0,
}

# Height caps alone are not enough: a long, flat model -- a fallen log, a
# corridor section, a wall run -- can pass the height check while being
# twenty metres wide. Those dominate a scene and read as debris dropped from
# orbit rather than set dressing.
MAX_WIDTHS = {
    "building": 20.0,
    "structure": 14.0,
    "vegetation": 8.0,
    "rock": 6.0,
    "prop": 5.0,
    "light_source": 3.0,
}

# Beyond this, a normalisation factor is more likely a broken model than a
# genuinely tiny one, so we clamp rather than produce a 500x monster.
MIN_SCALE, MAX_SCALE = 0.05, 40.0


def target_size(name: str, category: str) -> Tuple[str, float]:
    """What size should this object be in the real world?"""
    n = name.lower()
    best_key = None
    for key in TARGET_SIZES:
        if re.search(rf"\b{re.escape(key)}", n):
            if best_key is None or len(key) > len(best_key):
                best_key = key
    if best_key:
        return TARGET_SIZES[best_key]
    return CATEGORY_TARGETS.get(category, ("width", 1.5))


def normalisation_scale(name: str, category: str, radius: float,
                        height: float) -> float:
    """Factor that brings a model to a plausible real-world size.

    Packs model at arbitrary scales; without this a village is built from 2m
    houses and looks like scattered debris rather than architecture.

    Fits the target dimension first, then enforces a height ceiling so a tall
    narrow model doesn't become a tower when scaled to a sensible footprint.
    """
    dim, target = target_size(name, category)

    radius = max(float(radius), 1e-4)
    height = max(float(height), 1e-4)
    width = radius * 2.0

    current = height if dim == "height" else width
    scale = target / current

    ceiling = MAX_HEIGHTS.get(category)
    if ceiling and height * scale > ceiling:
        scale = ceiling / height

    # A long flat model can clear the height cap while still being enormous
    # across. Forest scenes filled with twenty-metre fallen logs because
    # nothing checked footprint.
    width_cap = MAX_WIDTHS.get(category)
    if width_cap and width * scale > width_cap:
        scale = width_cap / width

    return float(min(max(scale, MIN_SCALE), MAX_SCALE))


def cap_scale(category: str, radius: float, height: float) -> float:
    """The largest scale this model may take without breaking its size caps.

    Placement multiplies the normalised scale by a random jitter for variety,
    which can push an already-capped model back over the limit: an 8m log at
    1.35x jitter renders 10.8m wide. Callers clamp against this after
    jittering.
    """
    radius = max(float(radius), 1e-4)
    height = max(float(height), 1e-4)

    limits = []
    if category in MAX_WIDTHS:
        limits.append(MAX_WIDTHS[category] / (radius * 2.0))
    if category in MAX_HEIGHTS:
        limits.append(MAX_HEIGHTS[category] / height)
    return float(min(limits)) if limits else MAX_SCALE


# ---------------------------------------------------------------------------
# Colour palette.
#
# Keyed on what the object is, so a scene reads correctly even when the source
# materials are lost in conversion.
# ---------------------------------------------------------------------------
PALETTE = {
    # vegetation -- greens, varied so foliage isn't one flat mass
    "pine": (0.16, 0.33, 0.19),
    "tree": (0.22, 0.42, 0.21),
    "oak": (0.27, 0.45, 0.20),
    "palm": (0.30, 0.48, 0.24),
    "bush": (0.25, 0.43, 0.23),
    "shrub": (0.27, 0.44, 0.25),
    "hedge": (0.21, 0.38, 0.21),
    "grass": (0.38, 0.55, 0.26),
    "plant": (0.33, 0.50, 0.25),
    "flower": (0.72, 0.38, 0.45),
    "mushroom": (0.68, 0.34, 0.30),
    "cactus": (0.31, 0.47, 0.29),
    "log": (0.36, 0.26, 0.17),
    "stump": (0.34, 0.25, 0.16),

    # timber and thatch
    "roof": (0.45, 0.22, 0.17),
    "thatch": (0.62, 0.48, 0.24),
    "house": (0.72, 0.62, 0.48),
    "cottage": (0.74, 0.65, 0.50),
    "hut": (0.64, 0.52, 0.36),
    "shack": (0.55, 0.44, 0.31),
    "cabin": (0.50, 0.36, 0.24),
    "barn": (0.55, 0.26, 0.20),
    "tavern": (0.68, 0.56, 0.41),
    "windmill": (0.74, 0.66, 0.52),
    "farm": (0.66, 0.55, 0.38),
    "tent": (0.68, 0.62, 0.48),
    "fence": (0.47, 0.35, 0.23),
    "cart": (0.46, 0.33, 0.21),
    "wagon": (0.46, 0.33, 0.21),
    "barrel": (0.42, 0.30, 0.19),
    "crate": (0.56, 0.42, 0.26),
    "box": (0.56, 0.42, 0.26),
    "chest": (0.44, 0.31, 0.20),
    "bench": (0.48, 0.36, 0.24),
    "sign": (0.50, 0.38, 0.25),
    "bridge": (0.48, 0.38, 0.28),

    # stone
    "castle": (0.56, 0.55, 0.52),
    "tower": (0.54, 0.53, 0.50),
    "church": (0.62, 0.60, 0.56),
    "wall": (0.52, 0.51, 0.48),
    "gate": (0.48, 0.45, 0.42),
    "well": (0.50, 0.49, 0.46),
    "fountain": (0.58, 0.57, 0.55),
    "stairs": (0.54, 0.53, 0.50),
    "rock": (0.46, 0.45, 0.43),
    "stone": (0.50, 0.49, 0.47),
    "boulder": (0.44, 0.43, 0.41),
    "cliff": (0.48, 0.46, 0.43),
    "pebble": (0.52, 0.51, 0.49),

    # metal, industrial, sci-fi
    "anvil": (0.24, 0.24, 0.26),
    "forge": (0.40, 0.31, 0.26),
    "smithy": (0.44, 0.34, 0.27),
    "container": (0.36, 0.44, 0.50),
    "platform": (0.44, 0.47, 0.52),
    "dome": (0.58, 0.62, 0.66),
    "hangar": (0.48, 0.52, 0.57),
    "antenna": (0.50, 0.53, 0.56),
    "pipe": (0.42, 0.45, 0.48),
    "crystal": (0.42, 0.62, 0.72),

    # light sources -- warm, so they read as sources even unlit
    "campfire": (0.72, 0.38, 0.18),
    "brazier": (0.62, 0.36, 0.20),
    "torch": (0.60, 0.40, 0.22),
    "lantern": (0.72, 0.56, 0.28),
    "lamp": (0.66, 0.60, 0.42),
    "candle": (0.86, 0.80, 0.62),
    "neon": (0.40, 0.72, 0.80),

    # desert
    "sand": (0.76, 0.66, 0.46),
    "bone": (0.80, 0.77, 0.68),
    "skull": (0.82, 0.79, 0.70),
}

CATEGORY_FALLBACK = {
    "building": (0.68, 0.58, 0.45),
    "structure": (0.52, 0.46, 0.38),
    "vegetation": (0.26, 0.44, 0.23),
    "rock": (0.48, 0.47, 0.45),
    "prop": (0.52, 0.42, 0.30),
    "light_source": (0.66, 0.48, 0.26),
}


def base_colour(name: str, category: str) -> Tuple[float, float, float]:
    """Pick a colour for an asset from what it represents."""
    n = name.lower()
    best_key = None
    for key in PALETTE:
        if re.search(rf"\b{re.escape(key)}", n):
            if best_key is None or len(key) > len(best_key):
                best_key = key
    if best_key:
        return PALETTE[best_key]
    return CATEGORY_FALLBACK.get(category, (0.55, 0.50, 0.45))


def varied_colour(name: str, category: str,
                  variation_key: str = "") -> Tuple[float, float, float]:
    """Base colour with a small deterministic hue and brightness shift.

    A hundred identically coloured bushes read as a texture error. Nudging
    each asset slightly apart makes the same models look like a natural
    population. Deterministic (hashed, not random) so scenes stay reproducible.
    """
    r, g, b = base_colour(name, category)
    if not variation_key:
        return r, g, b

    seed = zlib.crc32(variation_key.encode("utf-8"))
    hue_shift = ((seed % 1000) / 1000.0 - 0.5) * 0.035
    val_shift = (((seed >> 10) % 1000) / 1000.0 - 0.5) * 0.16

    h, l, s = colorsys.rgb_to_hls(r, g, b)
    h = (h + hue_shift) % 1.0
    l = min(max(l * (1.0 + val_shift), 0.04), 0.95)
    return colorsys.hls_to_rgb(h, l, s)


def is_usable_colour(rgb: Optional[Tuple[float, float, float]]) -> bool:
    """Whether a colour recovered from a source material is worth keeping.

    Near-black and near-white usually mean the material was lost rather than
    genuinely chosen -- an unlit black silhouette is the classic symptom.
    """
    if rgb is None:
        return False
    r, g, b = rgb
    brightness = (r + g + b) / 3.0
    return 0.06 < brightness < 0.97


# ---------------------------------------------------------------------------
# Two-tone shading.
#
# A single flat colour per model is what makes a generated scene read as
# untextured blocks: houses have no roof, trees have no trunk. Splitting each
# model by height and colouring the two bands differently costs nothing --
# no textures, no extra geometry -- and is the difference between "grey box"
# and "cottage".
# ---------------------------------------------------------------------------

ROOF_COLOURS = {
    "thatch": (0.58, 0.44, 0.22),
    "tile": (0.46, 0.24, 0.19),
    "slate": (0.32, 0.33, 0.37),
    "metal": (0.40, 0.44, 0.48),
}

# Which roof a building gets, by what it is.
ROOF_BY_KEYWORD = {
    "hut": "thatch", "shack": "thatch", "cottage": "thatch",
    "farm": "thatch", "barn": "tile", "house": "tile",
    "tavern": "tile", "inn": "tile", "windmill": "thatch",
    "church": "slate", "tower": "slate", "castle": "slate",
    "keep": "slate", "smithy": "tile", "forge": "tile",
    "stall": "tile", "market": "tile", "tent": "thatch",
    "dome": "metal", "hangar": "metal", "station": "metal",
    "outpost": "metal", "platform": "metal",
}

TRUNK_COLOUR = (0.31, 0.22, 0.14)

# Vegetation with a trunk worth showing. A grass tuft has no trunk; a tree does.
TRUNKED = ("tree", "pine", "oak", "birch", "palm", "willow")


def two_tone(name: str, category: str,
             variation_key: str = "") -> Optional[Tuple[tuple, tuple, float]]:
    """Return (lower_colour, upper_colour, split) or None for a single tone.

    `split` is the fraction of the model's height where the lower band ends.
    """
    n = name.lower()
    base = varied_colour(name, category, variation_key)

    if category == "building":
        roof_kind = "tile"
        best = None
        for key, kind in ROOF_BY_KEYWORD.items():
            if re.search(rf"\b{re.escape(key)}", n):
                if best is None or len(key) > len(best):
                    best, roof_kind = key, kind
        roof = ROOF_COLOURS[roof_kind]
        # Walls below, roof above. Most low-poly houses put the roof in the
        # top third or so.
        return base, roof, 0.62

    if category == "vegetation" and any(t in n for t in TRUNKED):
        # Trunk below, foliage above.
        return TRUNK_COLOUR, base, 0.32

    if category == "light_source" and any(
            k in n for k in ("torch", "lantern", "lamp", "brazier", "candle")):
        # Warm glow at the top, dark post below.
        return (0.28, 0.22, 0.16), base, 0.70

    return None
