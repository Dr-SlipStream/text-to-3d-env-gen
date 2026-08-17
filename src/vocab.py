"""
Controlled vocabulary for the whole pipeline.

Everything downstream (prompt parsing, asset retrieval, placement, lighting)
agrees on the terms defined here. If you want to support a new theme or object
type, add it HERE first -- never hardcode strings elsewhere in the codebase.
"""

# ---------------------------------------------------------------------------
# Themes -- the high-level "kind of place" being generated.
# Keep this list small and well-supported. 3-4 great themes beat 15 bad ones.
# ---------------------------------------------------------------------------
THEMES = [
    "medieval_village",
    "forest_camp",
    "desert_outpost",
    "sci_fi_base",
]

# Human-readable words that map onto each theme (used by the fallback parser
# and to help the LLM stay inside our vocabulary).
THEME_KEYWORDS = {
    "medieval_village": [
        "medieval", "village", "town", "hamlet", "fantasy", "castle",
        "blacksmith", "tavern", "market", "peasant", "kingdom",
    ],
    "forest_camp": [
        "forest", "woods", "woodland", "camp", "campsite", "jungle",
        "clearing", "wilderness", "tent", "grove",
    ],
    "desert_outpost": [
        "desert", "sand", "dune", "oasis", "outpost", "arid", "canyon",
        "wasteland", "badlands",
    ],
    "sci_fi_base": [
        "sci-fi", "scifi", "science fiction", "space", "futuristic", "cyber",
        "cyberpunk", "station", "base", "colony", "alien", "neon", "tech",
    ],
}

# ---------------------------------------------------------------------------
# Terrain
# ---------------------------------------------------------------------------
TERRAIN_TYPES = [
    "grassland",
    "forest_floor",
    "desert_sand",
    "rocky",
    "barren_rock",
]

TERRAIN_KEYWORDS = {
    "grassland": ["grass", "meadow", "field", "plains", "green", "pasture"],
    "forest_floor": ["forest", "woods", "woodland", "mossy", "undergrowth"],
    "desert_sand": ["desert", "sand", "sandy", "dune", "arid"],
    # NB: deliberately excludes "stone" -- it's usually a material adjective
    # ("stone bridge"), not a description of the ground.
    "rocky": ["rocky", "mountain", "hilly", "cliff", "crag"],
    "barren_rock": ["barren", "alien", "lunar", "crater", "wasteland"],
}

# Default terrain for each theme, used when the prompt doesn't specify one.
THEME_DEFAULT_TERRAIN = {
    "medieval_village": "grassland",
    "forest_camp": "forest_floor",
    "desert_outpost": "desert_sand",
    "sci_fi_base": "barren_rock",
}

TERRAIN_SIZES = ["small", "medium", "large"]

# Side length of the generated terrain in world units (metres), per size.
TERRAIN_SIZE_METRES = {
    "small": 60,
    "medium": 120,
    "large": 200,
}

# ---------------------------------------------------------------------------
# Lighting / atmosphere
# ---------------------------------------------------------------------------
TIMES_OF_DAY = ["dawn", "day", "dusk", "night"]

TIME_KEYWORDS = {
    "dawn": ["dawn", "sunrise", "early morning", "daybreak"],
    "day": ["day", "daytime", "noon", "midday", "sunny", "bright", "afternoon"],
    "dusk": ["dusk", "sunset", "evening", "twilight", "golden hour"],
    "night": ["night", "midnight", "dark", "moonlit", "nighttime"],
}

WEATHER = ["clear", "fog", "rain", "storm", "snow", "sandstorm"]

WEATHER_KEYWORDS = {
    "clear": ["clear", "cloudless", "sunny"],
    "fog": ["fog", "foggy", "mist", "misty", "haze", "hazy"],
    "rain": ["rain", "rainy", "drizzle", "wet", "rain-slicked", "raining"],
    "storm": ["storm", "stormy", "thunder", "lightning", "tempest"],
    "snow": ["snow", "snowy", "blizzard", "frozen", "icy", "winter"],
    "sandstorm": ["sandstorm", "dust storm", "duststorm"],
}

MOODS = ["peaceful", "mysterious", "tense", "abandoned", "lively"]

MOOD_KEYWORDS = {
    "peaceful": ["peaceful", "calm", "serene", "quiet", "tranquil", "idyllic"],
    "mysterious": ["mysterious", "eerie", "strange", "haunting", "mystical"],
    "tense": ["tense", "dangerous", "ominous", "threatening", "hostile"],
    "abandoned": ["abandoned", "ruined", "deserted", "derelict", "forgotten",
                  "broken", "decayed"],
    "lively": ["lively", "bustling", "busy", "crowded", "vibrant", "thriving"],
}

ART_STYLES = ["low_poly", "stylized", "realistic"]

ART_STYLE_KEYWORDS = {
    "low_poly": ["low poly", "low-poly", "lowpoly", "minimal", "simple"],
    "stylized": ["stylized", "cartoon", "cartoony", "hand-painted", "painterly"],
    "realistic": ["realistic", "photorealistic", "realism", "lifelike"],
}

# ---------------------------------------------------------------------------
# Objects
#
# `category` decides which folder of the asset library we search, and drives
# placement behaviour + collision shape. `name` is the specific thing the user
# asked for ("blacksmith forge"), which the retrieval stage matches on.
# ---------------------------------------------------------------------------
OBJECT_CATEGORIES = [
    "building",     # houses, huts, towers, structures
    "prop",         # barrels, crates, signs, small set dressing
    "vegetation",   # trees, bushes, cacti, grass clumps
    "rock",         # boulders, stones, rock formations
    "structure",    # bridges, walls, fences, gates
    "light_source", # torches, lamps, campfires, neon signs
]

# ---------------------------------------------------------------------------
# Placement rules -- how the layout engine positions each object.
# ---------------------------------------------------------------------------
PLACEMENT_RULES = [
    "along_path",   # lined up either side of the main path/road
    "cluster",      # grouped together around a point
    "scatter",      # spread randomly across open terrain
    "perimeter",    # around the outer edge of the scene
    "center",       # at/near the middle, a focal point
]

# Sensible default placement per category, used when the LLM doesn't specify.
CATEGORY_DEFAULT_PLACEMENT = {
    "building": "along_path",
    "prop": "cluster",
    "vegetation": "scatter",
    "rock": "scatter",
    "structure": "center",
    "light_source": "along_path",
}

# Approximate footprint radius in metres, used for overlap checks during
# placement. Refined later per-asset once the asset library is indexed.
CATEGORY_DEFAULT_RADIUS = {
    "building": 4.0,
    "prop": 0.8,
    "vegetation": 1.5,
    "rock": 1.2,
    "structure": 3.0,
    "light_source": 0.5,
}

# Some object names are reliably mis-categorised by the LLM -- it called a
# forge a "structure" (it's a building you enter). Rather than trust the
# model, we correct known cases here. Checked as substrings of the name.
OBJECT_CATEGORY_OVERRIDES = {
    # buildings the LLM tends to call structures
    "forge": "building",
    "smithy": "building",
    "blacksmith": "building",
    "tavern": "building",
    "inn": "building",
    "windmill": "building",
    "mill": "building",
    "church": "building",
    "chapel": "building",
    "tower": "building",
    "barn": "building",
    "stable": "building",
    "shop": "building",
    "market stall": "building",
    "hut": "building",
    "cottage": "building",
    "house": "building",
    "tent": "building",
    # structures the LLM tends to call buildings
    "bridge": "structure",
    "fence": "structure",
    "wall": "structure",
    "gate": "structure",
    "well": "structure",
    "fountain": "structure",
    # light sources it tends to call props
    "torch": "light_source",
    "lantern": "light_source",
    "campfire": "light_source",
    "brazier": "light_source",
    "lamp": "light_source",
}


def override_category(name: str, proposed: str) -> str:
    """Correct a category when the object name says otherwise.

    Longest match wins, so "market stall" beats "stall".
    """
    n = name.lower().strip()
    best_key = None
    for key in OBJECT_CATEGORY_OVERRIDES:
        if key in n and (best_key is None or len(key) > len(best_key)):
            best_key = key
    return OBJECT_CATEGORY_OVERRIDES[best_key] if best_key else proposed


# Things an LLM lists as "objects" that are not placeable models. Smoke, fog
# and mist are atmosphere (already captured in lighting.weather); shadows and
# light rays are rendering effects. Placing a mesh for these is always wrong --
# we saw "smoke" resolve to a fire basket. Dropped during validation.
NON_PHYSICAL_OBJECTS = {
    "smoke", "fog", "mist", "haze", "steam", "vapour", "vapor", "cloud",
    "clouds", "shadow", "shadows", "light", "lights", "lighting", "sunlight",
    "moonlight", "sunbeam", "light ray", "god ray", "glow", "ambience",
    "atmosphere", "wind", "breeze", "sound", "music", "silence", "air",
    "darkness", "gloom", "dusk", "dawn", "night", "day", "sky", "skybox",
    "weather", "rain", "snow", "storm", "reflection", "particle", "particles",
    "dust motes", "aura", "mood", "feeling", "sense",
}


def is_non_physical(name: str) -> bool:
    """True if this 'object' is atmosphere or an effect, not a placeable model."""
    n = name.lower().strip()
    if n in NON_PHYSICAL_OBJECTS:
        return True
    # Also catch "smoke plume", "wisps of mist", "rain puddles" etc. where the
    # head noun is atmospheric.
    tokens = set(n.split())
    return bool(tokens & NON_PHYSICAL_OBJECTS) and len(tokens) <= 2


MAX_OBJECT_TYPES = 12    # distinct object entries allowed per scene
MAX_QUANTITY = 40        # instances of any single object type


def all_theme_keywords() -> dict:
    """Flat {keyword: theme} lookup, longest keywords first for greedy matching."""
    pairs = []
    for theme, words in THEME_KEYWORDS.items():
        for w in words:
            pairs.append((w, theme))
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    return dict(pairs)
