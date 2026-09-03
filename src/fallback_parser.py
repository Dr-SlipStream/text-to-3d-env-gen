"""
Keyword-based prompt parser.

This is the safety net. If Ollama isn't running, the model returns garbage, or
you're demoing on a machine with no GPU, this still produces a valid SceneSpec
so the rest of the pipeline never hard-blocks.

It is deliberately dumb: keyword matching only, no ML. That's the point -- it
must never fail.
"""

from __future__ import annotations

import re
from typing import List

from . import vocab
from .schema import Lighting, SceneObject, SceneSpec, Terrain

# Words that suggest a specific object, mapped to (category, default_quantity).
# Extend this as you add assets to the library.
OBJECT_HINTS = {
    # buildings
    "house": ("building", 6), "houses": ("building", 6),
    "cottage": ("building", 5), "hut": ("building", 5), "huts": ("building", 5),
    "tavern": ("building", 1), "inn": ("building", 1),
    "blacksmith": ("building", 1), "forge": ("building", 1),
    "tower": ("building", 1), "windmill": ("building", 1),
    "church": ("building", 1), "market": ("building", 3),
    "stall": ("building", 4), "stalls": ("building", 4),
    "tent": ("building", 4), "tents": ("building", 4),
    "shack": ("building", 3), "dome": ("building", 3),
    "hangar": ("building", 1), "lab": ("building", 1),
    # structures
    "bridge": ("structure", 1), "wall": ("structure", 4),
    "walls": ("structure", 4), "fence": ("structure", 8),
    "gate": ("structure", 1), "well": ("structure", 1),
    "platform": ("structure", 2),
    # vegetation
    "tree": ("vegetation", 20), "trees": ("vegetation", 20),
    "pine": ("vegetation", 20), "oak": ("vegetation", 15),
    "bush": ("vegetation", 15), "bushes": ("vegetation", 15),
    "shrub": ("vegetation", 12), "cactus": ("vegetation", 10),
    "cacti": ("vegetation", 10), "palm": ("vegetation", 8),
    "flower": ("vegetation", 20), "flowers": ("vegetation", 20),
    # rocks
    "rock": ("rock", 12), "rocks": ("rock", 12),
    "boulder": ("rock", 8), "boulders": ("rock", 8),
    "stone": ("rock", 10), "stones": ("rock", 10),
    # props
    "barrel": ("prop", 8), "barrels": ("prop", 8),
    "crate": ("prop", 8), "crates": ("prop", 8),
    "cart": ("prop", 3), "wagon": ("prop", 2),
    "sign": ("prop", 4), "signs": ("prop", 4),
    "crystal": ("rock", 6), "crystals": ("rock", 6),
    "debris": ("prop", 10), "container": ("prop", 6),
    # lights
    "torch": ("light_source", 8), "torches": ("light_source", 8),
    "lamp": ("light_source", 6), "lamps": ("light_source", 6),
    "lantern": ("light_source", 6), "campfire": ("light_source", 1),
    "fire": ("light_source", 1), "neon": ("light_source", 8),
}

# If the prompt names no objects at all, fall back to a sensible default set
# per theme so we still produce a populated scene.
THEME_DEFAULT_OBJECTS = {
    "medieval_village": [
        ("house", "building", 7), ("tree", "vegetation", 18),
        ("barrel", "prop", 8), ("well", "structure", 1),
        ("torch", "light_source", 6), ("rock", "rock", 8),
    ],
    "forest_camp": [
        ("tent", "building", 4), ("tree", "vegetation", 35),
        ("campfire", "light_source", 1), ("rock", "rock", 12),
        ("bush", "vegetation", 18), ("crate", "prop", 5),
    ],
    "desert_outpost": [
        ("shack", "building", 5), ("cactus", "vegetation", 12),
        ("rock", "rock", 14), ("crate", "prop", 8),
        ("fence", "structure", 6), ("lamp", "light_source", 4),
    ],
    # Named against what the library actually contains. Asking for domes,
    # neon and containers -- none of which exist in the sci-fi pack -- meant
    # every sci-fi scene opened with four guaranteed weak matches.
    "sci_fi_base": [
        ("hangar", "building", 5), ("platform", "structure", 6),
        ("machine", "prop", 8), ("barrel", "prop", 6),
        ("rock", "rock", 10), ("pipe", "prop", 5),
    ],
}

# Multi-word phrases that should collapse to a single object, so
# "blacksmith forge" doesn't become both a "blacksmith" and a "forge".
SYNONYM_GROUPS = [
    ({"blacksmith", "forge"}, "blacksmith forge", "building"),
    ({"tavern", "inn"}, "tavern", "building"),
    ({"campfire", "fire"}, "campfire", "light_source"),
    ({"rock", "stone", "boulder"}, "rock", "rock"),
    ({"market", "stall"}, "market stall", "building"),
]

# Categories a scene of each theme should always contain, so we never produce
# a "village" with no houses. Checked after keyword extraction.
THEME_ESSENTIAL_CATEGORIES = {
    "medieval_village": ["building", "vegetation"],
    "forest_camp": ["vegetation", "building"],
    "desert_outpost": ["building", "rock"],
    "sci_fi_base": ["building", "prop"],
}

# Rough quantity multipliers from size adjectives in the prompt.
SIZE_HINTS = {
    "small": 0.6, "tiny": 0.5, "little": 0.6, "few": 0.5,
    "large": 1.5, "big": 1.4, "huge": 1.8, "sprawling": 1.8,
    "many": 1.6, "dense": 1.7, "packed": 1.6,
}


def _find(text: str, keyword_map: dict, default: str,
          report_match: bool = False):
    """Return the first vocabulary value whose keywords appear in `text`.

    With `report_match`, also returns whether anything actually matched --
    the caller needs to know the difference between "the prompt said village"
    and "we fell back to village because nothing matched".
    """
    best_value, best_pos, best_len = default, len(text) + 1, 0
    for value, words in keyword_map.items():
        for w in words:
            pos = text.find(w)
            if pos != -1:
                # Prefer earlier mentions; break ties with longer keywords.
                if pos < best_pos or (pos == best_pos and len(w) > best_len):
                    best_value, best_pos, best_len = value, pos, len(w)
    matched = best_len > 0
    return (best_value, matched) if report_match else best_value


def parse(prompt: str) -> SceneSpec:
    """Turn a free-form prompt into a SceneSpec using keyword matching only."""
    text = " " + prompt.lower().strip() + " "

    theme, theme_matched = _find(text, vocab.THEME_KEYWORDS,
                                 "medieval_village", report_match=True)
    terrain_type = _find(
        text, vocab.TERRAIN_KEYWORDS, vocab.THEME_DEFAULT_TERRAIN[theme]
    )
    time_of_day = _find(text, vocab.TIME_KEYWORDS, "day")
    weather = _find(text, vocab.WEATHER_KEYWORDS, "clear")
    mood = _find(text, vocab.MOOD_KEYWORDS, "peaceful")
    art_style = _find(text, vocab.ART_STYLE_KEYWORDS, "low_poly")

    # Scene size from explicit adjectives
    size = "medium"
    if re.search(r"\b(small|tiny|little)\b", text):
        size = "small"
    elif re.search(r"\b(large|big|huge|sprawling|vast)\b", text):
        size = "large"

    # Density multiplier
    mult = 1.0
    for word, m in SIZE_HINTS.items():
        if re.search(rf"\b{word}\b", text):
            mult = m
            break

    # Objects mentioned explicitly in the prompt
    objects: List[SceneObject] = []
    seen = set()
    for word, (category, qty) in OBJECT_HINTS.items():
        if re.search(rf"\b{re.escape(word)}\b", text):
            base = word.rstrip("s") if word.endswith("s") else word
            if base in seen:
                continue
            seen.add(base)

            # Honour an explicit number if one precedes the word ("3 houses")
            m = re.search(rf"\b(\d+)\s+{re.escape(word)}\b", text)
            quantity = int(m.group(1)) if m else max(1, round(qty * mult))
            objects.append(
                SceneObject(name=base, category=category, quantity=quantity)
            )

    warnings = []
    if not theme_matched:
        warnings.append(
            f"no recognised theme in the prompt; generated as "
            f"'{theme}'. Supported themes: {', '.join(vocab.THEMES)}"
        )

    # Collapse synonym groups ("blacksmith" + "forge" -> one building).
    for group, canonical, category in SYNONYM_GROUPS:
        matched = [o for o in objects if o.name in group]
        if len(matched) > 1:
            keep_qty = max(o.quantity for o in matched)
            objects = [o for o in objects if o.name not in group]
            objects.append(
                SceneObject(name=canonical, category=category, quantity=keep_qty)
            )
            seen.add(canonical)

    if not objects:
        warnings.append("no objects found in prompt; used theme defaults")
        for name, category, qty in THEME_DEFAULT_OBJECTS[theme]:
            objects.append(
                SceneObject(
                    name=name, category=category, quantity=max(1, round(qty * mult))
                )
            )
    else:
        # A scene must contain the categories that define its theme -- a
        # village without houses, or a forest without trees, is not usable.
        present = {o.category for o in objects}
        missing = [
            c for c in THEME_ESSENTIAL_CATEGORIES[theme] if c not in present
        ]
        if missing:
            warnings.append(
                f"prompt lacked essential categories {missing}; added defaults"
            )
            for name, category, qty in THEME_DEFAULT_OBJECTS[theme]:
                if category in missing and name not in seen:
                    objects.append(
                        SceneObject(
                            name=name,
                            category=category,
                            quantity=max(1, round(qty * mult)),
                        )
                    )
                    seen.add(name)
                    missing.remove(category)

        # Still very sparse? Top up so the scene doesn't look empty.
        if len(objects) < 4:
            warnings.append("sparse prompt; topped up with theme defaults")
            for name, category, qty in THEME_DEFAULT_OBJECTS[theme]:
                if name not in seen and len(objects) < 6:
                    objects.append(
                        SceneObject(
                            name=name,
                            category=category,
                            quantity=max(1, round(qty * mult)),
                        )
                    )
                    seen.add(name)

    return SceneSpec(
        theme=theme,
        art_style=art_style,
        terrain=Terrain(type=terrain_type, size=size),
        lighting=Lighting(
            time_of_day=time_of_day, weather=weather, mood=mood
        ),
        objects=objects,
        source_prompt=prompt,
        parser="fallback",
        theme_recognised=theme_matched,
        warnings=warnings,
    )
