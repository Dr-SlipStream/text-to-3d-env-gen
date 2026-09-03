"""
Rules for turning a downloaded 3D model file into a catalogued asset.

Asset packs name their files inconsistently ("tree_default.glb",
"wall-wood.obj", "Barrel_01.fbx"). This module normalises those names and
sorts each model into one of our six categories so the retrieval stage has
something structured to search.

Everything here is keyword-driven and deliberately simple -- it runs once at
ingest time, and any mistakes are easy to override in the manifest.
"""

from __future__ import annotations

import re
from typing import List, Optional

# ---------------------------------------------------------------------------
# Category rules.
#
# Checked IN ORDER, first match wins, so put specific words above general
# ones: "watchtower" must hit `building` before "tower" is considered, and
# "stone wall" must hit `structure` (a wall) not `rock` (a stone).
# ---------------------------------------------------------------------------
CATEGORY_RULES: List[tuple] = [
    # -- light sources (checked first: "campfire" contains no other keyword,
    #    but "lamp post" would otherwise fall through to prop)
    ("light_source", [
        "torch", "lantern", "lamp", "candle", "campfire", "firepit",
        "fire_pit", "brazier", "neon", "streetlight", "spotlight", "glow",
    ]),

    # -- structures: things you walk on/through/past, not enter
    ("structure", [
        "bridge", "fence", "gate", "wall", "railing", "arch", "pillar",
        "column", "stairs", "staircase", "ramp", "platform", "pier", "dock",
        "path", "road", "track", "barrier", "well", "fountain",
    ]),

    # -- buildings: enterable/occupiable structures
    ("building", [
        "house", "cottage", "cabin", "hut", "shack", "building", "tower",
        "castle", "keep", "church", "chapel", "tavern", "inn", "shop",
        "store", "stall", "market", "barn", "stable", "mill", "windmill",
        "forge", "smithy", "tent", "yurt", "dome", "hangar", "silo",
        "warehouse", "structure_large", "base", "station", "outpost",
    ]),

    # -- vegetation
    ("vegetation", [
        "tree", "pine", "oak", "birch", "palm", "willow", "bush", "shrub",
        "hedge", "plant", "flower", "grass", "fern", "cactus", "cacti",
        "mushroom", "crop", "wheat", "corn", "vine", "leaf", "foliage",
        "log", "stump", "branch", "seaweed", "coral",
    ]),

    # -- rocks
    ("rock", [
        "rock", "stone", "boulder", "cliff", "crag", "pebble", "formation",
        "crystal", "geode", "ore", "mineral",
    ]),

    # -- props: everything portable / decorative
    ("prop", [
        "crate", "barrel", "box", "chest", "bag", "sack", "basket", "bucket",
        "pot", "vase", "jar", "bottle", "sign", "banner", "flag", "bench",
        "table", "chair", "stool", "cart", "wagon", "wheelbarrow", "anvil",
        "tool", "hammer", "axe", "sword", "shield", "debris", "rubble",
        "container", "canister", "crate_large", "supply", "antenna", "pipe",
    ]),
]

# Words that appear in filenames but carry no meaning for us.
NOISE_TOKENS = {
    "default", "detail", "detailed", "simple", "basic", "low", "poly",
    "lowpoly", "model", "mesh", "obj", "glb", "gltf", "fbx", "prefab",
    "variant", "type", "style", "new", "old", "final", "v1", "v2",
}

# Pack name -> themes it suits. Used to bias retrieval toward assets that
# match the scene's theme, so a sci-fi prompt doesn't pull medieval barrels.
PACK_THEME_HINTS = {
    "nature": ["forest_camp", "medieval_village"],
    "survival": ["forest_camp", "desert_outpost"],
    "medieval": ["medieval_village"],
    "castle": ["medieval_village"],
    "fantasy": ["medieval_village"],
    "town": ["medieval_village"],
    "city": ["sci_fi_base"],
    "space": ["sci_fi_base"],
    "scifi": ["sci_fi_base"],
    "sci-fi": ["sci_fi_base"],
    "future": ["sci_fi_base"],
    "desert": ["desert_outpost"],
    "western": ["desert_outpost"],
    "graveyard": ["medieval_village"],
    "holiday": [],
    "furniture": ["medieval_village", "sci_fi_base"],
}

SUPPORTED_EXTENSIONS = {".glb", ".gltf", ".obj", ".fbx", ".dae", ".stl", ".ply"}

# Formats trimesh can measure reliably. Others fall back to category defaults.
MEASURABLE_EXTENSIONS = {".glb", ".gltf", ".obj", ".stl", ".ply", ".dae"}

# ---------------------------------------------------------------------------
# Modular fragments.
#
# Many free packs are *modular*: instead of one "house.glb" you get
# "wall.glb", "roofCorner.glb", "chimneyBase.glb" that an artist snaps
# together on a grid. Those pieces are useless to us as standalone objects --
# a scene wanting a house should never be handed a roof corner.
#
# We can't drop them (they're most of some packs, and walls/fences are
# legitimately placeable), so we tag them and let retrieval rank them below
# complete models. If a real house exists it wins; if nothing else exists,
# a fragment is still better than nothing.
# ---------------------------------------------------------------------------
MODULAR_TOKENS = {
    "corner", "inner", "outer", "mid", "middle", "edge", "side", "half",
    "cap", "junction", "crossing", "tee", "segment", "piece", "part",
    "top", "base", "bottom", "end", "joint", "connector", "transition",
    # Cliff and terrain sections are meant to tile into a landscape, not to
    # be dropped in as boulders. Scattered individually they read as grey
    # archways littering the map -- which is how a forest scene ended up
    # covered in them.
    "cliff", "terrain", "wall", "ramp", "road", "track", "corridor",
}


def is_modular(name: str) -> bool:
    """True if this looks like a snap-together fragment, not a whole object.

    Deliberately conservative: requires a positional word AND another word,
    so "corner" alone isn't flagged but "roof corner inner" is.
    """
    tokens = set(name.lower().split())
    hits = tokens & MODULAR_TOKENS
    if not hits:
        return False
    # A bare "wall" or "cliff" is still a usable object; it's the qualified
    # variants ("cliff cave stone", "terrain road corner") that are fragments.
    return len(tokens) > len(hits) or len(tokens) > 1


VOWELS = set("aeiouy")

# Short tokens that are meaningful despite looking like noise -- pack authors
# use single letters for variants ("rock large a", "hangar round b").
MEANINGFUL_SHORT = {"a", "b", "c", "d", "e", "f"}


def _is_junk_token(token: str) -> bool:
    """True for random-ID fragments, false for real words.

    Model hosting sites append random IDs to filenames -- poly.pizza turns
    "House" into "House_k6tP5nFuD2", which our camelCase splitter shatters
    into "house k6t p5n fud2". Those fragments dilute the embedding and cost
    retrieval accuracy, so we drop them.
    """
    if not token:
        return True
    # Variant letters are meaningful even though they look like noise, and
    # most of them have no vowel -- check this before the vowel rule.
    if token in MEANINGFUL_SHORT:
        return False
    if any(ch.isdigit() for ch in token):          # "k6t", "ac1", "dg5h"
        return True
    if not (set(token) & VOWELS):                  # "clb", "wm", "sx"
        return True
    if len(token) == 1:
        return True
    return False


def clean_name(filename: str) -> str:
    """'tree_pineDefaultA_01.glb' -> 'tree pine a'."""
    stem = re.sub(r"\.[^.]+$", "", filename)

    # camelCase / PascalCase -> spaced words
    stem = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", stem)

    # separators -> spaces
    stem = re.sub(r"[_\-.]+", " ", stem).lower()

    # drop trailing index numbers ("barrel 01" -> "barrel")
    stem = re.sub(r"\b\d{1,3}\b", " ", stem)

    tokens = [t for t in stem.split() if t and t not in NOISE_TOKENS]

    # Strip random-ID fragments, but never everything -- if a name is *all*
    # junk we keep the original tokens rather than return nothing.
    kept = [t for t in tokens if not _is_junk_token(t)]
    if kept:
        tokens = kept

    # collapse repeats while preserving order
    seen, out = set(), []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)

    return " ".join(out) if out else re.sub(r"\.[^.]+$", "", filename).lower()


def classify(name: str, pack: str = "") -> str:
    """Sort an asset into one of our six categories from its cleaned name.

    Matches whole words only (with optional plural). An earlier version used
    a leading word boundary but no trailing one, so keywords matched as
    prefixes and produced nonsense: "inn" matched "inner", filing
    "cliff corner inner rock" as a building; "corn" matched "corner", filing
    "roof corner" as vegetation; "keep" matched "keeper".
    """
    haystack = f"{name} {pack}".lower().replace("_", " ").replace("-", " ")
    for category, keywords in CATEGORY_RULES:
        for kw in keywords:
            kw_pattern = re.escape(kw.replace("_", " "))
            if re.search(rf"\b{kw_pattern}(s|es)?\b", haystack):
                return category
    return "prop"          # safest default: small, scatterable, low impact


def theme_hints(pack: str) -> List[str]:
    """Themes this pack's assets are appropriate for (empty = all themes)."""
    p = pack.lower()
    for key, themes in PACK_THEME_HINTS.items():
        if key in p:
            return list(themes)
    return []


def build_tags(name: str, pack: str, category: str) -> List[str]:
    """Search tags for this asset: its words, its category, its pack."""
    tags = set(name.split())
    tags.add(category)
    if pack:
        tags.update(re.split(r"[_\-\s]+", pack.lower()))
    return sorted(t for t in tags if t and t not in NOISE_TOKENS)
