"""
Filling out a scene graph.

A graph describes what a place *is*: one campfire, six tents around it, trees
through the woods. It does not describe the hundred small things that make
somewhere look real -- ground cover, undergrowth, scattered stones, fallen
branches. Nobody writing a description mentions those, and nor does a model
asked to describe a place.

The old rule-based dressing solved this with a table per theme: what a forest
camp contains, what a desert outpost contains, and so on. This does it in
graph terms instead, and the rules it needs are general rather than
per-theme:

  1. Ground cover. Every outdoor scene has some. Grass and small stones exist
     in every asset pack, so this needs no theme knowledge -- only a region
     to fill.

  2. Variation. A forest of one repeated tree reads as wallpaper. Where the
     graph scatters something, siblings of the same thing are added so the
     population varies.

  3. Density. The graph's own numbers are whatever the model felt like. They
     are scaled together so the finished scene lands near a sensible richness
     for its size, which is what stops one prompt producing a crowd and the
     next an empty field.

The result is more groups in the same graph, solved by the same solver. No new
placement logic, and nothing here knows what a village is.
"""

from __future__ import annotations

from typing import List, Optional

from .scene_graph import Group, Region, Relation, SceneGraph

# Target richness, in instances per 1000 square metres.
#
# One number, not a table of what each theme contains. Deserts are sparser
# than woodland because there is less that grows there, but the difference is
# a factor of two, not of six.
TARGET_DENSITY = {
    "medieval_village": 58.0,
    "forest_camp": 66.0,
    "desert_outpost": 40.0,
    "sci_fi_base": 46.0,
}
DEFAULT_TARGET = 52.0

# What share of a scene a wilderness region is taken to cover. Must match the
# solver's own assumption, or the density normalisation aims at the wrong
# number.
WILDERNESS_SHARE = 0.55

# Ground cover, by the kind of ground. These are the only asset names this
# module knows, and they are deliberately generic -- every pack has grass and
# small stones, whatever its theme.
# "small rock" resolves to a 1.4m boulder, and a field of those reads as
# litter rather than ground. Pebbles and grass are what actually covers
# ground, so cover leans on those and leaves the bigger stones to the scatter
# groups the graph asked for.
GROUND_COVER = {
    "forest_floor": [("grass", 3.0), ("fern plant", 1.0), ("pebble", 0.5)],
    "grassland": [("grass", 3.2), ("flower", 1.0), ("pebble", 0.4)],
    "desert_sand": [("dry grass", 1.3), ("pebble", 1.2)],
    "rocky": [("pebble", 1.8), ("grass", 1.0)],
    "barren_rock": [("pebble", 2.2), ("grass", 0.4)],
}
DEFAULT_COVER = [("grass", 2.6), ("pebble", 0.6)]

# Siblings added beside a scattered group, so a population varies. The query
# is the group's own asset with a modifier, which retrieval resolves to
# whatever the library actually holds.
VARIANTS = {
    "vegetation": ["{} large", "{} small"],
    "rock": ["{} large", "{} small"],
}

# Scatter groups the solver may thin at a region's edge rather than cutting
# off sharply. A forest that stops dead at a clearing reads as a cut-out.
SOFT_EDGE_CATEGORIES = {"vegetation", "rock"}


def _unique_id(graph: SceneGraph, base: str) -> str:
    """A readable id that doesn't collide.

    Truncating to a fixed width produced names like
    "cover_small_rock_wilderness_" twice over, which the graph then dropped
    as duplicates.
    """
    taken = {g.id for g in graph.groups} | {r.id for r in graph.regions}
    slug = base.replace(" ", "_")[:26] or "group"
    if slug not in taken:
        return slug
    for n in range(2, 60):
        candidate = f"{slug[:24]}_{n}"
        if candidate not in taken:
            return candidate
    return f"{slug[:22]}_x"


def _wilderness_regions(graph: SceneGraph) -> List[Region]:
    return [r for r in graph.regions
            if r.kind in ("wilderness", "field", "high_ground")]


def _existing_assets(graph: SceneGraph) -> set:
    return {g.asset.lower() for g in graph.groups}


def add_ground_cover(graph: SceneGraph, terrain_type: str) -> int:
    """Scatter the small stuff through every open region.

    This is the single biggest difference between a scene that reads as a
    place and one that reads as models on a field, and no prompt ever asks
    for it.
    """
    regions = _wilderness_regions(graph)
    if not regions:
        # Nowhere open to fill: give the scene a surroundings region so
        # ground cover has somewhere to go.
        region = Region(id="surroundings", kind="wilderness", size="large")
        graph.regions.append(region)
        regions = [region]

    cover = GROUND_COVER.get(terrain_type, DEFAULT_COVER)
    existing = _existing_assets(graph)
    added = 0

    for region in regions[:2]:
        for asset, density in cover:
            if asset in existing:
                continue
            gid = _unique_id(graph, f"cover_{asset.replace(' ', '_')}")
            graph.groups.append(Group(
                id=gid, asset=asset, density=density,
                relations=[Relation(kind="scatter_in", target=region.id)]))
            existing.add(asset)
            added += 1

    return added


def add_variation(graph: SceneGraph) -> int:
    """Give scattered populations some variety.

    A hundred copies of one tree is the clearest sign of procedural
    placement. Asking retrieval for "pine tree large" finds a different model
    where the library has one and falls back to the same one where it
    doesn't, so this costs nothing when the library is thin.
    """
    from . import asset_rules

    existing = _existing_assets(graph)
    added = 0

    original = list(graph.groups)
    for group in original:
        if group.density is None:
            continue
        if "_large" in group.id or "_small" in group.id:
            continue        # don't vary a variant

        category = (asset_rules.classify_confident(group.asset)
                    or asset_rules.classify(group.asset))
        patterns = VARIANTS.get(category)
        if not patterns:
            continue

        placement = group.placement
        if placement is None or placement.kind != "scatter_in":
            continue

        # Split the group's density across itself and its siblings, so
        # variation does not also mean more objects.
        share = group.density / (len(patterns) + 1)
        group.density = share

        for pattern in patterns:
            asset = pattern.format(group.asset)
            if asset in existing:
                continue
            gid = _unique_id(graph, f"{group.id}_{pattern.split()[-1]}")
            graph.groups.append(Group(
                id=gid, asset=asset, density=share,
                relations=[r.model_copy() for r in group.relations]))
            existing.add(asset)
            added += 1

    return added


def normalise_density(graph: SceneGraph, theme: Optional[str],
                      size_metres: float, multiplier: float = 1.0) -> float:
    """Scale every density so the finished scene lands near its target.

    The model's own numbers are impressionistic -- "1.8 trees per 100 square
    metres" is a guess, and a different prompt gets a different guess. Scaling
    them together keeps the relative proportions it chose while making the
    overall richness predictable.
    """
    target_per_1000 = TARGET_DENSITY.get(theme or "", DEFAULT_TARGET)

    # The solver turns a density into a count using the area of the region a
    # group sits in, and a wilderness region is taken as roughly half the
    # scene rather than all of it. Measuring the target against the full
    # scene area therefore asks for about twice what arrives, which is why a
    # normalised graph still came out half empty.
    area = size_metres * size_metres
    effective_area = area * WILDERNESS_SHARE

    # What the graph currently asks for, counting only the area-based groups.
    # Fixed counts are the prompt's own content and are left alone.
    density_total = sum(g.density for g in graph.groups
                        if g.density is not None)
    if density_total <= 0:
        return 1.0

    expected = density_total * effective_area / 100.0
    wanted = target_per_1000 * multiplier * area / 1000.0

    scale = wanted / expected
    # A correction, not a rewrite: if the model asked for a deliberately
    # sparse scene, it stays sparser than a dense one.
    scale = min(max(scale, 0.35), 4.0)

    for group in graph.groups:
        if group.density is not None:
            group.density = round(group.density * scale, 4)

    return scale


def dress(graph: SceneGraph, spec, size_metres: float,
          density: float = 1.0) -> dict:
    """Fill out a graph: ground cover, variation, then density normalisation.

    Order matters. Cover and variation add groups; normalisation runs last so
    it accounts for everything the scene will actually contain.
    """
    if density <= 0:
        return {"cover": 0, "variants": 0, "scale": 1.0}

    terrain_type = getattr(getattr(spec, "terrain", None), "type", "grassland")
    theme = getattr(spec, "theme", None)

    cover = add_ground_cover(graph, terrain_type)
    variants = add_variation(graph)
    scale = normalise_density(graph, theme, size_metres, multiplier=density)

    return {"cover": cover, "variants": variants, "scale": round(scale, 2)}
