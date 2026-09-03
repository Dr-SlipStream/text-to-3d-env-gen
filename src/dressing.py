"""
Stage 3c: dress the scene.

The LLM lists what a prompt *mentions* -- houses, a forge, some trees. It does
not list the hundred small things that make a place look inhabited: grass
tufts, pebbles, fallen branches, flower clumps, crates by a wall, a lantern on
the path.

Without them a generated scene reads as a handful of models on an empty field.
This module adds that filler automatically, driven by the theme rather than the
prompt, at densities tuned per category.

This is the single largest visual-quality lever available to us. Triangle
budget is not the constraint -- a typical scene uses under 100k triangles where
a modern GPU handles millions -- so the scene should be as full as the layout
engine can validly pack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from . import vocab
from .schema import SceneObject


@dataclass(frozen=True)
class Filler:
    """One kind of background detail to scatter."""

    query: str              # what to search the asset library for
    category: str
    per_100m2: float        # target instances per 100 square metres
    placement: str = "scatter"
    max_count: int = 400


# Density notes: ground clutter (grass, pebbles) can be dense because each
# instance is tiny; trees and rocks need space or the scene becomes
# impassable. Values below are tuned so a 120m scene fills without the
# placement engine running out of room.
FILLER_PLANS = {
    "medieval_village": [
        Filler("grass tuft plant", "vegetation", 3.2, placement="clump"),
        Filler("flower", "vegetation", 1.1, placement="clump"),
        Filler("bush shrub", "vegetation", 0.9, placement="clump"),
        Filler("tree", "vegetation", 0.7, placement="clump"),
        Filler("small rock stone", "rock", 1.0, placement="clump"),
        Filler("crate barrel", "prop", 0.4, placement="cluster"),
        Filler("fence", "structure", 0.30, placement="cluster"),
        Filler("cart wagon", "prop", 0.12, placement="along_path"),
    ],
    "forest_camp": [
        Filler("grass tuft plant", "vegetation", 3.6, placement="clump"),
        Filler("tree", "vegetation", 2.4, placement="clump"),
        Filler("bush shrub", "vegetation", 1.6, placement="clump"),
        Filler("mushroom", "vegetation", 0.5, placement="clump"),
        Filler("log", "vegetation", 0.35, placement="clump"),
        Filler("rock stone", "rock", 1.4, placement="clump"),
        Filler("flower", "vegetation", 0.8, placement="clump"),
        Filler("crate", "prop", 0.15, placement="cluster"),
    ],
    "desert_outpost": [
        Filler("rock stone", "rock", 1.8, placement="clump"),
        Filler("cactus", "vegetation", 0.7, placement="clump"),
        Filler("dry grass plant", "vegetation", 1.4, placement="clump"),
        Filler("crate barrel", "prop", 0.4, placement="cluster"),
        Filler("fence", "structure", 0.4, placement="along_path"),
        Filler("bones skull debris", "prop", 0.2),
    ],
    # Retuned against the actual contents of the asset library rather than
    # what a sci-fi base "should" have. The pack turns out to hold hangars,
    # platforms, machines, pipes and monorails -- and no domes, antennas,
    # neon or containers, which is what we were fruitlessly asking for.
    # Every query below literally names something in the library.
    "sci_fi_base": [
        Filler("rock", "rock", 1.4, placement="clump"),
        Filler("rock crystals", "rock", 0.4, placement="clump"),
        Filler("barrel", "prop", 0.6, placement="cluster"),
        Filler("machine generator", "prop", 0.4, placement="cluster"),
        Filler("pipe", "prop", 0.5),
        Filler("machine wireless", "prop", 0.2),
        Filler("crater", "prop", 0.3, placement="clump"),
        Filler("platform", "structure", 0.6),
        Filler("monorail track", "structure", 0.3, placement="along_path"),
    ],
}

# Buildings are counted directly rather than by area, because a settlement's
# size is set by how many people live there, not by how much terrain surrounds
# it. Scattering houses per-square-metre across a large map gives a sprawl;
# a fixed count gives a village.
SETTLEMENT_BUILDINGS = {
    "medieval_village": [
        ("house cottage", 9), ("hut small house", 5),
        ("farm barn", 2), ("market stall", 3),
    ],
    "forest_camp": [("tent", 5), ("hut shack", 2)],
    "desert_outpost": [("shack hut", 6), ("storage building", 2)],
    # NB: each query's category must match what asset_rules.classify() would
    # assign it, or the literal-name filter searches the wrong category and
    # silently finds nothing. There's a test enforcing this.
    # Space Kit's only complete buildings are hangars and rocket bases.
    "sci_fi_base": [("hangar", 6), ("rocket base", 2)],
}

# Lights are placed along the path when it's dark. A dusk or night scene with
# no light sources looks unlit rather than atmospheric, and this is exactly
# the sort of thing a prompt never says but every real level has.
NIGHT_LIGHTS = {
    "medieval_village": Filler("torch lantern", "light_source", 0.55,
                               placement="along_path"),
    "forest_camp": Filler("campfire lantern", "light_source", 0.30,
                          placement="along_path"),
    "desert_outpost": Filler("lamp lantern", "light_source", 0.35,
                             placement="along_path"),
    # The sci-fi pack ships no light fixtures at all, so this falls back to
    # a lantern from another pack. Flagged as a weak match rather than hidden
    # -- it's an honest gap in the library, not a retrieval failure.
    "sci_fi_base": Filler("lamp", "light_source", 0.40,
                          placement="along_path"),
}

DARK_TIMES = {"dusk", "night"}


# Target scene density, in instances per 1000 square metres.
#
# Without this, filler counts are per-object and the totals drift wildly: a
# desert outpost came out at 13 instances per 1000m2 against a village's 78,
# a 5.9x spread, which is the difference between a populated scene and an
# empty field. Deserts *should* be sparser than woodland -- but by a factor of
# two, not six.
TARGET_DENSITY = {
    "medieval_village": 60.0,
    "forest_camp": 68.0,
    "desert_outpost": 46.0,
    "sci_fi_base": 52.0,
}
DEFAULT_TARGET_DENSITY = 55.0


def plan_filler(spec, size_metres: float,
                density: float = 1.0) -> List[SceneObject]:
    """Build the list of filler objects for a scene.

    `density` scales everything, so a demo can be dialled up or a slow machine
    dialled down without editing the plans.
    """
    plans = list(FILLER_PLANS.get(spec.theme, FILLER_PLANS["medieval_village"]))

    if spec.lighting.time_of_day in DARK_TIMES:
        light = NIGHT_LIGHTS.get(spec.theme)
        if light:
            plans.append(light)

    # Weather that reduces visibility means less distant detail is visible, so
    # heavy scatter is wasted; pull it back slightly to save triangles.
    if spec.lighting.weather in ("fog", "sandstorm", "storm"):
        density *= 0.85

    area_units = (size_metres * size_metres) / 100.0

    # Scale the whole plan so the scene lands near its target density,
    # whatever the individual per-object rates happen to add up to.
    raw_total = sum(
        min(int(round(f.per_100m2 * area_units)), f.max_count) for f in plans)
    target_total = (TARGET_DENSITY.get(spec.theme, DEFAULT_TARGET_DENSITY)
                    * (size_metres * size_metres) / 1000.0)
    balance = (target_total / raw_total) if raw_total > 0 else 1.0

    # Keep it a correction, not a rewrite of the artistic intent.
    balance = min(max(balance, 0.5), 2.5)

    existing = {o.name for o in spec.objects}

    # How many of each thing the prompt already asked for, by head noun --
    # filler tops these up rather than standing aside entirely.
    #
    # Skipping outright was silently gutting scenes: a desert prompt naming
    # "rock" lost the whole rock filler, which was its largest category, and
    # the scene came out at a fifth of the intended density.
    existing_counts: dict = {}
    for o in spec.objects:
        head = o.name.split()[0]
        existing_counts[head] = existing_counts.get(head, 0) + o.quantity

    out: List[SceneObject] = []

    # Settlement buildings first, so they claim the core before scatter runs.
    # A prompt naming "a village" implies dwellings; without these the scene
    # has whatever the LLM happened to list and reads as a hamlet.
    existing_buildings = sum(o.quantity for o in spec.objects
                             if o.category == "building")
    for query, count in SETTLEMENT_BUILDINGS.get(spec.theme, []):
        wanted = int(round(count * min(density, 1.6)))
        # Don't stack on top of what the prompt already asked for.
        wanted = max(0, wanted - existing_buildings // 4)
        if wanted <= 0:
            continue
        out.append(SceneObject(name=query, category="building",
                               placement="cluster", quantity=wanted))

    for f in plans:
        count = int(round(f.per_100m2 * area_units * density * balance))
        count = max(0, min(count, f.max_count))
        if count == 0:
            continue

        # Top up towards the target rather than skipping: the prompt's own
        # count is usually far below what the scene needs to look full.
        head = f.query.split()[0]
        count = max(0, count - existing_counts.get(head, 0))
        if count == 0:
            continue

        # SceneObject caps quantity at MAX_QUANTITY to stop a runaway LLM
        # requesting 200 houses. Background filler legitimately needs more
        # than that, so split it across several entries rather than weakening
        # a guard that exists for a different reason.
        remaining = count
        while remaining > 0:
            chunk = min(remaining, vocab.MAX_QUANTITY)
            out.append(SceneObject(
                name=f.query,
                category=f.category,
                placement=f.placement,
                quantity=chunk,
            ))
            remaining -= chunk

    return out


def dress_spec(spec, size_metres: float, density: float = 1.0):
    """Return a copy of `spec` with filler objects appended.

    The original prompt objects stay first in the list, which matters because
    the layout engine places in order and the things the user actually asked
    for should claim their ground before background detail fills in.
    """
    filler = plan_filler(spec, size_metres, density)
    if not filler:
        return spec, 0

    dressed = spec.model_copy(deep=True)
    # Bypass the object-count cap: filler is background detail, not the
    # prompt's content, and capping it at 12 types would defeat the purpose.
    dressed.__dict__["objects"] = list(spec.objects) + filler
    return dressed, sum(f.quantity for f in filler)
