"""
Bridges: where the road meets water.

The RL experiments found the generator's worst silent failure. A river carved
across the map with the road running into it, and nothing to cross on. Every
geometric check passed -- nothing overlapped, nothing floated, the road was
clear -- and yet the far half of the map could not be reached. An agent trying
to walk it was what noticed.

This module fixes it in two parts:

  plan_crossings()        run after the road is chosen. Finds each stretch
                          where the road is under water and records a bridge
                          span on the terrain, so walkability treats the
                          crossing as solid ground.

  add_bridge_instances()  run once assets are available. Puts a real bridge
                          model on each span, oriented along the road and
                          scaled to reach both banks.

It is also the first object in the system placed because of a relationship
between two other things -- a road and a river -- rather than by category
and density.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# How far past each bank a bridge reaches, in metres. A deck that ends exactly
# at the waterline looks like it is floating off the shore.
BANK_OVERLAP = 2.5

# Width of the walkable corridor across a bridge.
DECK_HALF_WIDTH = 2.5

# Spans shorter than this are not worth a bridge; the road just fords.
MIN_SPAN = 3.0


@dataclass
class Crossing:
    """One place where the road crosses water."""

    start: Tuple[float, float]       # dry ground on the near bank
    end: Tuple[float, float]         # dry ground on the far bank
    deck_height: float               # bank height the deck sits at

    @property
    def centre(self) -> Tuple[float, float]:
        return ((self.start[0] + self.end[0]) / 2,
                (self.start[1] + self.end[1]) / 2)

    @property
    def span(self) -> float:
        return math.hypot(self.end[0] - self.start[0],
                          self.end[1] - self.start[1])

    @property
    def heading(self) -> float:
        """Rotation about Y that points a model's +Z along the crossing."""
        return math.atan2(self.end[0] - self.start[0],
                          self.end[1] - self.start[1])

    def covers(self, x: float, z: float, margin: float = 0.0) -> bool:
        """True if (x, z) lies on the deck corridor."""
        ax, az = self.start
        bx, bz = self.end
        dx, dz = bx - ax, bz - az
        length_sq = dx * dx + dz * dz
        if length_sq < 1e-9:
            return math.hypot(x - ax, z - az) <= DECK_HALF_WIDTH + margin
        t = ((x - ax) * dx + (z - az) * dz) / length_sq
        if t < -0.05 or t > 1.05:
            return False
        px, pz = ax + t * dx, az + t * dz
        return math.hypot(x - px, z - pz) <= DECK_HALF_WIDTH + margin

    def to_dict(self) -> dict:
        return {
            "start": [round(v, 2) for v in self.start],
            "end": [round(v, 2) for v in self.end],
            "span": round(self.span, 2),
            "heading": round(self.heading, 4),
            "deck_height": round(self.deck_height, 3),
        }


def find_crossings(terrain, path: List[Tuple[float, float]]) -> List[Crossing]:
    """Each run of road samples that sits in water becomes one crossing."""
    if not path or terrain is None or terrain.water_level is None:
        return []

    wet = [terrain.is_water(x, z, margin=0.5) for x, z in path]
    crossings: List[Crossing] = []

    i = 0
    n = len(path)
    while i < n:
        if not wet[i]:
            i += 1
            continue
        j = i
        while j < n and wet[j]:
            j += 1

        # The dry points either side of the wet run, extended a little
        # further onto the bank so the deck lands on solid ground.
        before = path[max(i - 1, 0)]
        after = path[min(j, n - 1)]
        dx, dz = after[0] - before[0], after[1] - before[1]
        length = math.hypot(dx, dz) or 1.0
        ux, uz = dx / length, dz / length

        start = (before[0] - ux * BANK_OVERLAP, before[1] - uz * BANK_OVERLAP)
        end = (after[0] + ux * BANK_OVERLAP, after[1] + uz * BANK_OVERLAP)
        size = terrain.size
        start = (float(np.clip(start[0], 0, size)),
                 float(np.clip(start[1], 0, size)))
        end = (float(np.clip(end[0], 0, size)),
               float(np.clip(end[1], 0, size)))

        deck = max(terrain.height_at(*before), terrain.height_at(*after),
                   terrain.water_level + 0.4)
        crossing = Crossing(start=start, end=end, deck_height=float(deck))
        if crossing.span >= MIN_SPAN:
            crossings.append(crossing)
        i = j

    return crossings


def crossing_cost(terrain, path) -> float:
    """How awkward a road is to bridge: total span, plus a charge per bridge.

    Used to choose between candidate roads. One short, square crossing is a
    bridge; a road that wanders along the riverbed needs a forty-metre span
    or three bridges in a row, and reads as a mistake.
    """
    crossings = find_crossings(terrain, path)
    return sum(c.span for c in crossings) + 25.0 * len(crossings)


def trim_to_land(terrain, path):
    """Drop the ends of a road that run into the sea.

    Roads enter and leave at the map edges. When an edge is coast, the road
    should stop at the shore -- a bridge out to the horizon is not a
    crossing. Only the ends are trimmed; water in the middle is a river or a
    lake and still gets a bridge.
    """
    if not path or terrain is None or terrain.water_level is None:
        return path
    start, end = 0, len(path)
    while start < end and terrain.is_water(*path[start], margin=1.0):
        start += 1
    while end > start and terrain.is_water(*path[end - 1], margin=1.0):
        end -= 1
    return list(path[start:end])


def choose_road(terrain, size: float, rng, generate, tries: int = 10):
    """Generate several roads and keep the one that crosses water best."""
    if terrain is None or terrain.water_level is None:
        return generate(size, rng)

    best, best_cost = None, None
    for _ in range(tries):
        candidate = trim_to_land(terrain, generate(size, rng))
        if len(candidate) < 12:
            # Barely any road left once the sea is cut off; try another.
            cost = 1e6 - len(candidate)
        else:
            cost = crossing_cost(terrain, candidate)
        if best is None or cost < best_cost:
            best, best_cost = candidate, cost
        if cost == 0:
            break
    if len(best) < 2:
        return generate(size, rng)
    return best


def plan_crossings(terrain, path) -> List[Crossing]:
    """Find the crossings and record them on the terrain.

    Recording them on the terrain is what makes walkability, placement and
    the RL environment all agree that the river can now be crossed there.
    """
    crossings = find_crossings(terrain, path)
    if crossings and terrain.features is not None:
        terrain.features.bridges = crossings
    return crossings


# ---------------------------------------------------------------------------
# Putting a model on each span
# ---------------------------------------------------------------------------

# Target size of a bridge in the world. A bridge is as wide as the road it
# carries and stands a little above the bank -- not as big as the river is
# wide. Scaling one model up to the full span made a 23-metre crossing into a
# 23-metre-long, 12-metre-tall fortress.
DECK_WIDTH = 5.5
MAX_BRIDGE_HEIGHT = 4.0
MAX_SEGMENTS = 16


def _measure(asset_file: str, fallback_radius: float):
    """(long, short, height, long_axis_is_x) for a model, in model units.

    Bridge models in different packs are built along different axes, so
    reading the mesh is the only reliable way to orient one along a road.
    """
    try:
        import trimesh
        mesh = trimesh.load(asset_file, force="mesh", process=False)
        lo, hi = mesh.bounds
        ex, ey, ez = (float(v) for v in (hi - lo))
        along_x = ex >= ez
        long_, short = (ex, ez) if along_x else (ez, ex)
        return long_, max(short, 1e-3), max(ey, 1e-3), along_x
    except Exception:
        d = max(fallback_radius * 2.0, 1e-3)
        return d, d * 0.5, d * 0.3, None


def add_bridge_instances(scene, index, theme: Optional[str] = None) -> int:
    """Place a bridge model on every recorded crossing. Returns how many."""
    from .layout import PlacedInstance

    terrain = scene.terrain
    features = getattr(terrain, "features", None)
    crossings = getattr(features, "bridges", None) or []
    if not crossings or index is None:
        return 0

    match = index.best("bridge", category="structure", theme=theme)
    if match is None or "bridge" not in match.name.lower():
        # A bridge-shaped hole is better than a fence across a river. The
        # crossing stays walkable either way; only the model is missing.
        scene.warnings.append(
            "no bridge model in the library; crossings left as fords")
        return 0

    long_, short, height, along_x = _measure(match.file, match.radius)

    # Size the model by its width, not by the span: as wide as the road,
    # no taller than a footbridge. Then lay as many copies end to end as it
    # takes to reach both banks -- which is how a real bridge of repeating
    # arches or planks is built anyway.
    base_scale = DECK_WIDTH / short
    base_scale = min(base_scale, MAX_BRIDGE_HEIGHT / height)

    placed = 0
    for crossing in crossings:
        scale = base_scale
        segment = long_ * scale
        count = int(min(MAX_SEGMENTS, max(1, math.ceil(crossing.span / segment))))
        if count == 1 and segment > crossing.span * 1.25:
            # A long model over a narrow stream: shrink it to the span
            # rather than leave a deck running far up both banks.
            scale *= crossing.span * 1.15 / segment
            segment = long_ * scale

        rotation = crossing.heading
        if along_x:
            # The model's length runs along X; the heading aims +Z.
            rotation += math.pi / 2

        # The top of the model sits just above the bank, so the deck meets
        # the road and any arches or piers reach down into the water.
        base_y = crossing.deck_height - height * scale + 0.25

        (ax, az), (bx, bz) = crossing.start, crossing.end
        for k in range(count):
            t = (k + 0.5) / count
            scene.instances.append(PlacedInstance(
                name="bridge",
                category="structure",
                asset_name=match.name,
                asset_file=match.file,
                position=(ax + (bx - ax) * t, base_y, az + (bz - az) * t),
                rotation_y=rotation,
                scale=scale,
                radius=segment / 2,
                triangles=int(match.asset.get("triangles") or 0),
                role="bridge",
            ))
            placed += 1

    return placed
