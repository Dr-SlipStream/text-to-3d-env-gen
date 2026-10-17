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

# Lowest a deck may sit above the water, in metres.
DECK_CLEARANCE = 0.8

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
                   terrain.water_level + DECK_CLEARANCE)
        crossing = Crossing(start=start, end=end, deck_height=float(deck))
        if crossing.span >= MIN_SPAN:
            crossings.append(crossing)
        i = j

    return crossings


def _path_length(path) -> float:
    return sum(math.dist(path[k], path[k + 1]) for k in range(len(path) - 1))


def crossing_cost(terrain, path, full_length: float = 0.0) -> float:
    """How good a road is around water. Lower is better.

    A river splits the map in two, and the road is what joins the halves:
    a scene with a river and a road that never crosses it leaves half the
    place behind water. So with a river the target is exactly one crossing,
    short and square; with only lakes or sea the target is none. A road that
    wanders along the riverbed needs a forty-metre span or three bridges in
    a row, and reads as a mistake either way.

    `full_length` is the road's length before any trimming at the sea; a
    road cut much shorter is a dead end and costs accordingly.
    """
    crossings = find_crossings(terrain, path)
    features = getattr(terrain, "features", None)
    wants = 1 if (features is not None and features.rivers) else 0
    cost = sum(c.span for c in crossings)
    cost += 40.0 * abs(len(crossings) - wants)
    if full_length > 0:
        lost = max(0.0, full_length - _path_length(path)) / full_length
        cost += 30.0 * lost
    return cost


def trim_to_land(terrain, path):
    """Drop the ends of a road that run into the sea.

    Roads enter and leave at the map edges. When an edge is coast, the road
    should stop at the shore -- a bridge out to the horizon is not a
    crossing. Only the ends are trimmed; water in the middle is a river or a
    lake and still gets a bridge.
    """
    if not path or terrain is None or terrain.water_level is None:
        return path
    # Only the open sea cuts a road short. A river or lake at the edge of
    # the map is still something to bridge -- trimming there let the road
    # chooser dodge every bridge by stopping at the bank.
    is_sea = getattr(terrain, "is_sea", None)
    if is_sea is None:
        return list(path)
    start, end = 0, len(path)
    while start < end and is_sea(*path[start], margin=1.0):
        start += 1
    while end > start and is_sea(*path[end - 1], margin=1.0):
        end -= 1
    return list(path[start:end])


def choose_road(terrain, size: float, rng, generate, tries: int = 10):
    """Generate several roads and keep the one that crosses water best."""
    if terrain is None or terrain.water_level is None:
        return generate(size, rng)

    best, best_cost = None, None
    for _ in range(tries):
        raw = generate(size, rng)
        candidate = trim_to_land(terrain, raw)
        if len(candidate) < 12:
            # Barely any road left once the sea is cut off; try another.
            cost = 1e6 - len(candidate)
        else:
            cost = crossing_cost(terrain, candidate, _path_length(raw))
        if best is None or cost < best_cost:
            best, best_cost = candidate, cost
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
    if path and terrain.features is not None:
        crossings = crossings + connect_banks(terrain, path)
    return crossings


# Banks smaller than this share of the map are left as they are: a sliver of
# land in a river bend is scenery, not somewhere the player needs to go.
MIN_BANK_SHARE = 0.04
MAX_FOOTBRIDGE_SPAN = 45.0


def _land_grid(terrain, resolution: int):
    cell = terrain.size / resolution
    land = np.ones((resolution, resolution), dtype=bool)
    for r in range(resolution):
        for c in range(resolution):
            x, z = (c + 0.5) * cell, (r + 0.5) * cell
            if terrain.on_bridge(x, z, margin=cell * 0.5):
                continue
            # The same test the playtest's walkability grid uses, so a bank
            # counted as joined here is joined for the bot too.
            if terrain.is_water(x, z, margin=cell * 0.4) \
                    or terrain.slope_at(x, z) > 0.45:
                land[r, c] = False
    return land, cell


def _components(land):
    from collections import deque
    rows, cols = land.shape
    label = np.full(land.shape, -1, dtype=int)
    n = 0
    for r in range(rows):
        for c in range(cols):
            if not land[r, c] or label[r, c] >= 0:
                continue
            label[r, c] = n
            q = deque([(r, c)])
            while q:
                a, b = q.popleft()
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    na, nb = a + da, b + db
                    if 0 <= na < rows and 0 <= nb < cols and land[na, nb] \
                            and label[na, nb] < 0:
                        label[na, nb] = n
                        q.append((na, nb))
            n += 1
    return label, n


def connect_banks(terrain, path, resolution: int = 40) -> List[Crossing]:
    """Footbridges to any sizeable bank the road leaves cut off.

    A river that bends through a V can meet the map edge twice. The road
    crosses one arm, and everything beyond the other is unreachable -- the
    RL playtest found up to 46% of open ground cut off this way. Each such
    bank gets one bridge at the narrowest point between it and the land the
    road already reaches, and the search repeats until nothing big is left.
    """
    if terrain.water_level is None or not path:
        return []

    added: List[Crossing] = []
    size = terrain.size
    for _ in range(4):
        land, cell = _land_grid(terrain, resolution)
        label, n = _components(land)
        sr = int(np.clip(path[0][1] / cell, 0, resolution - 1))
        sc = int(np.clip(path[0][0] / cell, 0, resolution - 1))
        home = label[sr, sc]
        if home < 0:
            return added
        reached = label == home

        best = None
        for k in range(n):
            if k == home:
                continue
            comp = label == k
            if comp.sum() < MIN_BANK_SHARE * resolution * resolution:
                continue
            a = np.argwhere(reached)
            b = np.argwhere(comp)
            # Nearest pair of cells, one on each bank.
            d = np.hypot(a[:, None, 0] - b[None, :, 0],
                         a[:, None, 1] - b[None, :, 1])
            i, j = np.unravel_index(int(np.argmin(d)), d.shape)
            span = float(d[i, j]) * cell
            if best is None or span < best[0]:
                best = (span, a[i], b[j])

        if best is None or best[0] > MAX_FOOTBRIDGE_SPAN:
            return added

        _, (ar, ac), (br, bc) = best
        ax, az = (ac + 0.5) * cell, (ar + 0.5) * cell
        bx, bz = (bc + 0.5) * cell, (br + 0.5) * cell
        dx, dz = bx - ax, bz - az
        length = math.hypot(dx, dz) or 1.0
        ux, uz = dx / length, dz / length
        reach = BANK_OVERLAP + cell
        start = (float(np.clip(ax - ux * reach, 0, size)),
                 float(np.clip(az - uz * reach, 0, size)))
        end = (float(np.clip(bx + ux * reach, 0, size)),
               float(np.clip(bz + uz * reach, 0, size)))
        deck = max(terrain.height_at(ax, az), terrain.height_at(bx, bz),
                   terrain.water_level + DECK_CLEARANCE)
        bridge = Crossing(start=start, end=end, deck_height=float(deck))
        terrain.features.bridges.append(bridge)
        added.append(bridge)

    return added


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


def _deck_level(mesh) -> Optional[float]:
    """Height of the walking surface in a bridge model, in model units.

    The top of a bridge model is often not its deck: railings, posts and
    arches stand above the planks. Placing the model by its top put the
    deck of a post-and-rail bridge under the river. The deck is the largest
    upward-facing surface, so this weighs every upward face by its area
    and takes the height where most of that area sits.
    """
    try:
        normals = mesh.face_normals
        areas = mesh.area_faces
        heights = mesh.triangles_center[:, 1]
    except Exception:
        return None
    up = normals[:, 1] > 0.8
    if not up.any():
        return None
    lo, hi = float(mesh.bounds[0][1]), float(mesh.bounds[1][1])
    if hi - lo < 1e-6:
        return hi
    bins = np.linspace(lo, hi, 25)
    weight, edges = np.histogram(heights[up], bins=bins, weights=areas[up])
    k = int(np.argmax(weight))
    return float((edges[k] + edges[k + 1]) / 2)


def _measure(asset_file: str, fallback_radius: float):
    """(long, short, height, long_axis_is_x, deck) for a model.

    Model units throughout; `deck` is measured up from the model's base.
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
        deck = _deck_level(mesh)
        deck = (deck - float(lo[1])) if deck is not None else ey
        return long_, max(short, 1e-3), max(ey, 1e-3), along_x, deck
    except Exception:
        d = max(fallback_radius * 2.0, 1e-3)
        return d, d * 0.5, d * 0.3, None, d * 0.3


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

    long_, short, height, along_x, deck_model = _measure(match.file,
                                                         match.radius)

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

        # The deck -- not the top of the railings -- sits just above the
        # bank, so it meets the road and piers reach down into the water.
        base_y = crossing.deck_height + 0.2 - deck_model * scale

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
