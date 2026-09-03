"""
Stage 3b: decide where everything goes.

Takes a ResolvedScene (objects bound to real meshes, with real dimensions) and
a Terrain, and produces a PlacedScene: every instance with a concrete world
position, rotation and scale.

The three hard requirements, in priority order:
  1. Nothing overlaps        -- objects intersecting each other look broken
  2. Nothing floats or sinks -- every instance sits on the terrain surface
  3. The result is walkable  -- a path is kept clear through the scene

Everything else (visual variety, sensible composition) is layered on top.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .asset_resolution import ResolvedObject, ResolvedScene
from .terrain import Terrain

# Placement attempts before we give up on an instance. Rejection sampling:
# propose a position, reject if it collides, try again.
MAX_ATTEMPTS = 60

# Clear corridor either side of the path centre-line, in metres. Nothing is
# placed inside this -- it's what makes the scene traversable, and what the
# gameplay-flow agent will walk in Week 4.
PATH_HALF_WIDTH = 3.0

# Buildings need near-level ground; scatter objects can handle a slope.
MAX_SLOPE = {
    "building": 0.18,
    "structure": 0.25,
    "prop": 0.40,
    "light_source": 0.30,
    "vegetation": 0.55,
    "rock": 0.80,
}

# Per-instance random scale range, by category. Identical repeated meshes are
# the single biggest giveaway of procedural generation; slight variation reads
# as hand-placed. Buildings vary least -- a house twice the size of its
# neighbour looks wrong rather than natural.
SCALE_JITTER = {
    "building": (0.94, 1.08),
    "structure": (0.95, 1.05),
    "prop": (0.85, 1.15),
    "light_source": (0.92, 1.08),
    "vegetation": (0.75, 1.35),
    "rock": (0.70, 1.45),
}

# Categories that should face the path rather than spin freely.
ORIENTED_CATEGORIES = {"building", "structure"}


@dataclass
class PlacedInstance:
    """One concrete object in the world."""

    name: str                  # what was requested ("house")
    category: str
    asset_name: str            # which model was used
    asset_file: str
    position: Tuple[float, float, float]     # x, y, z in metres
    rotation_y: float                        # radians
    scale: float
    radius: float                            # footprint after scaling
    triangles: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "asset_name": self.asset_name,
            "asset_file": self.asset_file,
            "position": [round(v, 3) for v in self.position],
            "rotation_y": round(self.rotation_y, 4),
            "scale": round(self.scale, 3),
            "radius": round(self.radius, 3),
            "triangles": self.triangles,
        }


@dataclass
class PlacedScene:
    """A fully positioned scene, ready for export."""

    instances: List[PlacedInstance] = field(default_factory=list)
    path: List[Tuple[float, float]] = field(default_factory=list)
    terrain: Optional[Terrain] = None
    spec: object = None
    seed: int = 0
    centre: Optional[Tuple[float, float]] = None
    core_radius: float = 0.0
    warnings: List[str] = field(default_factory=list)
    skipped: int = 0

    @property
    def total_triangles(self) -> int:
        return sum(i.triangles for i in self.instances)

    def summary(self) -> str:
        by_cat = {}
        for inst in self.instances:
            by_cat[inst.category] = by_cat.get(inst.category, 0) + 1
        parts = ", ".join(f"{k}={v}" for k, v in sorted(by_cat.items()))
        return (f"{len(self.instances)} instances placed ({parts}); "
                f"{self.total_triangles:,} triangles")

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "terrain": self.terrain.stats() if self.terrain else None,
            "path": [[round(x, 2), round(z, 2)] for x, z in self.path],
            "settlement_centre": ([round(v, 2) for v in self.centre]
                                  if self.centre else None),
            "settlement_radius": round(self.core_radius, 2),
            "instance_count": len(self.instances),
            "total_triangles": self.total_triangles,
            "instances": [i.to_dict() for i in self.instances],
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Path
# ---------------------------------------------------------------------------

def generate_path(size: float, rng: np.random.Generator,
                  points: int = 6) -> List[Tuple[float, float]]:
    """A gently curving route across the scene, entering and leaving at edges.

    Buildings line it, scatter avoids it, and the player walks it. Giving the
    scene a spine like this is what stops it reading as a random field of
    objects.
    """
    # Enter on one edge, exit roughly opposite, so the path crosses the scene
    vertical = rng.random() < 0.5
    if vertical:
        start = (rng.uniform(0.25, 0.75) * size, 0.0)
        end = (rng.uniform(0.25, 0.75) * size, size)
    else:
        start = (0.0, rng.uniform(0.25, 0.75) * size)
        end = (size, rng.uniform(0.25, 0.75) * size)

    # Control points drift perpendicular to the direction of travel
    xs = np.linspace(start[0], end[0], points)
    zs = np.linspace(start[1], end[1], points)
    drift = size * 0.16
    if vertical:
        xs = xs + rng.normal(0, drift, points)
        xs[0], xs[-1] = start[0], end[0]
    else:
        zs = zs + rng.normal(0, drift, points)
        zs[0], zs[-1] = start[1], end[1]

    xs = np.clip(xs, size * 0.08, size * 0.92)
    zs = np.clip(zs, size * 0.08, size * 0.92)

    # Resample densely so distance queries are accurate
    t = np.linspace(0, 1, points)
    t_fine = np.linspace(0, 1, points * 12)
    return list(zip(np.interp(t_fine, t, xs), np.interp(t_fine, t, zs)))


def _distance_to_path(x: float, z: float,
                      path: List[Tuple[float, float]]) -> float:
    if not path:
        return float("inf")
    pts = np.asarray(path)
    return float(np.min(np.hypot(pts[:, 0] - x, pts[:, 1] - z)))


def _path_direction_at(x: float, z: float,
                       path: List[Tuple[float, float]]) -> float:
    """Heading of the path nearest this point, so buildings can face it."""
    if len(path) < 2:
        return 0.0
    pts = np.asarray(path)
    i = int(np.argmin(np.hypot(pts[:, 0] - x, pts[:, 1] - z)))
    j = min(i + 1, len(path) - 1)
    k = max(j - 1, 0)
    dx = pts[j, 0] - pts[k, 0]
    dz = pts[j, 1] - pts[k, 1]
    return math.atan2(dx, dz)


# ---------------------------------------------------------------------------
# Candidate position generators, one per placement rule
# ---------------------------------------------------------------------------

def _candidate(rule: str, size: float, rng: np.random.Generator,
               path: List[Tuple[float, float]],
               anchor: Optional[Tuple[float, float]]) -> Tuple[float, float]:
    margin = size * 0.06

    if rule == "along_path" and path:
        px, pz = path[rng.integers(0, len(path))]
        heading = _path_direction_at(px, pz, path)
        # Step out perpendicular to the path, on a random side
        side = 1.0 if rng.random() < 0.5 else -1.0
        offset = rng.uniform(PATH_HALF_WIDTH + 1.5, PATH_HALF_WIDTH + 9.0)
        x = px + math.cos(heading) * offset * side
        z = pz - math.sin(heading) * offset * side
        return float(np.clip(x, margin, size - margin)), \
               float(np.clip(z, margin, size - margin))

    if rule == "cluster" and anchor is not None:
        spread = size * 0.11
        return (float(np.clip(anchor[0] + rng.normal(0, spread), margin, size - margin)),
                float(np.clip(anchor[1] + rng.normal(0, spread), margin, size - margin)))

    if rule == "center":
        spread = size * 0.09
        return (float(np.clip(size / 2 + rng.normal(0, spread), margin, size - margin)),
                float(np.clip(size / 2 + rng.normal(0, spread), margin, size - margin)))

    if rule == "perimeter":
        band = size * 0.13
        edge = rng.integers(0, 4)
        along = rng.uniform(margin, size - margin)
        depth = rng.uniform(margin, band)
        return [(along, depth), (along, size - depth),
                (depth, along), (size - depth, along)][int(edge)]

    if rule == "clump" and anchor is not None:
        # Nature grows in clumps, not evenly. Uniform scatter of hundreds of
        # grass tufts reads as confetti; grouping them into patches reads as
        # meadow, undergrowth or a grove.
        spread = size * 0.055
        return (float(np.clip(anchor[0] + rng.normal(0, spread),
                              margin, size - margin)),
                float(np.clip(anchor[1] + rng.normal(0, spread),
                              margin, size - margin)))

    # scatter (and the fallback for anything unrecognised)
    return (float(rng.uniform(margin, size - margin)),
            float(rng.uniform(margin, size - margin)))


def _core_candidate(core, core_r: float, size: float, rng,
                    rule: str, path, anchor) -> Tuple[float, float]:
    """Propose a position inside the settlement.

    Density falls off towards the edge (sqrt of a uniform draw gives uniform
    area density; raising the exponent concentrates buildings towards the
    middle), so the village is tight at its heart and thins outward instead of
    forming an obvious ring.
    """
    if rule == "along_path" and path:
        # Prefer stretches of path that run through the settlement.
        near = [p for p in path
                if math.hypot(p[0] - core[0], p[1] - core[1]) < core_r * 1.15]
        if near:
            px, pz = near[rng.integers(0, len(near))]
            heading = _path_direction_at(px, pz, path)
            side = 1.0 if rng.random() < 0.5 else -1.0
            offset = rng.uniform(PATH_HALF_WIDTH + 1.5, PATH_HALF_WIDTH + 7.0)
            x = px + math.cos(heading) * offset * side
            z = pz - math.sin(heading) * offset * side
            m = size * 0.04
            return float(np.clip(x, m, size - m)), float(np.clip(z, m, size - m))

    if rule == "cluster" and anchor is not None:
        spread = core_r * 0.28
        x = anchor[0] + rng.normal(0, spread)
        z = anchor[1] + rng.normal(0, spread)
    else:
        angle = float(rng.uniform(0, 2 * math.pi))
        dist = core_r * float(rng.random()) ** 0.62
        x = core[0] + math.cos(angle) * dist
        z = core[1] + math.sin(angle) * dist

    m = size * 0.04
    return float(np.clip(x, m, size - m)), float(np.clip(z, m, size - m))


# ---------------------------------------------------------------------------
# Main placement
# ---------------------------------------------------------------------------

def _order_objects(objects: List[ResolvedObject]) -> List[ResolvedObject]:
    """Place big, important things first.

    Rejection sampling fills space greedily, so whatever goes last gets the
    scraps. Buildings must claim their ground before hundreds of trees make
    the scene too crowded to fit them.
    """
    priority = {"building": 0, "structure": 1, "light_source": 2,
                "prop": 3, "rock": 4, "vegetation": 5}
    return sorted(objects,
                  key=lambda o: (priority.get(o.category, 9), -o.radius))


# A settlement needs a centre. Without one, buildings spread evenly across the
# terrain and the result reads as scattered structures rather than a village:
# real settlements are dense in the middle and thin out into farmland and
# wilderness at the edge.
#
# Fraction of scene size occupied by the built-up core, per theme.
CORE_RADIUS = {
    "medieval_village": 0.30,
    "forest_camp": 0.22,
    "desert_outpost": 0.26,
    "sci_fi_base": 0.28,
}

# Categories that belong inside the settlement rather than scattered around it.
CORE_CATEGORIES = {"building", "structure", "light_source"}

# Big vegetation is pushed out of the core -- you don't get mature trees in
# the middle of a village square.
CORE_EXCLUSION = {"vegetation": 0.75, "rock": 0.55}


def _settlement_centre(size: float, path, rng) -> Tuple[float, float]:
    """Put the village on the path, near the middle of the scene.

    Anchoring to the path means the road runs through the settlement rather
    than past it.
    """
    if path:
        mid = path[len(path) // 2]
        jitter = size * 0.05
        return (float(np.clip(mid[0] + rng.normal(0, jitter),
                              size * 0.3, size * 0.7)),
                float(np.clip(mid[1] + rng.normal(0, jitter),
                              size * 0.3, size * 0.7)))
    return (size / 2, size / 2)


class _OccupancyGrid:
    """Spatial hash for collision queries.

    Naive rejection sampling checks every proposal against every placed object.
    That's fine for 40 instances and unusable for 800 -- the dressing pass
    pushes scenes into the hundreds, and O(n^2) with 60 attempts each would
    mean tens of millions of distance checks.

    Bucketing by cell means each proposal only compares against objects in the
    nine neighbouring cells.
    """

    def __init__(self, size: float, cell: float = 6.0):
        self.cell = cell
        self.size = size
        self.buckets: dict = {}
        # The search must reach far enough to find *large* neighbours, not
        # just those near the query point. Widening by only the query's own
        # radius misses a big object stored several cells away whose footprint
        # still reaches the proposal -- which let trees overlap.
        self.max_radius = 0.0

    def _key(self, x: float, z: float):
        return (int(x // self.cell), int(z // self.cell))

    def add(self, x: float, z: float, radius: float) -> None:
        self.buckets.setdefault(self._key(x, z), []).append((x, z, radius))
        self.max_radius = max(self.max_radius, radius)

    def collides(self, x: float, z: float, radius: float) -> bool:
        cx, cz = self._key(x, z)
        reach = radius + self.max_radius
        span = int(reach // self.cell) + 1
        for i in range(cx - span, cx + span + 1):
            for j in range(cz - span, cz + span + 1):
                for ox, oz, orad in self.buckets.get((i, j), ()):
                    if (x - ox) ** 2 + (z - oz) ** 2 < (radius + orad) ** 2:
                        return True
        return False

    def __len__(self) -> int:
        return sum(len(v) for v in self.buckets.values())


def layout_scene(resolved: ResolvedScene, terrain: Terrain,
                 seed: int = 0) -> PlacedScene:
    """Give every object instance a position on the terrain."""
    rng = np.random.default_rng(seed)
    size = terrain.size

    scene = PlacedScene(terrain=terrain, spec=resolved.spec, seed=seed)
    scene.path = generate_path(size, rng)

    # Grade the road before anything is placed, so objects sit on the final
    # surface rather than the pre-graded one.
    terrain.flatten_path(scene.path, half_width=PATH_HALF_WIDTH + 1.5)

    theme = getattr(resolved.spec, "theme", "medieval_village")
    core_r = CORE_RADIUS.get(theme, 0.28) * size
    core = _settlement_centre(size, scene.path, rng)
    scene.centre = core
    scene.core_radius = core_r

    occupied = _OccupancyGrid(size)

    # Cluster anchors are per-object-type, so all barrels group together
    # rather than each barrel starting its own pile.
    anchors: dict = {}

    for obj in _order_objects(resolved.objects):
        if not obj.resolved:
            continue

        rule = obj.placement
        cat = obj.category
        max_slope = MAX_SLOPE.get(cat, 0.4)
        lo_scale, hi_scale = SCALE_JITTER.get(cat, (0.9, 1.1))

        # Buildings and the things that serve them belong in the settlement.
        in_core = cat in CORE_CATEGORIES
        exclusion = CORE_EXCLUSION.get(cat, 0.0) * core_r

        if rule == "clump" and obj.name not in anchors:
            # Several patches rather than one, so a species covers the map in
            # groves instead of a single blob.
            n_clumps = int(np.clip(obj.quantity / 14, 3, 9))
            anchors[obj.name] = [
                (float(rng.uniform(size * 0.1, size * 0.9)),
                 float(rng.uniform(size * 0.1, size * 0.9)))
                for _ in range(n_clumps)
            ]

        if rule == "cluster" and obj.name not in anchors:
            # Clustered props gather at the edge of the settlement, where
            # storage and yards would be, rather than anywhere at random.
            angle = float(rng.uniform(0, 2 * math.pi))
            dist = core_r * float(rng.uniform(0.5, 1.0))
            anchors[obj.name] = (
                float(np.clip(core[0] + math.cos(angle) * dist,
                              size * 0.1, size * 0.9)),
                float(np.clip(core[1] + math.sin(angle) * dist,
                              size * 0.1, size * 0.9)),
            )

        norm = obj.norm_scale        # model units -> metres
        cap = obj.max_scale          # jitter must not breach the size caps
        placed_here = 0
        for _ in range(obj.quantity):
            jitter = float(rng.uniform(lo_scale, hi_scale))
            scale = min(norm * jitter, cap)   # what the exporter applies
            jitter = scale / norm if norm > 0 else jitter
            radius = max(obj.radius * jitter, 0.3)

            spot = None
            for _ in range(MAX_ATTEMPTS):
                anchor = anchors.get(obj.name)
                if isinstance(anchor, list):          # clump: pick a patch
                    anchor = anchor[rng.integers(0, len(anchor))]

                if in_core:
                    x, z = _core_candidate(core, core_r, size, rng, rule,
                                           scene.path, anchor)
                else:
                    x, z = _candidate(rule, size, rng, scene.path, anchor)

                # Keep the settlement's ground clear of forest and boulders
                if exclusion:
                    if math.hypot(x - core[0], z - core[1]) < exclusion:
                        continue

                # Keep objects fully inside the terrain, footprint included
                if not (radius <= x <= size - radius
                        and radius <= z <= size - radius):
                    continue

                # Keep the walkable corridor clear
                if _distance_to_path(x, z, scene.path) < PATH_HALF_WIDTH + radius:
                    continue

                # Don't put a house on a cliff
                if terrain.slope_at(x, z) > max_slope:
                    continue

                # Don't intersect anything already placed
                if occupied.collides(x, z, radius):
                    continue

                spot = (x, z)
                break

            if spot is None:
                scene.skipped += 1
                continue

            x, z = spot

            # Buildings get a level pad, so they don't straddle a slope
            if cat in ("building", "structure"):
                terrain.flatten_disc(x, z, radius * 1.6)

            y = terrain.height_at(x, z)

            if cat in ORIENTED_CATEGORIES:
                # Face the road if close to it, otherwise face the village
                # centre. Buildings that all face a distant road look like a
                # row of sheds; facing inward reads as a settlement.
                if _distance_to_path(x, z, scene.path) < core_r * 0.55:
                    rot = _path_direction_at(x, z, scene.path) + math.pi / 2
                else:
                    rot = math.atan2(core[0] - x, core[1] - z)
                rot += float(rng.normal(0, 0.14))
            else:
                rot = float(rng.uniform(0, 2 * math.pi))

            scene.instances.append(PlacedInstance(
                name=obj.name,
                category=cat,
                asset_name=obj.match.name,
                asset_file=obj.match.file,
                position=(x, y, z),
                rotation_y=rot,
                scale=scale,
                radius=radius,
                triangles=int(obj.match.asset.get("triangles") or 0),
            ))
            occupied.add(x, z, radius)
            placed_here += 1

        if placed_here < obj.quantity:
            scene.warnings.append(
                f"{obj.name}: placed {placed_here}/{obj.quantity} "
                "(ran out of free space)"
            )

    if scene.skipped:
        scene.warnings.append(
            f"{scene.skipped} instance(s) skipped -- scene is at capacity. "
            "A larger terrain or lower density would fit them."
        )

    return scene


# ---------------------------------------------------------------------------
# Validation -- the requirements above, checked rather than assumed
# ---------------------------------------------------------------------------

def validate(scene: PlacedScene) -> dict:
    """Verify the three hard requirements. Used in tests and reported metrics."""
    issues = {"overlaps": [], "off_terrain": [], "floating": [], "on_path": []}
    size = scene.terrain.size if scene.terrain else 0

    for i, a in enumerate(scene.instances):
        ax, ay, az = a.position

        if not (0 <= ax <= size and 0 <= az <= size):
            issues["off_terrain"].append(a.name)

        if scene.terrain:
            expected = scene.terrain.height_at(ax, az)
            if abs(ay - expected) > 0.75:
                issues["floating"].append(a.name)

        if _distance_to_path(ax, az, scene.path) < PATH_HALF_WIDTH * 0.6:
            issues["on_path"].append(a.name)

        for b in scene.instances[i + 1:]:
            bx, _, bz = b.position
            if (ax - bx) ** 2 + (az - bz) ** 2 < (a.radius + b.radius) ** 2 * 0.81:
                issues["overlaps"].append((a.name, b.name))

    return {
        "valid": all(len(v) == 0 for v in issues.values()),
        "instance_count": len(scene.instances),
        **{k: len(v) for k, v in issues.items()},
        "details": issues,
    }
