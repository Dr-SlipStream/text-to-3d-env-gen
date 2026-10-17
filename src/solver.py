"""
The solver: turn a scene graph into positions.

This is the part that replaces the hand-written placement tables. It knows
nothing about villages, forests or deserts -- only about regions, centres,
rings, spacing and avoidance. A new kind of place needs no new code here,
because the language model composes it out of the same relations.

HOW IT WORKS
------------
1. Regions are resolved to concrete geometry. A settlement becomes a disc
   somewhere sensible; wilderness is whatever is left; water comes from the
   terrain's carved mask; an approach is the road corridor.

2. Groups are solved in dependency order, so a campfire has a position before
   the tents that ring it.

3. Each positional relation has a sampler that proposes candidate positions.
   Every proposal is then tested against the same hard constraints as before:
   inside the terrain, out of the water, off steep ground, clear of the road,
   not overlapping anything already placed.

4. Orientation relations run last, once positions are known.

The hard constraints stay rule-based on purpose. A sampler can suggest a spot;
only a check can promise that nothing overlaps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .layout import (MAX_SLOPE, PATH_HALF_WIDTH, SCALE_JITTER, WATER_MARGIN,
                     _OccupancyGrid, _distance_to_path, _path_direction_at,
                     generate_path)
from .scene_graph import Group, Region, Relation, SceneGraph

# Attempts before a single instance is abandoned.
MAX_ATTEMPTS = 60

# How far a "near" or "cluster" relation spreads when no distance is given.
DEFAULT_NEAR = 6.0
DEFAULT_RING = 9.0
DEFAULT_AVOID = 8.0

# Floor on how small a settlement region can be, in metres. Below this the
# groups inside it cannot spread out, however few of them there are.
MIN_SETTLEMENT_RADIUS = 26.0

# How wide a band, as a fraction of a region's radius, over which a scatter
# group thins out rather than stopping dead. A forest that ends on a hard
# circle reads as a cut-out; one that thins over fifteen metres reads as a
# clearing someone made.
EDGE_SOFTNESS = 0.35

# Categories whose edges are softened. A building either fits inside the
# settlement or it doesn't; undergrowth genuinely peters out.
SOFT_EDGE = {"vegetation", "rock"}

# Scatter follows a noise field rather than being spread evenly.
#
# Clump anchors alone do not work: a rejected proposal re-picks a random
# anchor, so over sixty attempts the result averages back out to uniform --
# which is exactly what a generated forest looks like, and why real woodland
# doesn't. A low-frequency field gives each population its own pattern of
# groves and clearings, and the pattern is stable because the field is seeded
# per group.
DENSITY_FIELD_CELLS = 5        # lower = broader groves
DENSITY_FIELD_FLOOR = 0.22     # never quite empty anywhere
DENSITY_FIELD_CONTRAST = 1.7   # higher = starker clearings


# ---------------------------------------------------------------------------
# Region geometry
# ---------------------------------------------------------------------------

@dataclass
class ResolvedRegion:
    """A region with concrete geometry attached."""

    region: Region
    centre: Tuple[float, float]
    radius: float
    # A region may instead be defined by a line (a river, a road) or by a
    # mask (carved water). Samplers pick whichever is present.
    line: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    excluded: bool = False       # wilderness: "everywhere except the others"

    def contains(self, x: float, z: float, terrain=None) -> bool:
        if self.mask is not None and terrain is not None:
            return terrain.is_water(x, z)
        if self.line is not None:
            pts = self.line
            d = float(np.min(np.hypot(pts[:, 0] - x, pts[:, 1] - z)))
            return d <= max(self.radius, 4.0)
        return math.hypot(x - self.centre[0], z - self.centre[1]) <= self.radius


class RegionResolver:
    """Gives every region in the graph a concrete shape on this terrain."""

    def __init__(self, graph: SceneGraph, terrain, path, rng,
                 radii: Optional[Dict[str, float]] = None,
                 categories: Optional[Dict[str, str]] = None):
        self.radii = radii or {}
        self.categories = categories or {}
        self.graph = graph
        self.terrain = terrain
        self.path = path
        self.rng = rng
        self.size = terrain.size
        self.resolved: Dict[str, ResolvedRegion] = {}
        self._resolve_all()

    def _fit_radius(self, region_id: str) -> Optional[float]:
        """Radius that holds a region's fixed-count buildings on plots."""
        plots = 0.0
        for group in self.graph.groups:
            if group.count is None:
                continue
            if not any(r.kind == "inside" and r.target == region_id
                       for r in group.relations):
                continue
            r = self.radii.get(group.id)
            cat = self.categories.get(group.id)
            if r is None:
                continue
            if cat == "building":
                plot = 2 * r + 7.0
            elif cat == "structure" and r >= 2.5:
                plot = 2 * r + 3.0
            else:
                continue
            plots += group.count * plot * plot
        if plots <= 0:
            return None
        return math.sqrt(plots / math.pi) * 1.25

    def _dry_spot(self, radius: float, tries: int = 30) -> Tuple[float, float]:
        """A centre that isn't in the water and isn't hard against an edge."""
        margin = radius * 0.6 + self.size * 0.08
        lo, hi = margin, self.size - margin
        if lo >= hi:
            lo, hi = self.size * 0.35, self.size * 0.65

        best = (self.size / 2, self.size / 2)
        for _ in range(tries):
            x = float(self.rng.uniform(lo, hi))
            z = float(self.rng.uniform(lo, hi))
            if not self.terrain.is_water(x, z, margin=radius * 0.35):
                return x, z
            best = (x, z)
        return best

    def _beside_path(self, radius: float) -> Tuple[float, float]:
        """A settlement centre next to the road, not on top of it.

        Anchoring the centre to the path midpoint puts the focal point --
        a campfire, a well -- in the middle of the road, and everything
        ringed around it then fails the "don't block the path" check. Half a
        camp would simply not be built. Stepping aside by the road's own
        width keeps the road running through the settlement while leaving its
        heart clear.
        """
        pts = np.asarray(self.path)
        mid_index = len(pts) // 2
        mx, mz = pts[mid_index]

        j = min(mid_index + 1, len(pts) - 1)
        k = max(j - 1, 0)
        dx, dz = pts[j][0] - pts[k][0], pts[j][1] - pts[k][1]
        length = math.hypot(dx, dz) or 1.0
        perp = (-dz / length, dx / length)

        side = 1.0 if self.rng.random() < 0.5 else -1.0
        step = (PATH_HALF_WIDTH + 10.0) * side

        margin = radius * 0.5
        return (float(np.clip(mx + perp[0] * step, margin, self.size - margin)),
                float(np.clip(mz + perp[1] * step, margin, self.size - margin)))

    def _resolve_all(self) -> None:
        features = getattr(self.terrain, "features", None)

        # Water and high ground come from the terrain, so they resolve first
        # and the rest position themselves around them.
        for region in self.graph.regions:
            if region.kind == "water" and features is not None:
                line = None
                radius = self.size * 0.1
                centre = (self.size / 2, self.size / 2)

                # Match the region to the water it names, so "the sea" and
                # "the river" in one graph don't both resolve to the river.
                wanted = (region.feature or region.id or "").lower()
                kinds = ["river", "lake", "coast"]
                for kind in kinds:
                    if kind in wanted or (kind == "coast" and any(
                            w in wanted for w in ("sea", "ocean", "shore",
                                                  "beach", "bay", "harbo"))):
                        kinds.remove(kind)
                        kinds.insert(0, kind)
                        break

                for kind in kinds:
                    if kind == "river" and features.rivers:
                        line, half = features.rivers[0]
                        line = np.asarray(line)
                        radius = float(half)
                        centre = tuple(line[len(line) // 2])
                        break
                    if kind == "lake" and features.lakes:
                        cx, cz, r = features.lakes[0]
                        centre, radius = (cx, cz), r
                        break
                    if kind == "coast" and features.coasts:
                        _edge, shore, _depth = features.coasts[0]
                        # A coast is a line: the shoreline. "along" and
                        # "on_shore_of" then sample beside it like a river.
                        line = np.asarray(shore)
                        radius = 3.0
                        centre = tuple(line[len(line) // 2])
                        break
                self.resolved[region.id] = ResolvedRegion(
                    region=region, centre=centre, radius=radius,
                    line=line, mask=getattr(features, "water_mask", None))

            elif region.kind == "high_ground" and features is not None \
                    and features.hills:
                cx, cz, r, _h = features.hills[0]
                self.resolved[region.id] = ResolvedRegion(
                    region=region, centre=(cx, cz), radius=r)

            elif region.kind == "approach":
                pts = np.asarray(self.path) if self.path else None
                centre = (tuple(pts[len(pts) // 2]) if pts is not None
                          else (self.size / 2, self.size / 2))
                self.resolved[region.id] = ResolvedRegion(
                    region=region, centre=centre,
                    radius=PATH_HALF_WIDTH + 8.0, line=pts)

        # Settlements and fields are placed on dry, buildable ground, anchored
        # to the road so it runs through them rather than past.
        for region in self.graph.regions:
            if region.id in self.resolved:
                continue
            if region.kind in ("settlement", "field"):
                radius = self.size * region.extent
                # A settlement has a minimum useful size in metres, not just
                # as a fraction of the scene. A "small" camp on a 200m map is
                # 36m across, which is fine -- but the fixed-count groups
                # inside it need room to spread, or a dozen tents pile into
                # one clearing.
                if region.kind == "settlement":
                    # No bigger than its buildings need. A "large" village
                    # on a 200 m map was 84 m in radius around ten houses,
                    # which reads as farmsteads in a field, not a village.
                    fit = self._fit_radius(region.id)
                    if fit:
                        radius = min(radius, fit)
                radius = max(radius, MIN_SETTLEMENT_RADIUS)
                radius = min(radius, self.size * 0.42)
                if self.path and region.kind == "settlement":
                    centre = self._beside_path(radius)
                    if self.terrain.is_water(*centre, margin=radius * 0.3):
                        centre = self._dry_spot(radius)
                else:
                    centre = self._dry_spot(radius)
                self.resolved[region.id] = ResolvedRegion(
                    region=region, centre=centre, radius=radius)

        # Wilderness is the remainder: the whole scene, minus the others.
        for region in self.graph.regions:
            if region.id in self.resolved:
                continue
            self.resolved[region.id] = ResolvedRegion(
                region=region,
                centre=(self.size / 2, self.size / 2),
                radius=self.size * 0.72,
                excluded=(region.kind == "wilderness"))

    def get(self, region_id: str) -> Optional[ResolvedRegion]:
        return self.resolved.get(region_id)

    def others(self, region_id: str) -> List[ResolvedRegion]:
        """Regions a wilderness should stay out of."""
        return [r for rid, r in self.resolved.items()
                if rid != region_id and r.region.kind in
                ("settlement", "field", "water")]


# ---------------------------------------------------------------------------
# Solved output
# ---------------------------------------------------------------------------

@dataclass
class SolvedInstance:
    group_id: str
    asset: str
    position: Tuple[float, float, float]
    rotation_y: float
    scale_jitter: float
    radius: float


@dataclass
class SolvedScene:
    instances: List[SolvedInstance] = field(default_factory=list)
    anchors: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    regions: Dict[str, ResolvedRegion] = field(default_factory=dict)
    path: List[Tuple[float, float]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    skipped: int = 0

    def positions_of(self, group_id: str) -> List[Tuple[float, float]]:
        return [(i.position[0], i.position[2]) for i in self.instances
                if i.group_id == group_id]


# ---------------------------------------------------------------------------
# Samplers, one per positional relation
# ---------------------------------------------------------------------------

def _sample_in_region(region: ResolvedRegion, rng, size: float,
                      resolver: RegionResolver,
                      clumped: bool = False,
                      anchor: Optional[Tuple[float, float]] = None
                      ) -> Tuple[float, float]:
    """A point inside a region, uniformly or in a clump."""
    margin = size * 0.03

    if region.line is not None:
        # A linear region: pick a point along it and step aside.
        pts = region.line
        px, pz = pts[rng.integers(0, len(pts))]
        offset = rng.uniform(-region.radius, region.radius)
        angle = rng.uniform(0, 2 * math.pi)
        return (float(np.clip(px + math.cos(angle) * offset, margin,
                              size - margin)),
                float(np.clip(pz + math.sin(angle) * offset, margin,
                              size - margin)))

    if region.excluded:
        # Wilderness: anywhere in the scene. The avoid-relations and the
        # region exclusions keep it out of the settled parts.
        if clumped and anchor is not None:
            spread = size * 0.06
            return (float(np.clip(anchor[0] + rng.normal(0, spread),
                                  margin, size - margin)),
                    float(np.clip(anchor[1] + rng.normal(0, spread),
                                  margin, size - margin)))
        return (float(rng.uniform(margin, size - margin)),
                float(rng.uniform(margin, size - margin)))

    if clumped and anchor is not None:
        spread = region.radius * 0.30
        return (float(np.clip(anchor[0] + rng.normal(0, spread),
                              margin, size - margin)),
                float(np.clip(anchor[1] + rng.normal(0, spread),
                              margin, size - margin)))

    # Disc: sqrt for uniform area, a higher power to concentrate inward.
    angle = float(rng.uniform(0, 2 * math.pi))
    dist = region.radius * float(rng.random()) ** 0.62
    return (float(np.clip(region.centre[0] + math.cos(angle) * dist,
                          margin, size - margin)),
            float(np.clip(region.centre[1] + math.sin(angle) * dist,
                          margin, size - margin)))


class DensityField:
    """A smooth 0..1 field over the scene, deciding where a population thickens.

    Built from the same value noise as the terrain, at a low frequency, so a
    forest gets groves and glades instead of an even sprinkle. Each group gets
    its own field, so trees and undergrowth do not thicken in exactly the same
    places -- which would look just as artificial.
    """

    def __init__(self, size: float, seed: int, cells: int = DENSITY_FIELD_CELLS,
                 resolution: int = 64):
        from .terrain import _value_noise_layer

        rng = np.random.default_rng(seed)
        field = _value_noise_layer(resolution, cells, rng)

        lo, hi = float(field.min()), float(field.max())
        if hi - lo > 1e-9:
            field = (field - lo) / (hi - lo)

        # Push it towards extremes: dense stands and real clearings, rather
        # than a gentle wobble around the mean.
        field = field ** DENSITY_FIELD_CONTRAST
        self.field = DENSITY_FIELD_FLOOR + field * (1 - DENSITY_FIELD_FLOOR)
        self.size = size
        self.res = resolution

    def at(self, x: float, z: float) -> float:
        gx = int(np.clip(x / self.size * (self.res - 1), 0, self.res - 1))
        gz = int(np.clip(z / self.size * (self.res - 1), 0, self.res - 1))
        return float(self.field[gz, gx])


def _sample_ring(anchor: Tuple[float, float], radius: float, index: int,
                 total: int, rng, size: float) -> Tuple[float, float]:
    """Evenly spaced around a point, with a little irregularity.

    Perfectly even spacing reads as a machine; a small angular and radial
    jitter reads as people choosing where to pitch a tent.
    """
    base = (index / max(total, 1)) * 2 * math.pi
    angle = base + float(rng.normal(0, 0.18))
    dist = radius * float(rng.uniform(0.82, 1.18))
    margin = size * 0.03
    return (float(np.clip(anchor[0] + math.cos(angle) * dist,
                          margin, size - margin)),
            float(np.clip(anchor[1] + math.sin(angle) * dist,
                          margin, size - margin)))


def _sample_near(anchor: Tuple[float, float], distance: float, rng,
                 size: float) -> Tuple[float, float]:
    spread = max(distance, 2.0)
    margin = size * 0.03
    angle = float(rng.uniform(0, 2 * math.pi))
    dist = abs(float(rng.normal(0, spread * 0.6))) + spread * 0.35
    return (float(np.clip(anchor[0] + math.cos(angle) * dist,
                          margin, size - margin)),
            float(np.clip(anchor[1] + math.sin(angle) * dist,
                          margin, size - margin)))


def _sample_along(line: np.ndarray, offset: float, rng,
                  size: float) -> Tuple[float, float]:
    """Beside a line -- a road verge, a riverbank, the foot of a wall."""
    i = int(rng.integers(0, len(line)))
    px, pz = line[i]

    j = min(i + 1, len(line) - 1)
    k = max(j - 1, 0)
    dx, dz = line[j][0] - line[k][0], line[j][1] - line[k][1]
    length = math.hypot(dx, dz) or 1.0
    perp = (-dz / length, dx / length)

    side = 1.0 if rng.random() < 0.5 else -1.0
    step = offset * float(rng.uniform(0.6, 1.6)) * side
    margin = size * 0.02
    return (float(np.clip(px + perp[0] * step, margin, size - margin)),
            float(np.clip(pz + perp[1] * step, margin, size - margin)))


def _sample_shore(region: ResolvedRegion, terrain, rng, size: float,
                  band: float = 6.0) -> Tuple[float, float]:
    """Just outside a water region -- on the land, close to the edge."""
    for _ in range(40):
        if region.line is not None:
            x, z = _sample_along(region.line, region.radius + band * 0.6,
                                 rng, size)
        else:
            angle = float(rng.uniform(0, 2 * math.pi))
            dist = region.radius + float(rng.uniform(1.0, band))
            x = region.centre[0] + math.cos(angle) * dist
            z = region.centre[1] + math.sin(angle) * dist
            margin = size * 0.02
            x = float(np.clip(x, margin, size - margin))
            z = float(np.clip(z, margin, size - margin))
        if not terrain.is_water(x, z, margin=0.5):
            return x, z
    return (size / 2, size / 2)


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

def solve(graph: SceneGraph, terrain, radii: Dict[str, float],
          categories: Dict[str, str], seed: int = 0,
          path: Optional[List[Tuple[float, float]]] = None) -> SolvedScene:
    """Place every group in the graph.

    `radii` and `categories` map group id -> footprint radius in metres and
    -> object category, which the caller gets from asset resolution. The
    solver itself has no opinion about what a tent is; it only needs to know
    how much room it takes and how fussy it is about slope and water.
    """
    rng = np.random.default_rng(seed)
    size = terrain.size

    scene = SolvedScene()
    from .crossings import choose_road
    scene.path = (list(path) if path
                  else choose_road(terrain, size, rng, generate_path))

    # Record bridges where the road crosses water, before anything is placed,
    # so placement and walkability both know the crossing is there.
    from .crossings import plan_crossings
    plan_crossings(terrain, scene.path)

    resolver = RegionResolver(graph, terrain, scene.path, rng,
                              radii=radii, categories=categories)
    scene.regions = resolver.resolved

    occupied = _OccupancyGrid(size)
    clump_anchors: Dict[str, Tuple[float, float]] = {}

    for group in graph.resolution_order():
        radius_base = radii.get(group.id, 1.0)
        category = categories.get(group.id, "prop")
        max_slope = MAX_SLOPE.get(category, 0.4)
        water_margin = WATER_MARGIN.get(category, 2.0)
        lo_jit, hi_jit = SCALE_JITTER.get(category, (0.9, 1.1))

        placement = group.placement
        rel_kind = placement.kind if placement else "inside"
        target_id = placement.target if placement else ""

        # Where the relation points: a region, or the positions of a group
        # already placed.
        region = resolver.get(target_id)
        anchor_positions = scene.positions_of(target_id)
        anchor = anchor_positions[0] if anchor_positions else None

        if region is None and anchor is None and target_id:
            scene.warnings.append(
                f"{group.id}: {rel_kind}({target_id}) could not be resolved; "
                "placed without it")

        # How many to place.
        if group.count is not None:
            wanted = group.count
        else:
            if region is not None and region.line is not None:
                # A linear region -- a river, a road -- is a strip, not a
                # disc. Treating it as a disc of the strip's half-width
                # underestimates its area by an order of magnitude, so a
                # riverbank asked for three reeds and got one.
                pts = region.line
                length = float(np.sum(np.hypot(np.diff(pts[:, 0]),
                                               np.diff(pts[:, 1]))))
                area = length * region.radius * 2.0
            elif region is not None and not region.excluded:
                area = math.pi * region.radius ** 2
            else:
                area = size * size * 0.55        # wilderness, roughly
            wanted = int(round((group.density or 0.0) * area / 100.0))
            wanted = max(0, min(wanted, 400))

        # Avoidance constraints, resolved once.
        avoid: List[Tuple[Tuple[float, float], float]] = []
        for rel in group.avoidances:
            other = resolver.get(rel.target)
            distance = rel.distance or DEFAULT_AVOID
            if other is not None:
                avoid.append((other.centre, other.radius + distance))
            else:
                for pos in scene.positions_of(rel.target):
                    avoid.append((pos, distance))

        # Every scattered population gets its own pattern of groves and
        # clearings. Hashing the id keeps it stable across runs.
        field = None
        if rel_kind == "scatter_in":
            import zlib
            field = DensityField(
                size, seed + zlib.crc32(group.id.encode("utf-8")) % 100000)

        # A ring has to be big enough to hold what is going on it. The model
        # asks for "twelve tents at nine metres" without checking whether
        # twelve tents fit on a 56-metre circle; if they don't, the surplus
        # is simply dropped and the camp comes out half-built. Widening the
        # ring keeps the arrangement the model asked for.
        if rel_kind == "ring_around" and wanted > 1:
            requested = placement.distance or DEFAULT_RING
            # Circumference needed, with a little breathing room between
            # neighbours.
            # 1.7x rather than a tight fit: positions on the ring are
            # jittered so the arrangement doesn't look machined, and two
            # neighbours jittered towards each other need the slack.
            needed = wanted * radius_base * 2.0 * 1.7 / (2 * math.pi)
            if needed > requested:
                scene.warnings.append(
                    f"{group.id}: ring widened {requested:.0f}m -> "
                    f"{needed:.0f}m to fit {wanted}")
                placement = placement.model_copy(
                    update={"distance": round(needed, 1)})

        if rel_kind == "scatter_in" and group.id not in clump_anchors:
            n_clumps = int(np.clip(wanted / 14, 3, 9))
            clump_anchors[group.id] = [
                _sample_in_region(region, rng, size, resolver)
                if region is not None else
                (float(rng.uniform(0, size)), float(rng.uniform(0, size)))
                for _ in range(n_clumps)
            ]

        placed_here = 0
        for index in range(wanted):
            jitter = float(rng.uniform(lo_jit, hi_jit))
            radius = max(radius_base * jitter, 0.3)

            spot = None
            for _ in range(MAX_ATTEMPTS):
                # -- propose --------------------------------------------
                if rel_kind == "at_center_of" and region is not None:
                    jx = region.radius * 0.10
                    x = region.centre[0] + float(rng.normal(0, jx))
                    z = region.centre[1] + float(rng.normal(0, jx))
                elif rel_kind == "ring_around" and anchor is not None:
                    x, z = _sample_ring(anchor,
                                        placement.distance or DEFAULT_RING,
                                        index, wanted, rng, size)
                elif rel_kind == "cluster_near" and anchor is not None:
                    x, z = _sample_near(anchor,
                                        placement.distance or DEFAULT_NEAR,
                                        rng, size)
                elif rel_kind == "along" and region is not None \
                        and region.line is not None:
                    x, z = _sample_along(region.line,
                                         placement.distance or region.radius,
                                         rng, size)
                elif rel_kind == "along" and anchor is not None:
                    x, z = _sample_near(anchor,
                                        placement.distance or DEFAULT_NEAR,
                                        rng, size)
                elif rel_kind == "on_shore_of" and region is not None:
                    x, z = _sample_shore(region, terrain, rng, size,
                                         band=placement.distance or 6.0)
                elif rel_kind == "scatter_in":
                    anchors = clump_anchors.get(group.id)
                    clump = (anchors[int(rng.integers(0, len(anchors)))]
                             if anchors else None)
                    x, z = _sample_in_region(
                        region, rng, size, resolver,
                        clumped=True, anchor=clump) if region is not None \
                        else (float(rng.uniform(0, size)),
                              float(rng.uniform(0, size)))
                elif region is not None:
                    x, z = _sample_in_region(region, rng, size, resolver)
                else:
                    margin = size * 0.04
                    x = float(rng.uniform(margin, size - margin))
                    z = float(rng.uniform(margin, size - margin))

                # -- the hard constraints, unchanged --------------------
                if not (radius <= x <= size - radius
                        and radius <= z <= size - radius):
                    continue
                if terrain.is_water(x, z, margin=water_margin):
                    continue
                # Bridge landings stay clear, or a tree blocks the way on.
                if terrain.on_bridge(x, z, margin=radius):
                    continue
                if terrain.slope_at(x, z) > max_slope:
                    continue
                if _distance_to_path(x, z, scene.path) < \
                        PATH_HALF_WIDTH + radius:
                    continue
                # Avoidance. For scatter, the boundary is a gradient rather
                # than a wall: the closer to the forbidden centre, the less
                # likely a proposal survives.
                blocked = False
                for (ax, az), ad in avoid:
                    d = math.hypot(x - ax, z - az)
                    if d >= ad:
                        continue
                    if category in SOFT_EDGE and rel_kind == "scatter_in":
                        band = ad * EDGE_SOFTNESS
                        if d > ad - band:
                            # Inside the soft band: thin out with distance.
                            keep = (d - (ad - band)) / max(band, 1e-6)
                            if rng.random() < keep * 0.55:
                                continue
                    blocked = True
                    break
                if blocked:
                    continue
                # Wilderness stays out of the settled regions, thinning at
                # the boundary rather than ending on a hard circle.
                if region is not None and region.excluded:
                    intruding = False
                    for other in resolver.others(region.region.id):
                        if other.region.kind == "water":
                            if other.contains(x, z, terrain):
                                intruding = True
                                break
                            continue
                        d = math.hypot(x - other.centre[0],
                                       z - other.centre[1])
                        if d >= other.radius:
                            continue
                        if category in SOFT_EDGE:
                            band = other.radius * EDGE_SOFTNESS
                            if d > other.radius - band:
                                keep = ((d - (other.radius - band))
                                        / max(band, 1e-6))
                                if rng.random() < keep * 0.55:
                                    continue
                        intruding = True
                        break
                    if intruding:
                        continue
                # Thin out where the field is low: this is what produces
                # glades and stands rather than an even sprinkle.
                if field is not None and rng.random() > field.at(x, z):
                    continue

                if occupied.collides(x, z, radius):
                    continue

                spot = (x, z)
                break

            if spot is None:
                scene.skipped += 1
                continue

            x, z = spot
            if category in ("building", "structure"):
                terrain.flatten_disc(x, z, radius * 1.6)
            y = terrain.height_at(x, z)

            # -- orientation ------------------------------------------
            orient = group.orientation
            rot = float(rng.uniform(0, 2 * math.pi))
            if orient is not None:
                if orient.kind == "faces":
                    targets = scene.positions_of(orient.target)
                    focus = (targets[0] if targets else
                             (resolver.get(orient.target).centre
                              if resolver.get(orient.target) else None))
                    if focus is not None:
                        rot = math.atan2(focus[0] - x, focus[1] - z)
                        rot += float(rng.normal(0, 0.10))
                elif orient.kind == "aligns_with":
                    other = resolver.get(orient.target)
                    line = (other.line if other is not None else None)
                    if line is None and scene.path:
                        line = np.asarray(scene.path)
                    if line is not None:
                        rot = _path_direction_at(
                            x, z, [tuple(p) for p in line]) + math.pi / 2
                        rot += float(rng.normal(0, 0.12))

            scene.instances.append(SolvedInstance(
                group_id=group.id, asset=group.asset,
                position=(x, y, z), rotation_y=rot,
                scale_jitter=jitter, radius=radius))
            occupied.add(x, z, radius)
            placed_here += 1

            if placed_here == 1 and group.count is not None:
                scene.anchors[group.id] = (x, z)

        if wanted and placed_here < wanted:
            scene.warnings.append(
                f"{group.id}: placed {placed_here}/{wanted}")

    return scene
