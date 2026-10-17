"""
Stage 3a: generate the ground.

Produces a heightmap -- a 2D grid of elevations -- matching the scene's terrain
type. Objects are later placed *on* this surface, so it must be queryable at
arbitrary world coordinates, not just at grid points.

Implementation note: this uses fractal value noise written directly in numpy
rather than the `noise` PyPI package, which needs a C compiler and is a common
source of install failure on Windows. Doing it in numpy costs a few lines,
removes a whole class of setup problems, and is fully deterministic given a
seed -- which matters for reproducible results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from . import vocab


@dataclass(frozen=True)
class TerrainPreset:
    """Noise parameters that give a terrain type its character."""

    amplitude: float        # peak-to-trough height in metres
    octaves: int            # layers of detail
    base_cells: int         # grid resolution of the coarsest layer
    persistence: float      # how quickly finer octaves lose strength
    ridged: bool = False    # sharp ridges (mountains) vs smooth hills
    flat_bias: float = 0.0  # 0 = natural, 1 = pushed towards flat

    # Ground colours, blended by elevation. Low-poly assets look cheap on flat
    # grey ground and stylised on tinted, varied ground -- this is one of the
    # cheapest visual-quality wins available.
    colour_low: tuple = (0.35, 0.45, 0.24)
    colour_high: tuple = (0.52, 0.58, 0.38)
    # Bare earth, used for the road corridor and trampled village ground.
    # Without it a settlement sits on unbroken lawn and never reads as
    # somewhere people live.
    colour_dirt: tuple = (0.44, 0.35, 0.24)


# Amplitude is deliberately modest: dramatic terrain looks good in isolation
# but makes a village unplaceable, and playability matters more here.
PRESETS = {
    "grassland": TerrainPreset(
        amplitude=4.5, octaves=4, base_cells=3, persistence=0.5,
        flat_bias=0.18,
        colour_low=(0.29, 0.42, 0.20), colour_high=(0.48, 0.56, 0.30),
        colour_dirt=(0.46, 0.37, 0.25)),
    "forest_floor": TerrainPreset(
        amplitude=3.5, octaves=4, base_cells=5, persistence=0.5,
        flat_bias=0.2,
        colour_low=(0.20, 0.29, 0.16), colour_high=(0.33, 0.40, 0.22),
        colour_dirt=(0.32, 0.26, 0.18)),
    "desert_sand": TerrainPreset(
        amplitude=5.5, octaves=3, base_cells=3, persistence=0.45,
        flat_bias=0.12,
        colour_low=(0.72, 0.61, 0.40), colour_high=(0.85, 0.76, 0.55),
        colour_dirt=(0.62, 0.50, 0.33)),
    "rocky": TerrainPreset(
        amplitude=9.0, octaves=5, base_cells=4, persistence=0.55, ridged=True,
        colour_low=(0.34, 0.34, 0.32), colour_high=(0.58, 0.57, 0.54),
        colour_dirt=(0.40, 0.36, 0.31)),
    "barren_rock": TerrainPreset(
        amplitude=7.0, octaves=4, base_cells=4, persistence=0.5, ridged=True,
        flat_bias=0.1,
        colour_low=(0.30, 0.26, 0.25), colour_high=(0.50, 0.45, 0.42),
        colour_dirt=(0.38, 0.32, 0.29)),
    "snow": TerrainPreset(
        amplitude=6.0, octaves=4, base_cells=4, persistence=0.5,
        flat_bias=0.1,
        colour_low=(0.80, 0.83, 0.88), colour_high=(0.95, 0.96, 0.98),
        colour_dirt=(0.55, 0.52, 0.50)),
    "swamp": TerrainPreset(
        amplitude=2.2, octaves=4, base_cells=6, persistence=0.55,
        flat_bias=0.35,
        colour_low=(0.20, 0.24, 0.14), colour_high=(0.31, 0.35, 0.20),
        colour_dirt=(0.29, 0.24, 0.17)),
    "beach_sand": TerrainPreset(
        amplitude=3.5, octaves=3, base_cells=3, persistence=0.45,
        flat_bias=0.25,
        colour_low=(0.80, 0.73, 0.55), colour_high=(0.62, 0.66, 0.40),
        colour_dirt=(0.66, 0.57, 0.40)),
}


def _smoothstep(t: np.ndarray) -> np.ndarray:
    """Hermite curve: removes the visible grid artefacts of linear blending."""
    return t * t * (3.0 - 2.0 * t)


def _value_noise_layer(res: int, cells: int, rng: np.random.Generator) -> np.ndarray:
    """One octave: random values on a coarse grid, smoothly interpolated."""
    grid = rng.random((cells + 1, cells + 1))

    coords = np.linspace(0, cells, res, endpoint=False)
    i = np.floor(coords).astype(int)
    frac = _smoothstep(coords - i)

    iy, ix = np.meshgrid(i, i, indexing="ij")
    fy, fx = np.meshgrid(frac, frac, indexing="ij")

    v00 = grid[iy, ix]
    v01 = grid[iy, ix + 1]
    v10 = grid[iy + 1, ix]
    v11 = grid[iy + 1, ix + 1]

    top = v00 * (1 - fx) + v01 * fx
    bot = v10 * (1 - fx) + v11 * fx
    return top * (1 - fy) + bot * fy


def _fractal_noise(res: int, preset: TerrainPreset,
                   rng: np.random.Generator) -> np.ndarray:
    """Sum several octaves of value noise into a 0..1 heightfield."""
    total = np.zeros((res, res), dtype=np.float64)
    amplitude = 1.0
    total_amplitude = 0.0
    cells = preset.base_cells

    for _ in range(preset.octaves):
        layer = _value_noise_layer(res, cells, rng)
        if preset.ridged:
            layer = 1.0 - np.abs(layer * 2.0 - 1.0)
        total += layer * amplitude
        total_amplitude += amplitude
        amplitude *= preset.persistence
        cells *= 2

    total /= total_amplitude

    lo, hi = total.min(), total.max()
    if hi - lo > 1e-9:
        total = (total - lo) / (hi - lo)

    if preset.flat_bias > 0:
        total = total * (1 - preset.flat_bias) + 0.5 * preset.flat_bias

    return total


class Terrain:
    """A generated ground surface, queryable at any world coordinate.

    World space runs 0..size on X and Z, with Y as elevation -- matching the
    convention used by glTF and Unity.
    """

    def __init__(self, heights: np.ndarray, size: float, terrain_type: str,
                 seed: int):
        self.heights = heights
        self.size = float(size)
        self.terrain_type = terrain_type
        self.seed = seed
        self.res = heights.shape[0]
        self.preset = PRESETS.get(terrain_type, PRESETS["grassland"])
        # Populated by apply_features(); until then the terrain has none.
        self.features = None

    @classmethod
    def generate(cls, terrain_type: str, size_metres: float,
                 seed: int = 0, resolution: int = 128) -> "Terrain":
        preset = PRESETS.get(terrain_type, PRESETS["grassland"])
        rng = np.random.default_rng(seed)
        field = _fractal_noise(resolution, preset, rng)
        return cls(field * preset.amplitude, size_metres, terrain_type, seed)

    @classmethod
    def from_spec(cls, spec, seed: int = 0, resolution: int = 128) -> "Terrain":
        return cls.generate(
            spec.terrain.type,
            vocab.TERRAIN_SIZE_METRES[spec.terrain.size],
            seed=seed,
            resolution=resolution,
        )

    # -- queries ----------------------------------------------------------
    def height_at(self, x: float, z: float) -> float:
        """Elevation at a world position, bilinearly interpolated.

        Objects must sit exactly on the surface -- sampling the nearest grid
        cell would leave them visibly floating or sunken on slopes.
        """
        gx = np.clip(x / self.size * (self.res - 1), 0, self.res - 1)
        gz = np.clip(z / self.size * (self.res - 1), 0, self.res - 1)

        x0, z0 = int(np.floor(gx)), int(np.floor(gz))
        x1, z1 = min(x0 + 1, self.res - 1), min(z0 + 1, self.res - 1)
        fx, fz = gx - x0, gz - z0

        h = (self.heights[z0, x0] * (1 - fx) * (1 - fz)
             + self.heights[z0, x1] * fx * (1 - fz)
             + self.heights[z1, x0] * (1 - fx) * fz
             + self.heights[z1, x1] * fx * fz)
        return float(h)

    def slope_at(self, x: float, z: float, sample: float = 2.0) -> float:
        """Steepness as a gradient magnitude (rise over run).

        Used to keep buildings off cliff faces -- a house on a 45-degree slope
        looks broken even when nothing overlaps.
        """
        h = self.height_at(x, z)
        dx = abs(self.height_at(min(x + sample, self.size), z) - h)
        dz = abs(self.height_at(x, min(z + sample, self.size)) - h)
        return float(np.hypot(dx, dz) / sample)

    def flatten_disc(self, x: float, z: float, radius: float,
                     strength: float = 0.85) -> None:
        """Level the ground under a footprint.

        Buildings need a flat base -- a house straddling a slope has one
        corner buried and another in mid-air. We blend the terrain towards
        the centre height across the footprint, feathered at the edge so it
        doesn't leave a visible circular step.
        """
        target = self.height_at(x, z)
        cell = self.size / (self.res - 1)
        r_cells = max(1, int(np.ceil(radius / cell)))

        cx = int(np.clip(x / self.size * (self.res - 1), 0, self.res - 1))
        cz = int(np.clip(z / self.size * (self.res - 1), 0, self.res - 1))

        x0, x1 = max(0, cx - r_cells), min(self.res, cx + r_cells + 1)
        z0, z1 = max(0, cz - r_cells), min(self.res, cz + r_cells + 1)
        if x0 >= x1 or z0 >= z1:
            return

        zz, xx = np.meshgrid(np.arange(z0, z1), np.arange(x0, x1),
                             indexing="ij")
        dist = np.hypot((xx - cx) * cell, (zz - cz) * cell)
        # 1 at the centre, 0 at the rim
        weight = np.clip(1.0 - dist / max(radius, 1e-6), 0.0, 1.0)
        weight = _smoothstep(weight) * strength

        patch = self.heights[z0:z1, x0:x1]
        self.heights[z0:z1, x0:x1] = patch * (1 - weight) + target * weight

    def flatten_path(self, path, half_width: float = 4.0,
                     strength: float = 0.75) -> None:
        """Level the ground along a route.

        A road that rolls over every bump doesn't read as a road. Flattening
        the corridor gives it the graded look of something built rather than
        painted on.
        """
        if not path:
            return
        pts = np.asarray(path, dtype=float)
        cell = self.size / (self.res - 1)

        xs = np.linspace(0, self.size, self.res)
        zz, xx = np.meshgrid(xs, xs, indexing="ij")

        # Distance from every grid point to the nearest path sample.
        d = np.min(np.hypot(xx[..., None] - pts[:, 0],
                            zz[..., None] - pts[:, 1]), axis=-1)

        # Target height: the path's own height, smeared along it.
        near = np.argmin(np.hypot(xx[..., None] - pts[:, 0],
                                  zz[..., None] - pts[:, 1]), axis=-1)
        path_h = np.array([self.height_at(float(p[0]), float(p[1]))
                           for p in pts])
        target = path_h[near]

        w = np.clip(1.0 - d / max(half_width, 1e-6), 0.0, 1.0)
        w = _smoothstep(w) * strength
        self.heights = self.heights * (1 - w) + target * w

    # -- feature carving --------------------------------------------------
    def _grid_coords(self):
        """World-space X and Z for every heightmap cell, as 2D arrays."""
        axis = np.linspace(0, self.size, self.res)
        zz, xx = np.meshgrid(axis, axis, indexing="ij")
        return xx, zz

    def _distance_to_polyline(self, points: np.ndarray) -> np.ndarray:
        """Shortest distance from every grid cell to a polyline.

        Chunked so a dense line on a fine grid doesn't allocate a huge
        intermediate array.
        """
        xx, zz = self._grid_coords()
        best = np.full(xx.shape, np.inf)
        for i in range(0, len(points), 24):
            chunk = points[i:i + 24]
            d = np.min(np.hypot(xx[..., None] - chunk[:, 0],
                                zz[..., None] - chunk[:, 1]), axis=-1)
            best = np.minimum(best, d)
        return best

    def carve_river(self, rng, scale: float = 1.0) -> np.ndarray:
        """Cut a channel across the terrain and return its centre-line.

        The channel is cut to a fixed level rather than a fixed depth below
        the existing ground: water finds one surface, so a river that simply
        subtracted a constant would run uphill wherever the terrain rose.
        """
        line = _meander(self.size, rng)
        dist = self._distance_to_polyline(line)

        half = RIVER_HALF_WIDTH * scale
        bank = RIVER_BANK * scale

        # Water level: low enough that the channel is clearly below the
        # surrounding land, high enough that banks aren't sheer.
        lo, hi = float(self.heights.min()), float(self.heights.max())
        level = lo + (hi - lo) * 0.28
        bed = level - RIVER_DEPTH

        # 1 inside the channel, easing to 0 at the far edge of the bank.
        w = np.clip(1.0 - (dist - half) / max(bank, 1e-6), 0.0, 1.0)
        w = np.where(dist <= half, 1.0, w)
        w = _smoothstep(w)

        self.heights = self.heights * (1 - w) + bed * w

        if self.features is None:
            self.features = TerrainFeatures(self.res)
        # Water sits where the channel was actually cut, not merely wherever
        # the ground happens to be low.
        self.features.water_mask |= (dist <= half + bank * 0.35)
        # Keep the width alongside the centre-line: the surface mesh is built
        # from these numbers rather than from the grid, so its edge is smooth.
        self.features.rivers.append((line, half + bank * 0.30))
        self.features.water_level = (level if self.features.water_level is None
                                     else min(self.features.water_level, level))
        return line

    def carve_lake(self, rng, scale: float = 1.0) -> tuple:
        """Sink a basin and return (x, z, radius)."""
        radius = self.size * 0.16 * scale
        margin = radius + self.size * 0.08
        cx = float(rng.uniform(margin, self.size - margin))
        cz = float(rng.uniform(margin, self.size - margin))

        xx, zz = self._grid_coords()
        dist = np.hypot(xx - cx, zz - cz)

        lo, hi = float(self.heights.min()), float(self.heights.max())
        level = lo + (hi - lo) * 0.30
        bed = level - LAKE_DEPTH

        # Basin profile: flat bottom, easing up to the shoreline.
        w = np.clip(1.0 - dist / max(radius, 1e-6), 0.0, 1.0)
        w = _smoothstep(w) ** 0.7

        self.heights = self.heights * (1 - w) + bed * w

        if self.features is None:
            self.features = TerrainFeatures(self.res)
        self.features.water_mask |= (dist <= radius * 0.88)
        self.features.lakes.append((cx, cz, radius))
        self.features.water_level = (level if self.features.water_level is None
                                     else min(self.features.water_level, level))
        return cx, cz, radius

    def carve_coast(self, rng, scale: float = 1.0) -> np.ndarray:
        """Drop one edge of the map into the sea. Returns the shoreline.

        The shoreline wanders, so it reads as coast rather than as the map
        being cut off with a ruler. Land just behind it is lifted a little
        above the water so the beach is dry: a flat water plane laid over
        the sea would otherwise show through any natural dip near the shore.
        """
        edge = ["north", "south", "east", "west"][int(rng.integers(0, 4))]
        depth = self.size * 0.26 * scale

        # Distance from the chosen edge for every cell, and along it.
        xx, zz = self._grid_coords()
        if edge == "north":
            across, along = zz, xx
        elif edge == "south":
            across, along = self.size - zz, xx
        elif edge == "west":
            across, along = xx, zz
        else:
            across, along = self.size - xx, zz

        # Wobble the shoreline with a few low-frequency waves.
        phases = rng.uniform(0, 2 * np.pi, 3)
        def shore_at(t):
            return depth + self.size * 0.05 * (
                np.sin(t / self.size * 2 * np.pi * 1.0 + phases[0])
                + 0.6 * np.sin(t / self.size * 2 * np.pi * 2.3 + phases[1])
                + 0.3 * np.sin(t / self.size * 2 * np.pi * 4.1 + phases[2]))

        shore = shore_at(along)
        lo, hi = float(self.heights.min()), float(self.heights.max())
        level = lo + (hi - lo) * 0.25
        bed = level - 3.0

        # Sea side: down to the bed, deepening away from the shore.
        sea = across < shore
        w = np.clip((shore - across) / 8.0, 0.0, 1.0)
        w = _smoothstep(w)
        self.heights = np.where(sea, self.heights * (1 - w) + bed * w,
                                self.heights)

        # Beach: a dry strip that rises gently from the waterline.
        beach_width = 10.0
        land = (across >= shore) & (across < shore + beach_width)
        rise = level + 0.25 + (across - shore) * 0.12
        self.heights = np.where(land, np.maximum(self.heights, rise),
                                self.heights)

        if self.features is None:
            self.features = TerrainFeatures(self.res)
        self.features.water_mask |= across < shore - 0.5
        self.features.sea_mask |= across < shore - 0.5

        samples = np.linspace(0, self.size, 48)
        if edge in ("north", "south"):
            zs = shore_at(samples)
            if edge == "south":
                zs = self.size - zs
            line = np.column_stack([samples, zs])
        else:
            xs = shore_at(samples)
            if edge == "east":
                xs = self.size - xs
            line = np.column_stack([xs, samples])

        self.features.coasts.append((edge, line, depth))
        self.features.water_level = (level if self.features.water_level is None
                                     else min(self.features.water_level, level))
        return line

    def raise_hill(self, rng, scale: float = 1.0) -> tuple:
        """Raise a rounded rise and return (x, z, radius, height)."""
        radius = self.size * 0.20 * scale
        height = max(4.0, self.preset.amplitude * 1.4) * scale
        margin = radius * 0.6
        cx = float(rng.uniform(margin, self.size - margin))
        cz = float(rng.uniform(margin, self.size - margin))

        xx, zz = self._grid_coords()
        dist = np.hypot(xx - cx, zz - cz)
        w = _smoothstep(np.clip(1.0 - dist / max(radius, 1e-6), 0.0, 1.0))

        self.heights = self.heights + w * height

        if self.features is None:
            self.features = TerrainFeatures(self.res)
        self.features.hills.append((cx, cz, radius, height))
        return cx, cz, radius, height

    def carve_valley(self, rng, scale: float = 1.0) -> tuple:
        """Lower a broad depression. Returns (x, z, radius, depth)."""
        radius = self.size * 0.30 * scale
        depth = max(3.0, self.preset.amplitude * 0.9) * scale
        cx = float(rng.uniform(self.size * 0.3, self.size * 0.7))
        cz = float(rng.uniform(self.size * 0.3, self.size * 0.7))

        xx, zz = self._grid_coords()
        dist = np.hypot(xx - cx, zz - cz)
        w = _smoothstep(np.clip(1.0 - dist / max(radius, 1e-6), 0.0, 1.0))

        self.heights = self.heights - w * depth
        return cx, cz, radius, depth

    def flatten_clearing(self, rng, scale: float = 1.0) -> tuple:
        """Level an area of open ground. Returns (x, z, radius)."""
        radius = self.size * 0.15 * scale
        margin = radius + self.size * 0.05
        cx = float(rng.uniform(margin, self.size - margin))
        cz = float(rng.uniform(margin, self.size - margin))

        self.flatten_disc(cx, cz, radius, strength=0.9)

        if self.features is None:
            self.features = TerrainFeatures(self.res)
        self.features.clearings.append((cx, cz, radius))
        return cx, cz, radius

    def apply_features(self, features, seed: int = 0) -> None:
        """Carve every feature the scene specification asked for.

        Order matters: hills and valleys reshape the land, so water is cut
        afterwards and finds the level of the reshaped surface. Cutting a
        river first and then raising a hill through it would dam the river.
        """
        if not features:
            return

        rng = np.random.default_rng(seed + 4242)
        order = {"hill": 0, "valley": 1, "cliff": 2,
                 "clearing": 3, "coast": 4, "lake": 5, "river": 6}
        for feature in sorted(features, key=lambda f: order.get(f.type, 9)):
            scale = FEATURE_SCALE.get(feature.size, 1.0)
            if feature.type == "river":
                self.carve_river(rng, scale)
            elif feature.type == "coast":
                self.carve_coast(rng, scale)
            elif feature.type == "lake":
                self.carve_lake(rng, scale)
            elif feature.type == "hill":
                self.raise_hill(rng, scale)
            elif feature.type == "valley":
                self.carve_valley(rng, scale)
            elif feature.type == "clearing":
                self.flatten_clearing(rng, scale)
            elif feature.type == "cliff":
                # A cliff is a hill with a much tighter falloff, which gives
                # a steep face rather than a dome.
                self.raise_hill(rng, scale * 0.55)

    # -- water queries ----------------------------------------------------
    @property
    def water_level(self) -> Optional[float]:
        return self.features.water_level if self.features else None

    def on_bridge(self, x: float, z: float, margin: float = 0.0) -> bool:
        """True if (x, z) is on a bridge deck."""
        if self.features is None:
            return False
        return any(b.covers(x, z, margin) for b in self.features.bridges)

    def is_sea(self, x: float, z: float, margin: float = 0.0) -> bool:
        """True if (x, z) is open sea (not a river or lake), within margin."""
        if self.features is None or not self.features.sea_mask.any():
            return False
        mask = self.features.sea_mask
        cell = self.size / (self.res - 1)
        reach = max(int(np.ceil(margin / cell)), 0)
        gx = int(np.clip(x / self.size * (self.res - 1), 0, self.res - 1))
        gz = int(np.clip(z / self.size * (self.res - 1), 0, self.res - 1))
        return bool(mask[max(gz - reach, 0):gz + reach + 1,
                         max(gx - reach, 0):gx + reach + 1].any())

    def is_water(self, x: float, z: float, margin: float = 0.0) -> bool:
        """True if this position is water, or within `margin` metres of it.

        Reads the carved mask rather than comparing height to the water level.
        The level is a percentile of terrain height, so a height comparison
        marks a large share of perfectly dry ground as submerged.

        `margin` keeps objects back from the waterline, where a model would
        otherwise stand half-sunk.
        """
        if self.features is None or not self.features.has_water:
            return False

        mask = self.features.water_mask
        cell = self.size / (self.res - 1)
        reach = max(int(np.ceil(margin / cell)), 0)

        gx = int(np.clip(x / self.size * (self.res - 1), 0, self.res - 1))
        gz = int(np.clip(z / self.size * (self.res - 1), 0, self.res - 1))

        if reach == 0:
            return bool(mask[gz, gx])

        x0, x1 = max(0, gx - reach), min(self.res, gx + reach + 1)
        z0, z1 = max(0, gz - reach), min(self.res, gz + reach + 1)
        return bool(mask[z0:z1, x0:x1].any())

    def colour_at(self, x: float, z: float) -> tuple:
        """Ground colour at a position, blended by elevation."""
        h = self.height_at(x, z)
        lo, hi = float(self.heights.min()), float(self.heights.max())
        t = 0.0 if hi - lo < 1e-9 else (h - lo) / (hi - lo)
        c0, c1 = self.preset.colour_low, self.preset.colour_high
        return tuple(c0[i] + (c1[i] - c0[i]) * t for i in range(3))

    def stats(self) -> dict:
        return {
            "type": self.terrain_type,
            "size_m": self.size,
            "resolution": self.res,
            "min_height": round(float(self.heights.min()), 2),
            "max_height": round(float(self.heights.max()), 2),
            "mean_height": round(float(self.heights.mean()), 2),
            "seed": self.seed,
        }

    def __repr__(self) -> str:
        return (f"<Terrain {self.terrain_type} {self.size:.0f}m "
                f"h={self.heights.min():.1f}..{self.heights.max():.1f}>")


# ---------------------------------------------------------------------------
# Terrain features
#
# Rivers, lakes, hills and valleys are carved into the heightmap before any
# object is placed, because everything downstream has to respect them: you
# cannot build in a river, vegetation crowds its banks, and a road has to
# cross it rather than run through it.
#
# All of it is arithmetic on the height array -- no model, no dataset.
# ---------------------------------------------------------------------------

# Feature dimensions as a fraction of scene size, by declared size.
FEATURE_SCALE = {"small": 0.55, "medium": 1.0, "large": 1.55}

# How deep water cuts below the surrounding ground, in metres.
RIVER_DEPTH = 2.2
LAKE_DEPTH = 3.0

# Half-width of a river channel in metres, before size scaling.
RIVER_HALF_WIDTH = 5.0
RIVER_BANK = 7.0            # how far the banks slope up beyond the channel


def _meander(size: float, rng: np.random.Generator,
             points: int = 7) -> np.ndarray:
    """A wandering line across the scene, for a river to follow.

    Deliberately not the same generator as the road: a river should not run
    parallel to it or the two read as one feature.
    """
    vertical = rng.random() < 0.5
    if vertical:
        start = (rng.uniform(0.15, 0.85) * size, 0.0)
        end = (rng.uniform(0.15, 0.85) * size, size)
    else:
        start = (0.0, rng.uniform(0.15, 0.85) * size)
        end = (size, rng.uniform(0.15, 0.85) * size)

    xs = np.linspace(start[0], end[0], points)
    zs = np.linspace(start[1], end[1], points)

    drift = size * 0.20
    if vertical:
        xs = xs + rng.normal(0, drift, points)
        xs[0], xs[-1] = start[0], end[0]
    else:
        zs = zs + rng.normal(0, drift, points)
        zs[0], zs[-1] = start[1], end[1]

    xs = np.clip(xs, size * 0.05, size * 0.95)
    zs = np.clip(zs, size * 0.05, size * 0.95)

    t = np.linspace(0, 1, points)
    fine = np.linspace(0, 1, points * 14)
    return np.column_stack([np.interp(fine, t, xs), np.interp(fine, t, zs)])


class TerrainFeatures:
    """Everything carved into a terrain, and where the water sits."""

    def __init__(self, resolution: int = 128):
        self.water_level: Optional[float] = None
        # Where water actually is, rather than inferred from height.
        #
        # Deriving it as "anything below the water level" marked 42% of an
        # untouched grassland as submerged, because the level is a percentile
        # of terrain height and a lot of natural ground sits below it. Only
        # the cells we carve hold water.
        self.water_mask: np.ndarray = np.zeros((resolution, resolution),
                                               dtype=bool)
        # The open sea only. Roads stop at it; rivers and lakes are crossed.
        self.sea_mask: np.ndarray = np.zeros((resolution, resolution),
                                             dtype=bool)
        self.rivers: List[np.ndarray] = []
        self.lakes: List[tuple] = []          # (x, z, radius)
        self.hills: List[tuple] = []          # (x, z, radius, height)
        self.clearings: List[tuple] = []      # (x, z, radius)
        # Each coast is (edge, shoreline points, depth from that edge).
        self.coasts: List[tuple] = []
        # Crossings where the road meets water; see crossings.py.
        self.bridges: list = []

    @property
    def has_water(self) -> bool:
        return self.water_level is not None and bool(self.water_mask.any())

    @property
    def water_fraction(self) -> float:
        """Share of the terrain under water, for sanity checks."""
        return float(self.water_mask.mean())

    def to_dict(self) -> dict:
        return {
            "water_level": (round(self.water_level, 3)
                            if self.water_level is not None else None),
            "water_fraction": round(self.water_fraction, 4),
            "rivers": [{"centre_line": [[round(float(x), 2), round(float(z), 2)]
                                       for x, z in line],
                        "half_width": round(float(w), 2)}
                       for line, w in self.rivers],
            "lakes": [[round(v, 2) for v in lake] for lake in self.lakes],
            "hills": [[round(v, 2) for v in h] for h in self.hills],
            "clearings": [[round(v, 2) for v in c] for c in self.clearings],
            "bridges": [b.to_dict() for b in self.bridges],
            "coasts": [{"edge": edge,
                        "shoreline": [[round(float(x), 2), round(float(z), 2)]
                                      for x, z in line]}
                       for edge, line, _ in self.coasts],
        }
