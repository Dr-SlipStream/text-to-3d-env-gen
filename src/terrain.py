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
from typing import Tuple

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
