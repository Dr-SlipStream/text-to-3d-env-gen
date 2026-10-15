"""
Running a scene graph through the rest of the pipeline.

The graph says "six tents ringed around the fire". Retrieval turns "tent" into
a real model with a real footprint; the solver turns the relations into
positions; this module joins the two and hands back the same PlacedScene the
exporter already understands.

Keeping the output type identical to the rule-based path means everything
downstream -- export, viewer, Unity, the validation checks -- works unchanged,
and the two paths can be compared on the same scene.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from . import appearance
from .asset_index import AssetIndex
from .asset_resolution import LOW_CONFIDENCE_BY_EMBEDDER
from .layout import PlacedInstance, PlacedScene
from .scene_graph import SceneGraph
from .solver import solve


class ResolvedGroup:
    """A graph group bound to a concrete 3D model."""

    def __init__(self, group, match, category: str):
        self.group = group
        self.match = match
        self.category = category

    @property
    def norm_scale(self) -> float:
        if not self.match:
            return 1.0
        return appearance.normalisation_scale(
            self.match.name, self.category,
            self.match.radius, self.match.height)

    @property
    def radius(self) -> float:
        return (self.match.radius * self.norm_scale) if self.match else 1.0

    @property
    def cap(self) -> float:
        if not self.match:
            return 1e9
        return appearance.cap_scale(self.category, self.match.radius,
                                    self.match.height)


def resolve_groups(graph: SceneGraph, index: AssetIndex,
                   theme: Optional[str] = None) -> Tuple[Dict, List[str]]:
    """Bind every group in the graph to a real model.

    The graph names what it wants in plain words -- "campfire", "pine tree",
    "crate" -- and retrieval finds the closest model the library actually
    holds, exactly as it does for the flat path.
    """
    resolved: Dict[str, ResolvedGroup] = {}
    warnings: List[str] = []

    threshold = LOW_CONFIDENCE_BY_EMBEDDER.get(
        getattr(index.embedder, "name", ""), 0.35)

    for group in graph.groups:
        # The same keyword table that files the assets decides what kind of
        # thing this is, so the search looks on the right shelf.
        from . import asset_rules
        category = (asset_rules.classify_confident(group.asset)
                    or asset_rules.classify(group.asset))

        match = index.best(f"{group.asset} {category}",
                           category=category, theme=theme)
        if match is None:
            warnings.append(f"{group.id}: no model found for "
                            f"{group.asset!r}")
            continue

        if match.score < threshold:
            warnings.append(
                f"weak match for {group.asset!r}: '{match.name}' "
                f"({match.score:.2f})")

        resolved[group.id] = ResolvedGroup(group, match, category)

    return resolved, warnings


def build_scene(graph: SceneGraph, terrain, index: AssetIndex,
                spec=None, seed: int = 0,
                path: Optional[List[Tuple[float, float]]] = None
                ) -> PlacedScene:
    """Solve the graph and return a PlacedScene ready for export."""
    theme = getattr(spec, "theme", None)
    resolved, warnings = resolve_groups(graph, index, theme=theme)

    radii = {gid: rg.radius for gid, rg in resolved.items()}
    categories = {gid: rg.category for gid, rg in resolved.items()}

    solved = solve(graph, terrain, radii, categories, seed=seed, path=path)

    scene = PlacedScene(terrain=terrain, spec=spec, seed=seed)
    scene.path = solved.path
    scene.warnings = list(graph.warnings) + warnings + solved.warnings
    scene.skipped = solved.skipped

    # A settlement region, if there is one, doubles as the scene's centre --
    # the viewer and the ground painting both use it.
    for region in solved.regions.values():
        if region.region.kind == "settlement":
            scene.centre = region.centre
            scene.core_radius = region.radius
            break

    for item in solved.instances:
        rg = resolved.get(item.group_id)
        if rg is None or rg.match is None:
            continue

        scale = min(rg.norm_scale * item.scale_jitter, rg.cap)
        scene.instances.append(PlacedInstance(
            name=rg.group.asset,
            category=rg.category,
            asset_name=rg.match.name,
            asset_file=rg.match.file,
            position=item.position,
            rotation_y=item.rotation_y,
            scale=scale,
            radius=item.radius,
            triangles=int(rg.match.asset.get("triangles") or 0),
        ))

    return scene


def describe(graph: SceneGraph, resolved: Optional[Dict] = None) -> str:
    """A readable account of the graph, for the command line.

    Worth printing: it is the clearest evidence that the system understood
    the prompt as a structured place rather than a bag of objects.
    """
    lines = []
    lines.append(f"{'region':<16}{'kind':<14}{'size'}")
    lines.append("-" * 44)
    for region in graph.regions:
        lines.append(f"{region.id:<16}{region.kind:<14}{region.size}")

    lines.append("")
    lines.append(f"{'group':<22}{'asset':<18}{'amount':<12}relations")
    lines.append("-" * 96)
    for group in graph.groups:
        amount = (f"x{group.count}" if group.count is not None
                  else f"{group.density:.2f}/100m2")
        rels = "  ".join(repr(r) for r in group.relations) or "-"
        model = ""
        if resolved and group.id in resolved:
            match = resolved[group.id].match
            if match:
                model = f"  -> {match.name}"
        # Long ids used to run straight into the next column.
        lines.append(f"{group.id[:21]:<22}{group.asset[:17]:<18}"
                     f"{amount:<12}{rels}{model}")
    return "\n".join(lines)
