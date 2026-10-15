"""
The scene graph: a relational description of a place.

WHY THIS EXISTS
---------------
The original scene specification is a flat list of objects with counts:

    {"objects": [{"name": "campfire", "quantity": 134}]}

It carries no structure, so every piece of structure has to live somewhere
else -- and that somewhere else is a hand-written table. What a forest camp
contains, where buildings go, that vegetation avoids a village square, how far
a house stands from a river: all of it encoded per theme, per category, in
code. Adding a capability means adding a table.

It also cannot express what a prompt plainly says. "A campfire" is one fire, at
the heart of the camp, with tents facing it. A flat list can only say "134
light sources", which is what the system did.

A scene graph says instead:

    camp    is a settlement region
    woods   is wilderness, filling the remainder
    fire    1 campfire,  at_center_of(camp)
    tents   6 tents,     ring_around(fire, 8m) and faces(fire)
    trees   density,     scatter_in(woods), avoid(camp)

One solver satisfies those relations. It knows nothing about camps, villages
or forests -- only about centres, rings, spacing and avoidance. New kinds of
place need no new code, because the language model composes them out of the
same small vocabulary.

DESIGN CONSTRAINTS
------------------
The relation vocabulary is deliberately small and closed. A language model
emitting a large open-ended grammar produces plausible nonsense; a model
choosing from ten named relations produces something checkable. Everything
here validates, repairs, or rejects.
"""

from __future__ import annotations

import difflib
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Regions -- named areas of the scene.
#
# A region is *where* things go. Some come from the terrain (a river is a
# water region), some are abstract (the built-up part of a settlement).
# ---------------------------------------------------------------------------
REGION_KINDS = [
    "settlement",    # the built-up core: buildings, streets, yards
    "wilderness",    # whatever surrounds it: forest, scrub, dunes
    "water",         # a river or lake, from the terrain features
    "field",         # cultivated or cleared open ground
    "approach",      # the road corridor and its verges
    "high_ground",   # a hill or ridge
]

# How much of the scene a region occupies, as a fraction of its width.
REGION_EXTENTS = {"small": 0.18, "medium": 0.30, "large": 0.45}


# ---------------------------------------------------------------------------
# Relations -- how a group positions itself with respect to something else.
#
# Ten of them. Each has a sampler in the solver; adding an eleventh means
# adding one function, not a new table per theme.
# ---------------------------------------------------------------------------
RELATION_KINDS = [
    # -- placement ------------------------------------------------------
    "inside",          # somewhere within a region (the default)
    "at_center_of",    # the focal point of a region
    "ring_around",     # evenly spaced at a radius from a target
    "cluster_near",    # loosely gathered close to a target
    "along",           # following a line: a road, a riverbank, a wall
    "on_shore_of",     # on the land side of a water region
    "scatter_in",      # spread through a region, clumped like vegetation
    "avoid",           # a constraint rather than a position: keep away
    # -- orientation ----------------------------------------------------
    "faces",           # turn to look at a target
    "aligns_with",     # turn parallel to a line
]

# Relations that decide *where* something goes. A group needs exactly one;
# anything else it declares is a modifier.
POSITIONAL = {"inside", "at_center_of", "ring_around", "cluster_near",
              "along", "on_shore_of", "scatter_in"}

ORIENTING = {"faces", "aligns_with"}
CONSTRAINING = {"avoid"}


def _snap(value: str, allowed: List[str], default: str) -> str:
    """Coerce a loose string onto a closed vocabulary."""
    if value is None:
        return default
    v = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    if v in allowed:
        return v
    flat = {a.replace("_", ""): a for a in allowed}
    if v.replace("_", "") in flat:
        return flat[v.replace("_", "")]
    close = difflib.get_close_matches(v, allowed, n=1, cutoff=0.7)
    return close[0] if close else default


class Relation(BaseModel):
    """One spatial relationship, from a group to something named."""

    kind: str
    target: str = ""
    distance: float = 0.0        # metres; meaning depends on the relation

    @field_validator("kind")
    @classmethod
    def _v_kind(cls, v):
        return _snap(v, RELATION_KINDS, "inside")

    @field_validator("distance")
    @classmethod
    def _v_distance(cls, v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 0.0
        # A relation reaching further than a large scene is a model error.
        return max(0.0, min(v, 250.0))

    @property
    def is_positional(self) -> bool:
        return self.kind in POSITIONAL

    @property
    def is_orienting(self) -> bool:
        return self.kind in ORIENTING

    @property
    def needs_target(self) -> bool:
        return self.kind != "inside" or bool(self.target)

    def __repr__(self) -> str:
        d = f", {self.distance:g}m" if self.distance else ""
        return f"{self.kind}({self.target}{d})"


class Region(BaseModel):
    """A named area of the scene."""

    id: str
    kind: str = "wilderness"
    size: str = "medium"
    # Regions derived from terrain features point back at them, so the solver
    # can use the real carved geometry rather than an abstract disc.
    feature: Optional[str] = None

    @field_validator("id")
    @classmethod
    def _v_id(cls, v):
        v = str(v).strip().lower().replace(" ", "_")
        if not v:
            raise ValueError("region id cannot be empty")
        return v

    @field_validator("kind")
    @classmethod
    def _v_kind(cls, v):
        return _snap(v, REGION_KINDS, "wilderness")

    @field_validator("size")
    @classmethod
    def _v_size(cls, v):
        return _snap(v, ["small", "medium", "large"], "medium")

    @property
    def extent(self) -> float:
        return REGION_EXTENTS.get(self.size, 0.30)


class Group(BaseModel):
    """A set of identical objects placed together under shared relations.

    `count` is an exact number; `density` is per 100 square metres of the
    region it sits in. Exactly one should be given -- a campfire is one
    campfire, whereas trees are however many fit.
    """

    id: str
    asset: str                       # what to search the library for
    count: Optional[int] = None
    density: Optional[float] = None
    relations: List[Relation] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _v_id(cls, v):
        v = str(v).strip().lower().replace(" ", "_")
        if not v:
            raise ValueError("group id cannot be empty")
        return v

    @field_validator("asset")
    @classmethod
    def _v_asset(cls, v):
        v = str(v).strip().lower()
        if not v:
            raise ValueError("group asset cannot be empty")
        return v

    @field_validator("count")
    @classmethod
    def _v_count(cls, v):
        if v is None:
            return None
        try:
            return max(1, min(int(v), 400))
        except (TypeError, ValueError):
            return None

    @field_validator("density")
    @classmethod
    def _v_density(cls, v):
        if v is None:
            return None
        try:
            return max(0.0, min(float(v), 12.0))
        except (TypeError, ValueError):
            return None

    @model_validator(mode="after")
    def _post(self):
        # Exactly one of count or density. A group with neither places
        # nothing; a group with both is ambiguous.
        if self.count is None and self.density is None:
            self.count = 1
        elif self.count is not None and self.density is not None:
            self.density = None

        # Deduplicate relations, and keep only the first positional one --
        # an object can only be in one place.
        seen_positional = False
        kept: List[Relation] = []
        for rel in self.relations:
            if rel.is_positional:
                if seen_positional:
                    continue
                seen_positional = True
            kept.append(rel)
        self.relations = kept[:6]
        return self

    @property
    def placement(self) -> Optional[Relation]:
        for rel in self.relations:
            if rel.is_positional:
                return rel
        return None

    @property
    def orientation(self) -> Optional[Relation]:
        for rel in self.relations:
            if rel.is_orienting:
                return rel
        return None

    @property
    def avoidances(self) -> List[Relation]:
        return [r for r in self.relations if r.kind == "avoid"]

    @property
    def is_anchor(self) -> bool:
        """A group nothing else is positioned relative to can wait its turn.

        Anchors -- a campfire, a well, a bridge -- have to be placed first,
        because other groups are defined in terms of where they ended up.
        """
        return self.count is not None and self.count <= 3

    def __repr__(self) -> str:
        amount = (f"x{self.count}" if self.count is not None
                  else f"d={self.density}")
        rels = " ".join(repr(r) for r in self.relations)
        return f"<{self.id}: {self.asset} {amount} {rels}>"


class SceneGraph(BaseModel):
    """A relational description of a scene: regions, groups, relations."""

    regions: List[Region] = Field(default_factory=list)
    groups: List[Group] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _post(self):
        # Unique ids. A duplicate makes every relation naming it ambiguous.
        seen: Dict[str, int] = {}
        unique_regions = []
        for region in self.regions:
            if region.id in seen:
                self.warnings.append(f"duplicate region {region.id!r} dropped")
                continue
            seen[region.id] = 1
            unique_regions.append(region)
        self.regions = unique_regions[:8]

        group_ids: Dict[str, int] = {}
        unique_groups = []
        for group in self.groups:
            if group.id in group_ids or group.id in seen:
                self.warnings.append(f"duplicate group {group.id!r} dropped")
                continue
            group_ids[group.id] = 1
            unique_groups.append(group)
        self.groups = unique_groups[:40]

        # Every relation must name something that exists. A dangling target
        # is the commonest way a generated graph goes wrong, and silently
        # placing the group anywhere would hide it.
        known = set(seen) | set(group_ids)
        for group in self.groups:
            valid = []
            for rel in group.relations:
                if rel.kind == "inside" and not rel.target:
                    valid.append(rel)
                elif rel.target in known:
                    valid.append(rel)
                else:
                    self.warnings.append(
                        f"{group.id}: {rel.kind}({rel.target!r}) names "
                        "nothing in the graph; relation dropped"
                    )
            group.relations = valid

        return self

    # -- lookups ----------------------------------------------------------
    def region(self, region_id: str) -> Optional[Region]:
        for r in self.regions:
            if r.id == region_id:
                return r
        return None

    def group(self, group_id: str) -> Optional[Group]:
        for g in self.groups:
            if g.id == group_id:
                return g
        return None

    def dependencies(self, group: Group) -> List[str]:
        """Group ids this one is positioned relative to."""
        group_ids = {g.id for g in self.groups}
        return [r.target for r in group.relations
                if r.target in group_ids and r.target != group.id]

    def resolution_order(self) -> List[Group]:
        """Groups in an order where every dependency is placed first.

        Tents are positioned around a campfire, so the fire must already have
        a position when the tents are placed. A cycle -- two groups each
        defined relative to the other -- is broken by dropping the later
        dependency, since some order is better than none.
        """
        order: List[Group] = []
        placed: set = set()
        remaining = list(self.groups)

        # Anchors first among equals: a handful of focal objects, then the
        # things arranged around them, then bulk scatter.
        remaining.sort(key=lambda g: (not g.is_anchor,
                                      g.density is not None,
                                      g.id))

        progress = True
        while remaining and progress:
            progress = False
            for group in list(remaining):
                if all(d in placed for d in self.dependencies(group)):
                    order.append(group)
                    placed.add(group.id)
                    remaining.remove(group)
                    progress = True

        if remaining:
            # A cycle. Place them anyway, in declaration order, with their
            # unresolved relations degraded to plain containment.
            for group in remaining:
                self.warnings.append(
                    f"{group.id}: circular dependency; treated as unplaced"
                )
                order.append(group)

        return order

    def summary(self) -> str:
        n = sum(g.count for g in self.groups if g.count) or 0
        return (f"{len(self.regions)} regions, {len(self.groups)} groups "
                f"({n} fixed-count instances)")
