"""
Resolve a SceneSpec's abstract objects into concrete, placeable assets.

Input:  SceneSpec  ("7 houses, 18 trees, 1 blacksmith forge")
Output: ResolvedScene -- the same scene with every object bound to a real
        model file, with real dimensions, ready for the layout stage.

Handles the awkward cases:
  - the requested category is empty in the library (substitute a sensible one)
  - nothing matches well (keep the best effort, flag it as low confidence)
  - the same asset gets picked for several different objects (allowed, but
    we record it so the scene doesn't end up visually monotonous)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .asset_index import AssetIndex, AssetMatch
from .schema import SceneObject, SceneSpec

# If a category has no assets, fall back to these in order. Better to place a
# visually wrong object than to silently drop part of the user's prompt.
CATEGORY_FALLBACKS = {
    "building": ["structure", "prop"],
    "structure": ["building", "prop"],
    "vegetation": ["prop", "rock"],
    "rock": ["prop", "vegetation"],
    "prop": ["rock", "structure"],
    "light_source": ["prop", "structure"],
}

# Below this similarity we treat the match as a guess rather than a hit.
# The two embedders produce different score distributions, so each needs its
# own threshold: character matching rarely exceeds 0.6 even on good hits,
# while semantic embeddings score 0.85+ on a real match and hover around
# 0.5-0.65 when they're reaching for something that isn't in the library.
LOW_CONFIDENCE_BY_EMBEDDER = {
    "hashing": 0.35,
    "sentence-transformers": 0.70,
}
LOW_CONFIDENCE = 0.35          # default when the embedder is unknown

# How much worse a match may be before we stop swapping it in purely to avoid
# reusing an asset. Keeps variety without accepting a badly wrong model.
DIVERSITY_TOLERANCE = 0.08


@dataclass
class ResolvedObject:
    """A scene object bound to a specific 3D model."""

    spec: SceneObject
    match: Optional[AssetMatch]
    confidence: float = 0.0
    substituted_category: Optional[str] = None

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def quantity(self) -> int:
        return self.spec.quantity

    @property
    def placement(self) -> str:
        return self.spec.placement or "scatter"

    @property
    def radius(self) -> float:
        """Footprint radius, from the real mesh when we have it."""
        return self.match.radius if self.match else self.spec.radius

    @property
    def height(self) -> float:
        return self.match.height if self.match else self.spec.radius * 2

    @property
    def file(self) -> Optional[str]:
        return self.match.file if self.match else None

    @property
    def resolved(self) -> bool:
        return self.match is not None


@dataclass
class ResolvedScene:
    """A SceneSpec with every object bound to a real asset."""

    spec: SceneSpec
    objects: List[ResolvedObject] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    low_confidence_threshold: float = LOW_CONFIDENCE

    @property
    def total_instances(self) -> int:
        return sum(o.quantity for o in self.objects if o.resolved)

    def summary(self) -> str:
        ok = sum(1 for o in self.objects if o.resolved)
        low = sum(1 for o in self.objects
                  if o.resolved and o.confidence < self.low_confidence_threshold)
        return (
            f"{ok}/{len(self.objects)} object types resolved, "
            f"{self.total_instances} instances"
            + (f", {low} low-confidence" if low else "")
        )

    def report(self) -> str:
        """Human-readable table, useful for demos and debugging."""
        lines = [
            f"{'requested':<20} {'->':<3} {'asset':<28} "
            f"{'cat':<12} {'qty':>4} {'score':>6}",
            "-" * 78,
        ]
        for o in self.objects:
            if o.resolved:
                flag = " *" if o.confidence < self.low_confidence_threshold else ""
                sub = f" [{o.substituted_category}]" if o.substituted_category else ""
                lines.append(
                    f"{o.name:<20} {'->':<3} {o.match.name[:28]:<28} "
                    f"{o.match.category:<12} {o.quantity:>4} "
                    f"{o.confidence:>6.3f}{flag}{sub}"
                )
            else:
                lines.append(
                    f"{o.name:<20} {'->':<3} {'NO MATCH':<28} "
                    f"{'-':<12} {o.quantity:>4} {'-':>6}"
                )
        if any(o.resolved and o.confidence < self.low_confidence_threshold
               for o in self.objects):
            lines.append(
                f"\n* below {self.low_confidence_threshold:.2f} confidence -- "
                "your library probably lacks a good model for these"
            )
        return "\n".join(lines)


def resolve_scene(spec: SceneSpec, index: AssetIndex) -> ResolvedScene:
    """Bind every object in `spec` to a concrete asset from `index`."""
    scene = ResolvedScene(spec=spec)
    available = index.categories()

    # Threshold depends on which embedder built the index.
    threshold = LOW_CONFIDENCE_BY_EMBEDDER.get(
        getattr(index.embedder, "name", ""), LOW_CONFIDENCE
    )
    scene.low_confidence_threshold = threshold

    used_asset_ids = set()

    for obj in spec.objects:
        category = obj.category
        substituted = None

        # If the library has nothing in this category, substitute.
        if category not in available:
            for alt in CATEGORY_FALLBACKS.get(category, []):
                if alt in available:
                    substituted = category
                    category = alt
                    scene.warnings.append(
                        f"no '{substituted}' assets; using '{alt}' for {obj.name!r}"
                    )
                    break

        # Query with the object name plus its category, which helps the
        # embedder disambiguate ("bridge" the structure, not a card game).
        query = f"{obj.name} {obj.category}"
        candidates = index.search(query, category=category, theme=spec.theme,
                                  top_k=8)

        if not candidates:
            scene.objects.append(ResolvedObject(spec=obj, match=None))
            scene.warnings.append(f"no asset found for {obj.name!r}")
            continue

        # Prefer an asset we haven't used yet, so "house" and "cottage" don't
        # both become the same model -- but only if it's nearly as good.
        match = candidates[0]
        if match.asset["id"] in used_asset_ids:
            for alt in candidates[1:]:
                if alt.asset["id"] in used_asset_ids:
                    continue
                if match.score - alt.score <= DIVERSITY_TOLERANCE:
                    match = alt
                break

        used_asset_ids.add(match.asset["id"])

        if match.score < threshold:
            scene.warnings.append(
                f"weak match for {obj.name!r}: '{match.name}' "
                f"({match.score:.2f}) -- library may lack this object"
            )

        scene.objects.append(
            ResolvedObject(
                spec=obj,
                match=match,
                confidence=match.score,
                substituted_category=substituted,
            )
        )

    # Flag monotony: many different objects resolving to the same model makes
    # a visibly repetitive scene, which matters for the demo.
    used = [o.match.asset["id"] for o in scene.objects if o.resolved]
    if used and len(set(used)) < len(used) * 0.6:
        scene.warnings.append(
            "many objects resolved to the same asset; scene may look repetitive"
        )

    return scene
