"""
The Scene Specification -- the single data contract for the whole pipeline.

Every stage reads a SceneSpec and/or writes one. Prompt parsing produces it;
retrieval, terrain, layout and export all consume it. If a stage needs new
information, add a field HERE rather than passing side-channel arguments.

Validation is deliberately forgiving: an LLM will occasionally return a value
slightly outside our vocabulary ("sunset" instead of "dusk"), so we snap
near-misses onto the nearest legal value instead of crashing the pipeline.
"""

from __future__ import annotations

import difflib
import json
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from . import vocab


def _snap(value: str, allowed: List[str], default: str) -> str:
    """Coerce a loose string onto our controlled vocabulary.

    Tries, in order: exact match, case/underscore-insensitive match, then
    closest fuzzy match. Falls back to `default` if nothing is close enough.
    """
    if value is None:
        return default

    v = str(value).strip().lower().replace(" ", "_").replace("-", "_")

    if v in allowed:
        return v

    # Try matching ignoring underscores entirely ("lowpoly" -> "low_poly")
    flat = {a.replace("_", ""): a for a in allowed}
    if v.replace("_", "") in flat:
        return flat[v.replace("_", "")]

    # Fuzzy match for typos and near-synonyms
    close = difflib.get_close_matches(v, allowed, n=1, cutoff=0.75)
    if close:
        return close[0]

    return default


class Terrain(BaseModel):
    type: str = "grassland"
    size: str = "medium"

    @field_validator("type")
    @classmethod
    def _v_type(cls, v):
        return _snap(v, vocab.TERRAIN_TYPES, "grassland")

    @field_validator("size")
    @classmethod
    def _v_size(cls, v):
        return _snap(v, vocab.TERRAIN_SIZES, "medium")

    @property
    def size_metres(self) -> int:
        return vocab.TERRAIN_SIZE_METRES[self.size]


class Lighting(BaseModel):
    time_of_day: str = "day"
    weather: str = "clear"
    mood: str = "peaceful"

    @field_validator("time_of_day")
    @classmethod
    def _v_tod(cls, v):
        return _snap(v, vocab.TIMES_OF_DAY, "day")

    @field_validator("weather")
    @classmethod
    def _v_weather(cls, v):
        return _snap(v, vocab.WEATHER, "clear")

    @field_validator("mood")
    @classmethod
    def _v_mood(cls, v):
        return _snap(v, vocab.MOODS, "peaceful")


class TerrainFeature(BaseModel):
    """A landscape element carved into the ground, not placed on it.

    Rivers, lakes, hills and valleys are shaped into the heightmap before any
    object is positioned, because everything else has to respect them -- you
    cannot build in a river, and vegetation crowds its banks.
    """

    type: str
    size: str = "medium"

    @field_validator("type")
    @classmethod
    def _v_type(cls, v):
        # Resolve synonyms before snapping. "creek" and "stream" are rivers,
        # "oasis" is a lake -- without this they fuzzy-match to whatever is
        # closest by spelling, which turned a creek into a hill.
        word = str(v).lower().strip()
        for feature, words in vocab.FEATURE_KEYWORDS.items():
            if word == feature or word in words:
                return feature
        return _snap(v, vocab.TERRAIN_FEATURES, "hill")

    @field_validator("size")
    @classmethod
    def _v_size(cls, v):
        return _snap(v, vocab.FEATURE_SIZES, "medium")

    @property
    def holds_water(self) -> bool:
        return self.type in vocab.WATER_FEATURES


class SceneObject(BaseModel):
    """One kind of object in the scene, plus how many of it to place."""

    name: str                       # free text, e.g. "blacksmith forge"
    category: str = "prop"          # controlled, drives asset folder + placement
    placement: Optional[str] = None # controlled; defaults per category
    quantity: int = 1

    @field_validator("name")
    @classmethod
    def _v_name(cls, v):
        v = str(v).strip().lower()
        if not v:
            raise ValueError("object name cannot be empty")
        return v

    @field_validator("category")
    @classmethod
    def _v_category(cls, v):
        return _snap(v, vocab.OBJECT_CATEGORIES, "prop")

    @field_validator("quantity")
    @classmethod
    def _v_quantity(cls, v):
        try:
            v = int(v)
        except (TypeError, ValueError):
            v = 1
        return max(1, min(v, vocab.MAX_QUANTITY))

    @model_validator(mode="after")
    def _fill_placement(self):
        # Categorise the request with the same keyword table that categorises
        # the assets. Retrieval searches within a category, so if the two
        # disagree the search looks on the wrong shelf entirely: the model
        # called a firepit a "structure" and it matched a metal wall, and
        # called a log a "prop" and it matched a bed -- while the library held
        # perfectly good campfires and logs the search never saw.
        #
        # Only overrides when a keyword actually matched. An unrecognised name
        # keeps whatever the model chose, since we have nothing better.
        from . import asset_rules

        confident = asset_rules.classify_confident(self.name)
        corrected = confident or self.category

        # The explicit table still wins where the keyword rules are wrong --
        # a forge matches "forge" under buildings, but "well" would otherwise
        # land oddly.
        corrected = vocab.override_category(self.name, corrected)
        if corrected != self.category:
            self.category = corrected
            self.placement = None      # re-derive from the corrected category

        if self.placement is None:
            self.placement = vocab.CATEGORY_DEFAULT_PLACEMENT.get(
                self.category, "scatter"
            )
        else:
            self.placement = _snap(
                self.placement,
                vocab.PLACEMENT_RULES,
                vocab.CATEGORY_DEFAULT_PLACEMENT.get(self.category, "scatter"),
            )
        return self

    @property
    def radius(self) -> float:
        return vocab.CATEGORY_DEFAULT_RADIUS.get(self.category, 1.0)


class SceneSpec(BaseModel):
    """The structured form of a user's prompt. The backbone of the pipeline."""

    theme: str = "medieval_village"
    art_style: str = "low_poly"
    terrain: Terrain = Field(default_factory=Terrain)
    lighting: Lighting = Field(default_factory=Lighting)
    features: List[TerrainFeature] = Field(default_factory=list)
    objects: List[SceneObject] = Field(default_factory=list)

    # Provenance / debugging -- never used for generation, but invaluable when
    # something looks wrong and you need to know which stage produced it.
    source_prompt: str = ""
    parser: str = "unknown"          # "llm" or "fallback"
    # Whether the prompt actually named a theme we support, or we picked a
    # default. An unrecognised prompt still produces a valid scene, but the
    # user should be told it isn't what they asked for -- silently returning a
    # medieval village for "an underwater city" is worse than saying so.
    theme_recognised: bool = True
    warnings: List[str] = Field(default_factory=list)

    @field_validator("theme")
    @classmethod
    def _v_theme(cls, v):
        return _snap(v, vocab.THEMES, "medieval_village")

    @field_validator("art_style")
    @classmethod
    def _v_style(cls, v):
        return _snap(v, vocab.ART_STYLES, "low_poly")

    @model_validator(mode="after")
    def _post(self):
        # Two of the same feature is almost always the model repeating itself,
        # and two rivers in a 120m scene would leave nowhere to build.
        seen_features = set()
        unique = []
        for f in self.features:
            if f.type not in seen_features:
                seen_features.add(f.type)
                unique.append(f)
        self.features = unique[:3]

        # Drop atmosphere and effects the LLM listed as objects. "smoke" has
        # no mesh; trying to place one always picks something wrong.
        physical = []
        for obj in self.objects:
            if vocab.is_non_physical(obj.name):
                self.warnings.append(
                    f"dropped {obj.name!r}: atmosphere/effect, not a placeable object"
                )
            else:
                physical.append(obj)
        self.objects = physical

        # Cap object variety so a runaway LLM can't request 200 object types.
        if len(self.objects) > vocab.MAX_OBJECT_TYPES:
            self.warnings.append(
                f"trimmed objects from {len(self.objects)} to "
                f"{vocab.MAX_OBJECT_TYPES}"
            )
            self.objects = self.objects[: vocab.MAX_OBJECT_TYPES]

        # Merge duplicate object names (LLMs sometimes repeat themselves).
        merged: dict = {}
        for obj in self.objects:
            key = (obj.name, obj.category)
            if key in merged:
                merged[key].quantity = min(
                    merged[key].quantity + obj.quantity, vocab.MAX_QUANTITY
                )
            else:
                merged[key] = obj
        self.objects = list(merged.values())

        return self

    # -- convenience ------------------------------------------------------
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.model_dump(), indent=indent)

    @classmethod
    def from_json(cls, raw: str) -> "SceneSpec":
        return cls.model_validate(json.loads(raw))

    @property
    def has_water(self) -> bool:
        return any(f.holds_water for f in self.features)

    def summary(self) -> str:
        """One-line human summary, handy for logs and demo output."""
        n = sum(o.quantity for o in self.objects)
        feature_text = ""
        if self.features:
            feature_text = " | " + ", ".join(
                f"{f.size} {f.type}" for f in self.features)
        return (
            f"{self.theme} | {self.terrain.type} ({self.terrain.size})"
            f"{feature_text} | "
            f"{self.lighting.time_of_day}/{self.lighting.weather}/"
            f"{self.lighting.mood} | {len(self.objects)} object types, "
            f"{n} instances"
        )
