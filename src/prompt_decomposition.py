"""
Stage 1 of the pipeline: free-form prompt  ->  SceneSpec.

Strategy:
  1. Try the local LLM (Ollama). It handles paraphrase, implication and
     ambiguity far better than keyword rules -- "a place where a smith works"
     becomes a forge, and "rain-slicked" implies wet weather.
  2. Validate hard against our schema. LLM output is never trusted directly.
  3. If the LLM is unavailable or its output is unusable, fall back to the
     keyword parser so the pipeline still produces something valid.
"""

from __future__ import annotations

from typing import Optional

from . import fallback_parser, vocab
from .llm_client import LLMClient, LLMUnavailable
from .schema import SceneSpec

SYSTEM_PROMPT = f"""You convert descriptions of video-game environments into strict JSON.

You must reply with ONLY a JSON object, no prose, matching this shape:

{{
  "theme": one of {vocab.THEMES},
  "art_style": one of {vocab.ART_STYLES},
  "terrain": {{
    "type": one of {vocab.TERRAIN_TYPES},
    "size": one of {vocab.TERRAIN_SIZES}
  }},
  "lighting": {{
    "time_of_day": one of {vocab.TIMES_OF_DAY},
    "weather": one of {vocab.WEATHER},
    "mood": one of {vocab.MOODS}
  }},
  "objects": [
    {{
      "name": short lowercase noun for the object, e.g. "house", "pine tree",
      "category": one of {vocab.OBJECT_CATEGORIES},
      "placement": one of {vocab.PLACEMENT_RULES},
      "quantity": integer between 1 and {vocab.MAX_QUANTITY}
    }}
  ]
}}

Rules:
- Choose the closest allowed value. Never invent values outside the lists.
- Include between 4 and {vocab.MAX_OBJECT_TYPES} object entries.
- Infer objects that the description implies but does not name. A village
  needs houses; a forest needs trees; a camp needs tents. Populate the scene
  so it feels complete, not just the nouns literally mentioned.
- Quantities should suit the scene size: a small scene has fewer objects.
- Scenery like trees and rocks should use "scatter"; buildings usually
  "along_path" or "cluster"; a single landmark uses "center".
- List only PHYSICAL objects that could be a 3D model. Never list atmosphere
  or effects -- no smoke, fog, mist, shadows, light rays, wind or weather.
  Those belong in the "lighting" fields, not "objects".
"""

USER_TEMPLATE = """Environment description:
"{prompt}"

Return the JSON object describing this environment."""


def decompose(
    prompt: str,
    client: Optional[LLMClient] = None,
    force_fallback: bool = False,
) -> SceneSpec:
    """Parse `prompt` into a validated SceneSpec.

    Never raises on bad model output -- always returns a usable spec, with
    `spec.parser` telling you which path produced it and `spec.warnings`
    recording anything that went wrong.
    """
    if force_fallback:
        return fallback_parser.parse(prompt)

    client = client or LLMClient()

    if not client.is_available():
        spec = fallback_parser.parse(prompt)
        spec.warnings.append(
            "LLM unavailable (is Ollama running?); used keyword fallback"
        )
        return spec

    try:
        raw = client.generate_json(
            system=SYSTEM_PROMPT, user=USER_TEMPLATE.format(prompt=prompt)
        )
    except LLMUnavailable as e:
        spec = fallback_parser.parse(prompt)
        spec.warnings.append(f"LLM error ({e}); used keyword fallback")
        return spec

    # The schema's validators snap near-miss values onto our vocabulary, so
    # small deviations survive; only structurally broken output fails here.
    try:
        raw.setdefault("objects", [])
        spec = SceneSpec.model_validate(raw)
    except Exception as e:
        spec = fallback_parser.parse(prompt)
        spec.warnings.append(f"LLM output failed validation ({e}); used fallback")
        return spec

    spec.source_prompt = prompt
    spec.parser = "llm"

    # The LLM must pick from our theme list, so it always returns something
    # plausible. Cross-check against keyword evidence: if the prompt contains
    # no word associated with any supported theme, the choice was a guess and
    # the user should know.
    lowered = f" {prompt.lower()} "
    if not any(kw in lowered
               for words in vocab.THEME_KEYWORDS.values() for kw in words):
        spec.theme_recognised = False
        spec.warnings.append(
            f"prompt names no supported theme; generated as "
            f"'{spec.theme}'. Supported: {', '.join(vocab.THEMES)}"
        )

    # An LLM occasionally returns an empty or near-empty object list. A scene
    # with nothing in it is useless, so top it up from the keyword parser.
    if len(spec.objects) < 3:
        backup = fallback_parser.parse(prompt)
        existing = {o.name for o in spec.objects}
        for obj in backup.objects:
            if obj.name not in existing and len(spec.objects) < 6:
                spec.objects.append(obj)
        spec.warnings.append("LLM returned too few objects; topped up from fallback")

    return spec
