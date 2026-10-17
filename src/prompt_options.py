"""
Generation settings, read from the prompt itself.

Every switch the command line used to need -- seed, size, density, how many
layouts to try, whether to export for Unity -- can be said in the sentence:

    "a large foggy village beside a river, best of 5, for unity"
    "a sparse desert outpost at night, variation 7"
    "a forest camp, quick draft"

The phrases that are purely settings ("best of 5", "variation 7", "for
unity") are removed before the prompt reaches the language model, so they do
not end up as objects in the scene. Words that describe the place as well as
set something ("dense", "sparse") are left in, because the model should hear
them too.

parse() returns the cleaned prompt, the settings, and a list of plain-English
notes saying what was understood -- which the web interface shows back to the
user, so nothing is changed silently.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Options:
    seed: Optional[int] = None
    size: Optional[str] = None            # small | medium | large
    density: float = 1.0
    mode: str = "graph"                   # graph | rules
    offline: bool = False                 # skip the language model
    candidates: int = 3                   # layouts generated and playtested
    playtest: bool = True                 # run the RL bot on the winner
    unity: bool = False
    palette: bool = False
    baked: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
              "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_NUM = r"(\d+|" + "|".join(_NUM_WORDS) + r")"


def _num(text: str) -> int:
    return int(text) if text.isdigit() else _NUM_WORDS[text]


# (pattern, handler). A handler gets the match and the options, and returns
# True if the phrase is purely a setting and should be cut from the prompt.
def _seed(m, o):
    o.seed = int(m.group(1))
    o.notes.append(f"variation {o.seed} (fixed seed, reproducible)")
    return True


def _best_of(m, o):
    o.candidates = max(1, min(8, _num(m.group(1))))
    o.notes.append(f"generate {o.candidates} layouts, keep the one that "
                   "plays best")
    return True


def _metres(m, o):
    metres = int(m.group(1))
    o.size = "small" if metres <= 90 else "medium" if metres <= 160 else "large"
    o.notes.append(f"{metres} m across -> {o.size} map")
    return True


def _unity(m, o):
    o.unity = True
    o.notes.append("export a Unity bundle as well")
    return True


def _offline(m, o):
    o.offline = True
    o.notes.append("offline: keyword parsing, no language model")
    return True


def _rules(m, o):
    o.mode = "rules"
    o.notes.append("simple rule-based layout instead of the scene graph")
    return True


def _palette(m, o):
    o.palette = True
    o.notes.append("recolour everything from one palette")
    return True


def _baked(m, o):
    o.baked = True
    o.notes.append("bake into a single .glb")
    return True


def _no_bot(m, o):
    o.playtest = False
    o.candidates = 1
    o.notes.append("skip the bot playtest")
    return True


def _tiny(m, o):
    o.size = "small"
    o.notes.append("small map")
    return False


def _huge(m, o):
    o.size = "large"
    o.notes.append("large map")
    return False


def _dense(m, o):
    o.density = 1.4
    o.notes.append("denser dressing")
    return False


def _sparse(m, o):
    o.density = 0.6
    o.notes.append("sparser dressing")
    return False


RULES = [
    (r"\b(?:seed|variation|version|take|variant)\s*#?\s*(\d+)\b", _seed),
    (r"\bbest\s+of\s+" + _NUM + r"\b", _best_of),
    (r"\b(?:try|generate|make)\s+" + _NUM
     + r"\s+(?:layouts?|candidates?|options?|versions?)\b", _best_of),
    (r"\b(\d{2,3})\s*(?:m|metres?|meters?)\s*(?:wide|across|map|square)?\b",
     _metres),
    (r"\b(?:for|into|to|export(?:ed)?\s+(?:for|to))\s+unity\b|\bunity\s+"
     r"(?:export|bundle|ready)\b", _unity),
    (r"\b(?:offline|no\s+ai|without\s+(?:the\s+)?(?:ai|llm|model)|"
     r"quick\s+draft)\b", _offline),
    (r"\b(?:simple|rule[- ]based|classic)\s+layout\b", _rules),
    (r"\b(?:one|single|unified|consistent)\s+(?:colou?r\s+)?palette\b|"
     r"\bconsistent\s+colou?rs\b", _palette),
    (r"\b(?:baked|single[- ]file)\b", _baked),
    (r"\b(?:no|skip(?:\s+the)?|without(?:\s+the)?)\s+(?:bot|playtest)\b",
     _no_bot),
    (r"\b(?:tiny|small|little|compact)\b", _tiny),
    (r"\b(?:huge|vast|sprawling|large|big|massive)\b", _huge),
    (r"\b(?:dense|densely|packed|crowded|lush|overgrown|thick)\b", _dense),
    (r"\b(?:sparse|sparsely|minimal|empty|bare)\b", _sparse),
]


def parse(prompt: str) -> Tuple[str, Options]:
    """Pull settings out of a prompt. Returns (cleaned prompt, options)."""
    options = Options()
    text = prompt.strip()

    for pattern, handler in RULES:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        strip = handler(match, options)
        if strip:
            text = text[:match.start()] + " " + text[match.end():]

    # Tidy whatever the cuts left behind: doubled commas, stray "and".
    text = re.sub(r"\s*,\s*(,\s*)+", ", ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(?:,|\band\b|\bwith\b)\s*$", "", text.strip(),
                  flags=re.IGNORECASE)
    text = text.strip(" ,.;")
    return text or prompt.strip(), options
