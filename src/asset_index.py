"""
Stage 2 of the pipeline: SceneObject -> concrete 3D asset.

The scene spec says "blacksmith forge"; the asset library contains files named
things like "building_smithy" or "forge_stone". Exact string matching fails on
that, so we embed both sides into vectors and match by meaning.

Two embedders, same interface:

  SentenceTransformerEmbedder -- real semantic embeddings (all-MiniLM-L6-v2,
      ~90 MB, downloads once, then runs offline on CPU or GPU). Understands
      that "smithy" and "forge" are related.

  HashingEmbedder -- deterministic character-ngram vectors, no download, no
      dependencies. Weaker (matches spelling, not meaning) but always works,
      so tests and offline demos never break.

The index itself is plain numpy cosine similarity. With a few thousand assets
that's microseconds per query -- a heavier vector database would add install
friction and dependencies for no gain at this scale.
"""

from __future__ import annotations

import json
import re
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Protocol

import numpy as np

from . import vocab

DEFAULT_MANIFEST = Path("data/asset_library/manifest.json")

# How far to demote snap-together fragments relative to complete models.
MODULAR_PENALTY = 0.20

# Theme adjustments are a tie-breaker, not a veto.
#
# Originally +0.12 / -0.10 -- a 0.22 swing, large enough to beat the semantic
# signal outright. A medieval prompt asking for a "barrel" got a *canoe*,
# because the canoe sat in an on-theme pack while all seven real barrels sat
# in off-theme ones. Theme should nudge between comparable options, never
# override a better match.
THEME_BONUS = 0.05
THEME_PENALTY = 0.04

# A literal word match is strong evidence and should outrank pack politics:
# an asset actually named "barrel" beats an on-theme canoe.
EXACT_WORD_BONUS = 0.15


# ---------------------------------------------------------------------------
# Embedders
# ---------------------------------------------------------------------------

class Embedder(Protocol):
    name: str

    def encode(self, texts: List[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised row vectors."""
        ...


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


class HashingEmbedder:
    """Character-ngram hashing. No model download, fully deterministic.

    Catches shared substrings, so "tree" matches "tree_pine" and "barrel"
    matches "barrel_large", but it has no concept of synonyms.

    Uses CRC32 rather than Python's built-in hash(): hash() is randomised per
    process for strings, which would make retrieval results differ between
    runs and break reproducibility.
    """

    name = "hashing"

    def __init__(self, dim: int = 512, ngram_range: tuple = (3, 5)):
        self.dim = dim
        self.ngram_range = ngram_range

    @staticmethod
    def _stable_hash(text: str) -> int:
        return zlib.crc32(text.encode("utf-8"))

    def _features(self, text: str) -> List[str]:
        text = re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()
        words = text.split()
        feats = list(words)                       # whole words matter most
        padded = f" {' '.join(words)} "
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(padded) - n + 1):
                feats.append(padded[i : i + n])
        return feats

    def encode(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat in self._features(text):
                idx = self._stable_hash(feat) % self.dim
                # Whole words weigh more than character fragments.
                out[row, idx] += 2.0 if " " not in feat and len(feat) > 5 else 1.0
        return _normalise(out)


class SentenceTransformerEmbedder:
    """Real semantic embeddings. Downloads ~90 MB on first use, then offline."""

    name = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is not installed.\n"
                "  pip install sentence-transformers"
            ) from e
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )
        return _normalise(np.asarray(vecs, dtype=np.float32))


def get_embedder(prefer_semantic: bool = True) -> Embedder:
    """Best available embedder, degrading gracefully if the model is absent."""
    if prefer_semantic:
        try:
            return SentenceTransformerEmbedder()
        except Exception:
            pass
    return HashingEmbedder()


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

class AssetMatch:
    """One retrieval result."""

    def __init__(self, asset: dict, score: float, reason: str = ""):
        self.asset = asset
        self.score = score
        self.reason = reason

    @property
    def name(self) -> str:
        return self.asset["name"]

    @property
    def category(self) -> str:
        return self.asset["category"]

    @property
    def file(self) -> str:
        return self.asset["file"]

    @property
    def radius(self) -> float:
        return float(self.asset.get("radius", 1.0))

    @property
    def height(self) -> float:
        return float(self.asset.get("height", 2.0))

    def __repr__(self) -> str:
        return f"<AssetMatch {self.name!r} ({self.category}) {self.score:.3f}>"


class AssetIndex:
    """Searchable index over the asset manifest."""

    def __init__(self, assets: List[dict], embedder: Optional[Embedder] = None):
        if not assets:
            raise ValueError("cannot build an index with no assets")
        self.assets = assets
        self.embedder = embedder or get_embedder()
        self._matrix = self.embedder.encode(
            [self._searchable_text(a) for a in assets]
        )
        self._by_category: Dict[str, List[int]] = {}
        # Word sets per asset, for the literal-match pre-filter.
        self._name_words: List[set] = []
        for i, a in enumerate(assets):
            self._by_category.setdefault(a["category"], []).append(i)
            self._name_words.append({
                w for w in re.split(r"[^a-z0-9]+", a.get("name", "").lower())
                if len(w) > 2
            })

    # -- construction -----------------------------------------------------
    @classmethod
    def from_manifest(
        cls,
        path: Path = DEFAULT_MANIFEST,
        embedder: Optional[Embedder] = None,
    ) -> "AssetIndex":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No manifest at {path}.\n"
                "Download asset packs (docs/ASSETS.md), then run:\n"
                "  python scripts/ingest_assets.py"
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data["assets"], embedder=embedder)

    @staticmethod
    def _searchable_text(asset: dict) -> str:
        """What we embed for an asset: its name, weighted, plus its tags."""
        name = asset.get("name", "")
        tags = " ".join(asset.get("tags", []))
        # Repeat the name so it dominates the vector over incidental tags.
        return f"{name} {name} {asset.get('category', '')} {tags}".strip()

    # -- search -----------------------------------------------------------
    def search(
        self,
        query: str,
        category: Optional[str] = None,
        theme: Optional[str] = None,
        top_k: int = 5,
    ) -> List[AssetMatch]:
        """Find assets matching `query`.

        `category` restricts the search (a "house" must be a building, never a
        bush). `theme` softly boosts assets from packs suited to that theme,
        so a sci-fi scene prefers sci-fi props without excluding neutral ones.
        """
        qvec = self.embedder.encode([query])[0]
        scores = self._matrix @ qvec                     # cosine similarity

        query_words = {w for w in re.split(r"[^a-z0-9]+", query.lower())
                       if len(w) > 2}

        candidates = self._by_category.get(category) if category else None
        if category and not candidates:
            # Nothing in that category -- search everything rather than fail.
            candidates = None

        idxs = list(candidates if candidates is not None
                    else range(len(self.assets)))

        # If any asset's name literally contains a word from the query, search
        # only those. A word match is far stronger evidence than embedding
        # similarity, and without this the search talks itself out of the
        # obvious answer whenever one theme dominates the library: a sci-fi
        # scene asking for a "platform" matched an "archery building", and
        # "crystal" matched "astronaut", because medieval assets outnumbered
        # sci-fi ones ten to one.
        if query_words:
            literal = [i for i in idxs
                       if query_words & self._name_words[i]]
            if literal:
                idxs = literal

        results = []
        for i in idxs:
            asset = self.assets[i]
            score = float(scores[i])
            reason = ""

            # A literal word match is the strongest signal we have. Without
            # this, embedding similarity alone let an on-theme canoe outrank
            # an asset literally named "barrel".
            asset_words = set(asset.get("name", "").lower().split())
            if query_words & asset_words:
                score += EXACT_WORD_BONUS
                reason = "exact word"

            # A modular fragment ("roof corner inner") is a poor stand-in for
            # a whole object, so rank it below complete models -- but keep it
            # available in case the library has nothing better.
            if asset.get("modular"):
                score -= MODULAR_PENALTY
                reason = reason or "modular fragment"

            if theme:
                hints = asset.get("theme_hints") or []
                if hints:
                    if theme in hints:
                        score += THEME_BONUS
                        reason = reason or "theme match"
                    else:
                        score -= THEME_PENALTY
                        reason = reason or "off-theme pack"

            results.append(AssetMatch(asset, score, reason))

            results.append(AssetMatch(asset, score, reason))

        results.sort(key=lambda m: m.score, reverse=True)
        return results[:top_k]

    def best(
        self,
        query: str,
        category: Optional[str] = None,
        theme: Optional[str] = None,
    ) -> Optional[AssetMatch]:
        hits = self.search(query, category=category, theme=theme, top_k=1)
        return hits[0] if hits else None

    # -- introspection ----------------------------------------------------
    def categories(self) -> Dict[str, int]:
        return {c: len(v) for c, v in sorted(self._by_category.items())}

    def __len__(self) -> int:
        return len(self.assets)
