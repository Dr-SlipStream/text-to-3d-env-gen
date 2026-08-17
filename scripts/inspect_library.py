"""
Inspect the built asset library.

Answers "what do I actually have?" -- which matters because retrieval can only
ever return what's in the manifest. A weak match usually means a missing asset,
not a broken matcher.

Usage:
    python scripts/inspect_library.py                    # overview
    python scripts/inspect_library.py --category building
    python scripts/inspect_library.py --search house
    python scripts/inspect_library.py --pack fantasy
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import vocab  # noqa: E402

MANIFEST = Path("data/asset_library/manifest.json")


def load() -> list:
    if not MANIFEST.exists():
        print(f"No manifest at {MANIFEST}. Run: python scripts/ingest_assets.py")
        sys.exit(1)
    return json.loads(MANIFEST.read_text(encoding="utf-8"))["assets"]


def overview(assets: list) -> None:
    print(f"\n{len(assets)} assets\n")

    print("By category, by pack:")
    packs = sorted({a["pack"] for a in assets})
    width = max(len(p) for p in packs) + 2

    header = " " * width + "".join(f"{c[:9]:>11}" for c in vocab.OBJECT_CATEGORIES)
    print(header)
    for pack in packs:
        row = f"{pack:<{width}}"
        for cat in vocab.OBJECT_CATEGORIES:
            n = sum(1 for a in assets if a["pack"] == pack and a["category"] == cat)
            row += f"{n if n else '-':>11}"
        print(row)

    row = f"{'TOTAL':<{width}}"
    for cat in vocab.OBJECT_CATEGORIES:
        row += f"{sum(1 for a in assets if a['category'] == cat):>11}"
    print(row)

    # The words that actually appear in asset names tell you what the library
    # can match on. If "house" isn't here, no query will ever find one.
    print("\nMost common words in asset names:")
    words = Counter()
    for a in assets:
        words.update(a["name"].split())
    for word, n in words.most_common(25):
        print(f"  {word:<18} {n:>4}")


def show_category(assets: list, category: str, limit: int) -> None:
    sel = [a for a in assets if a["category"] == category]
    print(f"\n{len(sel)} assets in '{category}':\n")
    n_mod = sum(1 for a in sel if a.get("modular"))
    if n_mod:
        print(f"  ({n_mod} of these are modular fragments, marked [mod])\n")
    for a in sorted(sel, key=lambda x: x["name"])[:limit]:
        tag = " [mod]" if a.get("modular") else ""
        print(f"  {a['name']:<38} {a['pack']}{tag}")
    if len(sel) > limit:
        print(f"  ... and {len(sel) - limit} more (use --limit)")


def search(assets: list, term: str) -> None:
    term = term.lower()
    hits = [a for a in assets if term in a["name"].lower()]
    print(f"\n{len(hits)} asset(s) with '{term}' in the name:\n")
    for a in sorted(hits, key=lambda x: x["name"]):
        tag = " [mod]" if a.get("modular") else ""
        print(f"  {a['name']:<38} {a['category']:<14} {a['pack']}{tag}")
    if not hits:
        print("  NONE. Retrieval cannot return what isn't here --")
        print("  any query for this will match something else instead.")


def show_pack(assets: list, pack_substr: str, limit: int) -> None:
    sel = [a for a in assets if pack_substr.lower() in a["pack"].lower()]
    print(f"\n{len(sel)} assets in packs matching '{pack_substr}':\n")
    by_cat = {}
    for a in sel:
        by_cat.setdefault(a["category"], []).append(a["name"])
    for cat, names in sorted(by_cat.items()):
        print(f"  {cat} ({len(names)}):")
        for n in sorted(names)[:limit]:
            print(f"      {n}")
        if len(names) > limit:
            print(f"      ... and {len(names) - limit} more")


def main() -> int:
    p = argparse.ArgumentParser(description="Inspect the asset library")
    p.add_argument("--category", help="list assets in one category")
    p.add_argument("--search", help="find assets whose name contains this")
    p.add_argument("--pack", help="list assets from packs matching this")
    p.add_argument("--limit", type=int, default=40)
    args = p.parse_args()

    assets = load()

    if args.search:
        search(assets, args.search)
    elif args.category:
        show_category(assets, args.category, args.limit)
    elif args.pack:
        show_pack(assets, args.pack, args.limit)
    else:
        overview(assets)
    return 0


if __name__ == "__main__":
    sys.exit(main())
