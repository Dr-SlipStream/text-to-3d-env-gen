"""
Scan downloaded 3D asset packs and build the asset manifest.

Usage:
    python scripts/ingest_assets.py
    python scripts/ingest_assets.py --raw-dir data/asset_library/raw --verbose

Expects packs unzipped under data/asset_library/raw/, one folder per pack:

    data/asset_library/raw/
        kenney_nature-kit/
        kenney_survival-kit/
        kenney_space-kit/

Writes data/asset_library/manifest.json, which the retrieval stage reads.
Safe to re-run at any time -- it rebuilds the manifest from scratch.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make `src` importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import asset_rules, vocab  # noqa: E402

DEFAULT_RAW = Path("data/asset_library/raw")
DEFAULT_MANIFEST = Path("data/asset_library/manifest.json")

# Prefer these formats when the same model ships in several. GLB first: single
# file, materials embedded, loads cleanly in trimesh, Unity and three.js.
FORMAT_PRIORITY = [".glb", ".gltf", ".obj", ".fbx", ".dae", ".stl", ".ply"]


def measure(path: Path) -> tuple:
    """Return (radius, height) in metres, or (None, None) if unmeasurable.

    `radius` is the horizontal footprint used for overlap checks during
    placement; `height` is used to sit objects correctly on the terrain.
    """
    if path.suffix.lower() not in asset_rules.MEASURABLE_EXTENSIONS:
        return None, None
    try:
        import trimesh

        mesh = trimesh.load(path, force="mesh", process=False)
        if mesh is None or not hasattr(mesh, "bounds") or mesh.bounds is None:
            return None, None

        lo, hi = mesh.bounds
        dx, dy, dz = (hi - lo)

        # glTF/OBJ are Y-up; treat Y as height and X/Z as the footprint.
        radius = float(max(dx, dz)) / 2.0
        height = float(dy)

        if not (radius > 0) or radius > 1000:      # guard against broken files
            return None, None
        return round(radius, 3), round(height, 3)
    except Exception:
        return None, None


def pick_best_file(files: list) -> Path:
    """When one model ships in multiple formats, choose the best one."""
    def rank(p: Path) -> int:
        suffix = p.suffix.lower()
        return FORMAT_PRIORITY.index(suffix) if suffix in FORMAT_PRIORITY else 99

    return sorted(files, key=rank)[0]


def ingest(raw_dir: Path, manifest_path: Path, verbose: bool = False) -> dict:
    if not raw_dir.exists():
        print(f"ERROR: {raw_dir} does not exist.")
        print("Download asset packs first -- see docs/ASSETS.md")
        return {}

    pack_dirs = [d for d in sorted(raw_dir.iterdir()) if d.is_dir()]
    if not pack_dirs:
        print(f"ERROR: no pack folders found inside {raw_dir}")
        print("Unzip each downloaded pack into its own folder there.")
        return {}

    assets = []
    skipped = 0
    unmeasured = 0

    for pack_dir in pack_dirs:
        pack = pack_dir.name
        hints = asset_rules.theme_hints(pack)

        # Group model files by cleaned name so multi-format models collapse
        # into a single catalogue entry.
        #
        # Two different things can share a cleaned name:
        #   1. the same model in several formats (house.glb + house.obj)
        #   2. genuinely different models whose IDs we stripped
        #      ("House_k6tP5nFuD2" and "House_vZ1ClbWmSx" both -> "house")
        # We must merge (1) but keep (2) as separate variants, or a village
        # ends up built from one repeated mesh.
        by_name: dict = {}
        for path in pack_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in asset_rules.SUPPORTED_EXTENSIONS:
                continue
            # Kenney packs ship previews and source files we don't want.
            lowered = str(path).lower()
            if any(x in lowered for x in ("preview", "/source/", "\\source\\")):
                continue

            name = asset_rules.clean_name(path.name)
            # Keyed by stem: same stem = same model, different format.
            by_name.setdefault(name, {}).setdefault(path.stem, []).append(path)

        pack_count = 0
        for name, by_stem in sorted(by_name.items()):
            if not name.strip():
                skipped += 1
                continue

            variants = sorted(by_stem.items())
            multi = len(variants) > 1

            for v_idx, (stem, files) in enumerate(variants, start=1):
                path = pick_best_file(files)
                category = asset_rules.classify(name, pack)
                radius, height = measure(path)

                if radius is None:
                    unmeasured += 1
                    radius = vocab.CATEGORY_DEFAULT_RADIUS.get(category, 1.0)
                    height = radius * 2

                # Same display name (so retrieval matches all variants
                # equally) but a unique id, letting the resolver pick
                # different meshes for repeated objects.
                asset_id = f"{pack}/{name}".replace(" ", "_")
                if multi:
                    asset_id += f"__v{v_idx}"

                assets.append({
                    "id": asset_id,
                    "name": name,
                    "category": category,
                    "pack": pack,
                    "file": str(path.as_posix()),
                    "format": path.suffix.lower(),
                    "radius": radius,
                    "height": height,
                    "measured": radius is not None,
                    "modular": asset_rules.is_modular(name),
                    "variant": v_idx if multi else None,
                    "theme_hints": hints,
                    "tags": asset_rules.build_tags(name, pack, category),
                })
                pack_count += 1

        if verbose:
            print(f"  {pack:<32} {pack_count:>4} assets")

    manifest = {
        "version": 1,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "asset_count": len(assets),
        "assets": assets,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # -- report -----------------------------------------------------------
    print(f"\nIndexed {len(assets)} assets from {len(pack_dirs)} pack(s)")
    print(f"Manifest: {manifest_path}\n")

    print("By category:")
    for cat in vocab.OBJECT_CATEGORIES:
        n = sum(1 for a in assets if a["category"] == cat)
        flag = "  <-- EMPTY" if n == 0 else ""
        print(f"  {cat:<14} {n:>4}{flag}")

    n_modular = sum(1 for a in assets if a.get("modular"))
    if n_modular:
        print(f"\n{n_modular} asset(s) look like modular fragments "
              "(roof corners, wall segments).")
        print("These are ranked below complete models during retrieval.")

    if unmeasured:
        print(f"\n{unmeasured} asset(s) could not be measured; using category "
              "default sizes (fine, but placement will be less precise).")

    empty = [c for c in vocab.OBJECT_CATEGORIES
             if not any(a["category"] == c for a in assets)]
    if empty:
        print(f"\nWARNING: no assets for {empty}.")
        print("Scenes needing those will fall back to a substitute category.")
        print("Consider downloading another pack -- see docs/ASSETS.md")

    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description="Build the 3D asset manifest")
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    p.add_argument("--out", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    manifest = ingest(args.raw_dir, args.out, args.verbose)
    return 0 if manifest else 1


if __name__ == "__main__":
    sys.exit(main())
