"""
Inspect a generated scene: what's actually in it, how big, and what colour.

Screenshots show that something is wrong; this says which asset is wrong.
Built after a forest scene filled with twenty-metre fallen logs and a village
grew cyan trees -- both invisible in the metrics, which only measure
placement validity and retrieval confidence, not whether the result looks
right.

    python scripts/inspect_scene.py outputs/a-medieval-village-on-a-sunny-day
    python scripts/inspect_scene.py outputs/<scene> --colours
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load(scene_dir: Path) -> tuple:
    manifest_path = scene_dir / "scene_manifest.json"
    record_path = scene_dir / "scene.json"
    if not manifest_path.exists():
        print(f"No scene_manifest.json in {scene_dir}")
        sys.exit(1)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = (json.loads(record_path.read_text(encoding="utf-8"))
              if record_path.exists() else None)
    return manifest, record


def biggest(scene_dir: Path, manifest: dict, limit: int) -> None:
    """Largest objects in world space -- where scale bugs show up."""
    import trimesh

    sizes = {}
    for a in manifest["assets"]:
        path = scene_dir / a["file"]
        try:
            m = trimesh.load(path, force="mesh", process=False)
            lo, hi = m.bounds
            sizes[a["file"]] = (float(max(hi[0] - lo[0], hi[2] - lo[2])),
                                float(hi[1] - lo[1]))
        except Exception:
            sizes[a["file"]] = (1.0, 1.0)

    rows = []
    for inst in manifest["instances"]:
        w, h = sizes.get(inst["asset"], (1.0, 1.0))
        s = inst["s"]
        rows.append((w * s, h * s, inst["name"], inst["category"],
                     inst["asset"].split("/")[-1], s))

    rows.sort(reverse=True)
    print(f"\nlargest instances in world space ({limit} of {len(rows)})")
    print(f"{'width_m':>9}{'height_m':>10}  {'requested':<20}"
          f"{'category':<14}{'model':<28}{'scale':>8}")
    print("-" * 92)
    seen = set()
    shown = 0
    for w, h, name, cat, model, s in rows:
        if model in seen:          # one row per model, the worst case
            continue
        seen.add(model)
        flag = "  <-- oversized" if w > 15 or h > 15 else ""
        print(f"{w:>9.1f}{h:>10.1f}  {name[:19]:<20}{cat:<14}"
              f"{model[:27]:<28}{s:>8.2f}{flag}")
        shown += 1
        if shown >= limit:
            break


def colours(scene_dir: Path, manifest: dict, limit: int) -> None:
    """Average colour per model -- where lost materials show up.

    A model that comes out neutral grey or an unexpected hue usually means
    its material didn't survive loading.
    """
    import numpy as np
    import trimesh

    print(f"\nmodel colours ({len(manifest['assets'])} models)")
    print(f"{'model':<32}{'count':>7}  {'rgb':<18}{'note'}")
    print("-" * 78)

    rows = []
    for a in sorted(manifest["assets"], key=lambda x: -x["count"]):
        path = scene_dir / a["file"]
        note, rgb = "", None
        try:
            loaded = trimesh.load(path)
            geoms = (list(loaded.geometry.values())
                     if hasattr(loaded, "geometry") else [loaded])
            samples = []
            for g in geoms:
                visual = getattr(g, "visual", None)
                kind = getattr(visual, "kind", None)
                if kind in ("vertex", "face"):
                    src = (visual.vertex_colors if kind == "vertex"
                           else visual.face_colors)
                    samples.append(np.asarray(src, dtype=float)[:, :3].mean(axis=0))
                elif kind == "texture":
                    note = "textured"
                    mat = getattr(visual, "material", None)
                    img = getattr(mat, "image", None)
                    if img is not None:
                        arr = np.asarray(img.convert("RGB"), dtype=float)
                        samples.append(arr.reshape(-1, 3).mean(axis=0))
                else:
                    note = note or "NO COLOUR DATA"
            if samples:
                rgb = np.mean(samples, axis=0)
        except Exception as e:
            note = f"unreadable ({type(e).__name__})"

        if rgb is not None:
            r, g, b = (int(v) for v in rgb)
            spread = max(r, g, b) - min(r, g, b)
            if spread < 6 and 90 < (r + g + b) / 3 < 118:
                note = "placeholder grey"
            elif b > r + 40 and b > g + 20:
                note = note or "BLUE/CYAN -- check material"
            text = f"({r:>3},{g:>3},{b:>3})"
        else:
            text = "-"
        rows.append((a["file"].split("/")[-1], a["count"], text, note))

    for model, count, text, note in rows[:limit]:
        print(f"{model[:31]:<32}{count:>7}  {text:<18}{note}")


def main() -> int:
    p = argparse.ArgumentParser(description="Inspect a generated scene")
    p.add_argument("scene_dir", type=Path)
    p.add_argument("--limit", type=int, default=25)
    p.add_argument("--colours", action="store_true",
                   help="report each model's colour and material state")
    args = p.parse_args()

    manifest, record = load(args.scene_dir)
    stats = manifest.get("stats", {})

    print(f"\n{args.scene_dir}")
    print(f"  {stats.get('instance_count', 0)} instances, "
          f"{stats.get('unique_assets', 0)} unique models, "
          f"{stats.get('rendered_triangles', 0):,} triangles")

    cats = Counter(i["category"] for i in manifest["instances"])
    print(f"  {dict(sorted(cats.items()))}")

    if args.colours:
        colours(args.scene_dir, manifest, args.limit)
    else:
        biggest(args.scene_dir, manifest, args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
