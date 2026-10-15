"""
Measure how the pipeline holds up across many prompts, not just the one we
happen to have been testing.

A system tuned on a single prompt looks finished right up until someone types
something else. This runs a spread of prompts -- in-vocabulary, near-miss,
and deliberately out-of-scope -- and reports the metrics from the project
specification: structural validity, prompt coverage, retrieval confidence,
scene density and generation time.

The point is to find where quality falls off, with numbers rather than
screenshots.

    python scripts/evaluate.py                     # standard suite
    python scripts/evaluate.py --fallback          # no LLM, faster
    python scripts/evaluate.py --suite stress      # the hard cases
    python scripts/evaluate.py --csv results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import vocab                                       # noqa: E402
from src.asset_index import AssetIndex                      # noqa: E402
from src.asset_resolution import (LOW_CONFIDENCE_BY_EMBEDDER,  # noqa: E402
                                  resolve_scene)
from src.dressing import dress_spec                         # noqa: E402
from src.layout import layout_scene, validate               # noqa: E402
from src.llm_client import LLMClient                        # noqa: E402
from src.prompt_decomposition import decompose              # noqa: E402
from src.terrain import Terrain                             # noqa: E402

# Prompts the system is designed for: the four themes, varied lighting,
# weather, scale and named objects.
CORE_SUITE = [
    "a foggy medieval village at dusk with a blacksmith forge",
    "a medieval village on a sunny day",
    "a small abandoned medieval hamlet at night",
    "a large bustling market town at noon",
    "a dense forest camp at night with a campfire",
    "a quiet woodland clearing with tents at dawn",
    "a small desert outpost in a sandstorm",
    "an abandoned desert settlement at dusk",
    "a sci-fi base on an alien planet",
    "a futuristic research station at night with neon lights",
]

# Landscape features. These were silently dropped before -- the theme matched,
# so nothing warned, and the river simply never appeared.
FEATURE_SUITE = [
    "a dense forest with a river running through it",
    "a medieval village beside a lake at dusk",
    "a camp on a hill overlooking a valley",
    "a desert outpost around a small oasis",
    "a village in a valley with a stream",
    "a forest clearing with tents at dawn",
    "a sci-fi base on a rocky hill",
    "a settlement by a wide river at night",
]

# Prompts that stretch the vocabulary: phrasings, implied objects, unusual
# combinations. These should still work.
STRETCH_SUITE = [
    "somewhere a blacksmith would work, at sunset",
    "a place travellers rest on a long road",
    "ruins of a settlement reclaimed by the forest",
    "a windswept camp on rocky ground",
    "a trading post at the edge of the wasteland",
    "an outpost where scientists study alien rock",
    "a village after a storm, everything soaked",
    "a tiny hamlet, just a few huts and a well",
]

# Deliberately outside the four supported themes. These *will* be approximated
# -- the question is whether they degrade gracefully or produce nonsense.
STRESS_SUITE = [
    "a cyberpunk alley with neon signs and rain",
    "a snowy mountain pass in a blizzard",
    "a swamp temple overgrown with vines",
    "a pirate cove with ships and barrels",
    "an underwater city of coral towers",
    "a Japanese garden with cherry blossom",
    "",                                   # empty
    "asdfgh qwerty zxcvb",                # nonsense
]

# Places outside the four settings the system started with. These used to
# collapse to "a medieval village on grass"; they are the test of whether it
# handles a prompt it was not built around.
BREADTH_SUITE = [
    "a snowy mountain village in winter",
    "a frozen outpost on the tundra at night",
    "a fishing harbour town on the coast",
    "a pirate cove with boats on the beach",
    "a tropical beach camp at sunset",
    "a witch's hut in a foggy swamp",
    "a bayou settlement on stilts at dusk",
    "a ruined castle on a hill",
    "a farm with fields beside a river",
    "a graveyard at night beside a chapel",
    "a monastery on a hilltop above a valley",
    "a lumber camp by a lake in the mountains",
    "a mining outpost in a rocky canyon",
    "a volcanic wasteland with an abandoned base",
    "a quiet meadow with a single cottage",
    "a market square in a busy town",
]

SUITES = {
    "breadth": BREADTH_SUITE,
    "core": CORE_SUITE,
    "features": FEATURE_SUITE,
    "stretch": STRETCH_SUITE,
    "stress": STRESS_SUITE,
    "all": CORE_SUITE + FEATURE_SUITE + STRETCH_SUITE + STRESS_SUITE,
}


GRAPH_MODE = False


def evaluate_prompt(prompt: str, index: AssetIndex, seed: int,
                    fallback: bool, density: float, client) -> dict:
    """Run one prompt end to end and collect metrics."""
    t0 = time.time()
    row = {"prompt": prompt or "(empty)"}

    try:
        spec = decompose(prompt, client=client, force_fallback=fallback)
        t_parse = time.time() - t0

        # What the prompt itself asked for, before any filler is added --
        # coverage is measured against this, not against the dressing.
        requested = [(o.name, o.quantity) for o in spec.objects]

        size = vocab.TERRAIN_SIZE_METRES[spec.terrain.size]
        if density > 0:
            spec, _ = dress_spec(spec, size, density=density)

        resolved = resolve_scene(spec, index)
        terrain = Terrain.from_spec(spec, seed=seed)
        terrain.apply_features(spec.features, seed=seed)
        if GRAPH_MODE:
            # The relational path: graph -> dressing -> solver.
            from src.graph_builder import build_graph
            from src.graph_dressing import dress as dress_graph
            from src.graph_pipeline import build_scene
            graph, _src = build_graph(prompt, spec, client=client,
                                      force_fallback=fallback,
                                      size_metres=size)
            dress_graph(graph, spec, size, density=max(density, 0.0))
            placed = build_scene(graph, terrain, index, spec=spec, seed=seed)
            row["_graph_skipped"] = [w for w in placed.warnings
                                     if w.startswith("skipped")]
        else:
            placed = layout_scene(resolved, terrain, seed=seed)
            from src.crossings import add_bridge_instances
            add_bridge_instances(placed, index, theme=spec.theme)
        report = validate(placed)

        threshold = LOW_CONFIDENCE_BY_EMBEDDER.get(
            getattr(index.embedder, "name", ""), 0.35)

        confidences = [o.confidence for o in resolved.objects if o.resolved]
        weak = sum(1 for c in confidences if c < threshold)

        # Record *which* queries are failing, not just how many. Knowing the
        # count tells you a theme is weak; knowing the names tells you which
        # assets to add or which query to reword.
        row["_weak_detail"] = [
            (o.name, o.match.name if o.match else "-", round(o.confidence, 2))
            for o in resolved.objects
            if o.resolved and o.confidence < threshold
        ]

        # Objects the prompt asked for that never made it into the scene.
        row["_missing"] = []

        # Coverage: of the objects the prompt asked for, how many actually
        # made it into the scene?
        placed_names = {i.name for i in placed.instances}
        covered = sum(1 for name, _ in requested if name in placed_names)
        row["_missing"] = [name for name, _ in requested
                           if name not in placed_names]

        row.update({
            "theme": spec.theme,
            "theme_recognised": spec.theme_recognised,
            "features": ",".join(f.type for f in spec.features) or "-",
            "water_pct": (round(terrain.features.water_fraction * 100, 1)
                          if terrain.features else 0.0),
            "terrain": spec.terrain.type,
            "size_m": size,
            "parser": spec.parser,
            "requested_types": len(requested),
            "covered_types": covered,
            "coverage_pct": round(100 * covered / max(len(requested), 1)),
            "instances": len(placed.instances),
            "density_per_1000m2": round(
                1000 * len(placed.instances) / (size * size), 1),
            "buildings": sum(1 for i in placed.instances
                             if i.category == "building"),
            "skipped": placed.skipped,
            "valid": report["valid"],
            "connected": report.get("connected"),
            "bridges": sum(1 for i in placed.instances
                           if getattr(i, "role", "") == "bridge"),
            "overlaps": report["overlaps"],
            "floating": report["floating"],
            "off_terrain": report["off_terrain"],
            "on_path": report["on_path"],
            "mean_conf": round(statistics.mean(confidences), 3)
                         if confidences else 0.0,
            "weak_matches": weak,
            "triangles": placed.total_triangles,
            "parse_s": round(t_parse, 2),
            "total_s": round(time.time() - t0, 2),
            "error": "",
        })
    except Exception as e:                       # never let one prompt stop the run
        row.update({"error": f"{type(e).__name__}: {e}",
                    "valid": False, "instances": 0})

    return row


def print_table(rows: list) -> None:
    print(f"\n{'prompt':<42}{'theme':<17}{'terrain':<14}{'features':<14}"
          f"{'inst':>6}{'cov':>6}{'conf':>7}{'valid':>7}{'conn':>6}{'sec':>6}")
    print("-" * 125)
    for r in rows:
        if r.get("error"):
            print(f"{r['prompt'][:43]:<44}{'ERROR':<18}{r['error'][:40]}")
            continue
        flag = "ok" if r["valid"] else "FAIL"
        conn = {True: "yes", False: "NO"}.get(r.get("connected"), "-")
        print(f"{r['prompt'][:41]:<42}{r['theme'][:16]:<17}"
              f"{r.get('terrain', '-')[:13]:<14}"
              f"{r.get('features', '-')[:13]:<14}{r['instances']:>6}"
              f"{r['coverage_pct']:>5}%{r['mean_conf']:>7.2f}"
              f"{flag:>7}{conn:>6}{r['total_s']:>6.1f}")


def report_weak(rows: list, limit: int = 30) -> None:
    """List the queries retrieval handled badly, worst first.

    A weak-match *count* says a theme is under-served; the actual query and
    what it matched says whether to add assets or reword the query.
    """
    weak = []
    for r in rows:
        for query, matched, score in r.get("_weak_detail", []):
            weak.append((score, query, matched, r["theme"]))
    if not weak:
        print("\nno weak matches")
        return

    # One row per distinct query, keeping its worst score.
    best: dict = {}
    for score, query, matched, theme in weak:
        key = (query, theme)
        if key not in best or score < best[key][0]:
            best[key] = (score, matched)

    print(f"\nweakest retrievals ({len(best)} distinct queries)")
    print(f"{'requested':<26}{'theme':<18}{'matched':<28}{'score':>7}")
    print("-" * 80)
    for (query, theme), (score, matched) in sorted(
            best.items(), key=lambda kv: kv[1][0])[:limit]:
        print(f"{query[:25]:<26}{theme:<18}{matched[:27]:<28}{score:>7.2f}")

    missing = {}
    for r in rows:
        for name in r.get("_missing", []):
            missing[name] = missing.get(name, 0) + 1
    if missing:
        print(f"\nrequested but never placed: "
              f"{', '.join(f'{k} (x{v})' for k, v in sorted(missing.items(), key=lambda kv: -kv[1])[:12])}")


def summarise(rows: list) -> None:
    ok = [r for r in rows if not r.get("error")]
    if not ok:
        print("\nevery prompt errored")
        return

    valid = sum(1 for r in ok if r["valid"])
    instances = [r["instances"] for r in ok]
    density = [r["density_per_1000m2"] for r in ok]
    coverage = [r["coverage_pct"] for r in ok]
    conf = [r["mean_conf"] for r in ok]
    times = [r["total_s"] for r in ok]

    print(f"\n{'':<26}{'min':>10}{'median':>10}{'max':>10}")
    print("-" * 56)
    for label, series in [
        ("instances per scene", instances),
        ("instances per 1000m2", density),
        ("prompt coverage %", coverage),
        ("mean confidence", conf),
        ("seconds per scene", times),
    ]:
        print(f"{label:<26}{min(series):>10.2f}"
              f"{statistics.median(series):>10.2f}{max(series):>10.2f}")

    print(f"\nstructurally valid: {valid}/{len(ok)}")
    connected = sum(1 for r in ok if r.get("connected"))
    print(f"start-to-end walkable: {connected}/{len(ok)}"
          f"   bridges placed: {sum(r.get('bridges', 0) for r in ok)}")
    skipped = sorted({w.split("'")[1] for r in ok
                      for w in r.get("_graph_skipped", []) if "'" in w})
    if skipped:
        print(f"left out (nothing like it in the library): {', '.join(skipped)}")
    errors = [r for r in rows if r.get("error")]
    if errors:
        print(f"errored: {len(errors)}")

    # Consistency is the thing being tested: a system tuned on one prompt
    # shows a wide spread here.
    spread = max(density) / max(min(density), 0.01)
    print(f"density spread (max/min): {spread:.1f}x", end="")
    print("  <- under 3x is consistent" if spread < 3
          else "  <- too variable, some scenes will look empty")

    themes = {}
    for r in ok:
        themes[r["theme"]] = themes.get(r["theme"], 0) + 1
    print(f"themes used: {themes}")


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate pipeline consistency")
    p.add_argument("--suite", choices=list(SUITES), default="all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--density", type=float, default=1.0)
    p.add_argument("--fallback", action="store_true",
                   help="skip the LLM (much faster, weaker parsing)")
    p.add_argument("--model", help="override the Ollama model")
    p.add_argument("--csv", type=Path, help="write full results here")
    p.add_argument("--weak", action="store_true",
                   help="list the queries retrieval handled badly")
    p.add_argument("--manifest", type=Path,
                   help="asset manifest to use (default: the built library)")
    p.add_argument("--graph", action="store_true",
                   help="evaluate the scene-graph path instead of the flat one")
    args = p.parse_args()
    global GRAPH_MODE
    GRAPH_MODE = args.graph

    try:
        index = (AssetIndex.from_manifest(args.manifest) if args.manifest
                 else AssetIndex.from_manifest())
    except FileNotFoundError as e:
        print(e)
        return 1

    client = LLMClient(model=args.model) if args.model else None
    prompts = SUITES[args.suite]

    print(f"library: {len(index)} assets ({index.embedder.name})")
    print(f"suite:   {args.suite}, {len(prompts)} prompts, "
          f"seed {args.seed}, density {args.density}")

    rows = []
    for i, prompt in enumerate(prompts, 1):
        print(f"  [{i}/{len(prompts)}] {prompt[:60] or '(empty)'}",
              end="\r", flush=True)
        rows.append(evaluate_prompt(prompt, index, args.seed,
                                    args.fallback, args.density, client))
    print(" " * 78, end="\r")

    print_table(rows)
    if args.weak:
        report_weak(rows)
    summarise(rows)

    unrecognised = [r for r in rows
                    if not r.get("error") and not r.get("theme_recognised", True)]
    if unrecognised:
        print(f"\ntheme not recognised for {len(unrecognised)} prompt(s) "
              "-- approximated from the closest supported theme:")
        for r in unrecognised:
            print(f"  {r['prompt'][:52]:<54} -> {r['theme']}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        # Drop the private diagnostic fields; they're lists, not CSV cells.
        clean = [{k: v for k, v in r.items() if not k.startswith("_")}
                 for r in rows]
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(clean[0].keys()))
            w.writeheader()
            w.writerows(clean)
        print(f"\nwrote {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
