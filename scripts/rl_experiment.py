"""
Train navigation agents on generated scenes and report what happens.

    python scripts/rl_experiment.py --episodes 800 --seeds 5

Writes a JSON record of everything measured, plus the figures the report
uses. Every number in the report comes from a run of this script.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import fallback_parser, vocab                     # noqa: E402
from src.asset_index import AssetIndex                     # noqa: E402
from src.graph_builder import compile_graph                # noqa: E402
from src.graph_dressing import dress                       # noqa: E402
from src.graph_pipeline import build_scene                 # noqa: E402
from src.rl import (aggregate, env_from_scene, greedy_rollout,  # noqa: E402
                    make_agent, render_ascii, run_experiment, summarise,
                    train, evaluate)
from src.terrain import Terrain                            # noqa: E402

# The scenes agents are trained on. Chosen to vary in how open they are:
# a village is cluttered with buildings, a forest camp is mostly passable
# undergrowth, a desert outpost is nearly empty.
SCENES = [
    ("village", "a medieval village on a sunny day"),
    ("forest_camp", "a dense forest camp around a campfire"),
    ("desert_outpost", "a small desert outpost"),
    ("river_village", "a medieval village beside a river"),
]

# Each entry is (label, agent kind, whether reward shaping is enabled).
#
# Q-learning appears twice on purpose. With one goal cell in sixteen hundred a
# random walk reaches it in roughly one episode in a hundred, so an
# unshaped agent sees almost no reward and learns nothing -- and that failure
# is worth reporting rather than tuning away quietly.
METHODS = [
    ("Random policy", "random", False),
    ("Q-learning (sparse reward)", "q_learning", False),
    ("Q-learning (shaped reward)", "q_learning", True),
    ("SARSA (shaped reward)", "sarsa", True),
]

HYPERPARAMETERS = {
    "alpha": 0.15,
    "gamma": 0.95,
    "epsilon": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.995,
}


def build_scene_for(prompt: str, index: AssetIndex, seed: int,
                    size: str = "medium"):
    spec = fallback_parser.parse(prompt)
    spec.terrain.size = size
    metres = vocab.TERRAIN_SIZE_METRES[size]

    graph = compile_graph(spec)
    dress(graph, spec, metres)

    terrain = Terrain.from_spec(spec, seed=seed)
    terrain.apply_features(spec.features, seed=seed)
    return build_scene(graph, terrain, index, spec=spec, seed=seed)


def main() -> int:
    p = argparse.ArgumentParser(description="RL experiments on generated scenes")
    p.add_argument("--episodes", type=int, default=800)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--resolution", type=int, default=40)
    p.add_argument("--slip", type=float, default=0.0,
                   help="probability a move goes astray; >0 makes the MDP "
                        "stochastic")
    p.add_argument("--manifest", type=Path)
    p.add_argument("--out", type=Path, default=Path("outputs/rl"))
    p.add_argument("--figures", action="store_true",
                   help="also write plots (needs matplotlib)")
    args = p.parse_args()

    index = (AssetIndex.from_manifest(args.manifest) if args.manifest
             else AssetIndex.from_manifest())
    seeds = list(range(args.seeds))
    args.out.mkdir(parents=True, exist_ok=True)

    record = {
        "episodes": args.episodes,
        "seeds": seeds,
        "resolution": args.resolution,
        "slip": args.slip,
        "hyperparameters": HYPERPARAMETERS,
        "scenes": {},
    }

    print(f"\n{'scene':<16}{'states':>8}{'walkable':>10}{'reachable':>11}"
          f"{'optimal':>9}  connected")
    print("-" * 66)

    trainable = []
    for name, prompt in SCENES:
        scene = build_scene_for(prompt, index, seed=0)
        env = env_from_scene(scene, resolution=args.resolution, seed=0,
                             slip=args.slip, shaping=1.0)
        optimal = env.shortest_path_length()
        reachable = env.reachable_fraction()

        record["scenes"][name] = {
            "prompt": prompt,
            "instances": len(scene.instances),
            "states": env.n_states,
            "walkable_cells": int(env.walkable.sum()),
            "walkable_fraction": float(env.walkable.mean()),
            "reachable_fraction": reachable,
            "optimal_steps": optimal,
            "connected": optimal is not None,
        }

        print(f"{name:<16}{env.n_states:>8}{env.walkable.mean():>9.0%}"
              f"{reachable:>11.0%}"
              f"{(str(optimal) if optimal else '-'):>9}  "
              f"{'yes' if optimal else 'NO -- goal unreachable'}")

        (args.out / f"map_{name}.txt").write_text(
            render_ascii(env), encoding="utf-8")

        if optimal is not None:
            trainable.append((name, prompt, optimal))

    if not trainable:
        print("\nNo connected scene to train on.")
        return 1

    # ---- training -------------------------------------------------------
    print(f"\nTraining {len(METHODS)} methods x {len(seeds)} seeds x "
          f"{args.episodes} episodes on {len(trainable)} scenes\n")

    for name, prompt, optimal in trainable:
        print(f"  {name} (optimal route {optimal} steps)")

        scene_record = record["scenes"][name]
        scene_record["methods"] = {}

        for label, method, shaped in METHODS:
            def factory(seed, _prompt=prompt, _shaped=shaped):
                scene = build_scene_for(_prompt, index, seed=seed)
                return env_from_scene(scene, resolution=args.resolution,
                                      seed=seed, slip=args.slip,
                                      shaping=1.0 if _shaped else 0.0)

            t0 = time.time()
            kwargs = {} if method == "random" else dict(HYPERPARAMETERS)
            histories, evaluations = run_experiment(
                factory, method, seeds, episodes=args.episodes, **kwargs)

            summary = summarise(evaluations)
            mean_curve, std_curve = aggregate(histories)

            # How many episodes before the smoothed return reaches most of
            # what the method eventually achieves. A convergence speed that
            # does not depend on knowing the optimum in advance.
            final = float(mean_curve[-1]) if len(mean_curve) else 0.0
            threshold = final - abs(final) * 0.1
            convergence = [h.episodes_to_threshold(threshold)
                           for h in histories]
            converged = [c for c in convergence if c is not None]

            scene_record["methods"][label] = {
                "agent": method,
                "shaped": shaped,
                **summary,
                "final_train_return": final,
                "episodes_to_converge": (float(np.mean(converged))
                                         if converged else None),
                "optimal_steps": optimal,
                "excess_steps": summary["mean_steps"] - optimal,
                "train_seconds": round(time.time() - t0, 1),
                "curve_mean": [round(v, 2) for v in mean_curve.tolist()],
                "curve_std": [round(v, 2) for v in std_curve.tolist()],
                "success_by_episode": [
                    round(100 * float(np.mean(
                        [h.successes[i] for h in histories])), 1)
                    for i in range(0, args.episodes, max(args.episodes // 40, 1))
                ],
            }

            conv = (f"{np.mean(converged):.0f}" if converged else "-")
            print(f"    {label:<28} return {summary['mean_return']:>7.1f} "
                  f"+/-{summary['mean_return_std']:>5.1f}   "
                  f"success {summary['success_rate']:>5.1f}%   "
                  f"steps {summary['mean_steps']:>6.1f} "
                  f"(opt {optimal})   conv@{conv}")

        # A picture of the route the trained agent settles on.
        env = env_from_scene(build_scene_for(prompt, index, seed=0),
                             resolution=args.resolution, seed=0,
                             slip=args.slip, shaping=1.0)
        agent = make_agent("q_learning", env.n_states, env.n_actions, seed=0,
                           **HYPERPARAMETERS)
        train(env, agent, episodes=args.episodes)
        path = greedy_rollout(env, agent)
        (args.out / f"route_{name}.txt").write_text(
            render_ascii(env, path), encoding="utf-8")

    (args.out / "results.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nWritten to {args.out}/results.json")

    if args.figures:
        write_figures(record, args.out)
        print(f"Figures written to {args.out}/")

    return 0


def write_figures(record: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, scene in record["scenes"].items():
        if "methods" not in scene:
            continue

        fig, ax = plt.subplots(figsize=(7, 4.2))
        for method, data in scene["methods"].items():
            mean = np.asarray(data["curve_mean"])
            std = np.asarray(data["curve_std"])
            x = np.arange(len(mean))
            ax.plot(x, mean, label=method.replace("_", "-"), linewidth=1.6)
            ax.fill_between(x, mean - std, mean + std, alpha=0.18)

        ax.set_xlabel("Episode")
        ax.set_ylabel("Episodic return (50-episode moving average)")
        ax.set_title(f"Training performance: {name.replace('_', ' ')}")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / f"training_{name}.png", dpi=160)
        plt.close(fig)

    # Success rate over training, all scenes, Q-learning only.
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, scene in record["scenes"].items():
        if "methods" not in scene:
            continue
        data = scene["methods"]["Q-learning (shaped reward)"][
            "success_by_episode"]
        x = np.linspace(0, record["episodes"], len(data))
        ax.plot(x, data, label=name.replace("_", " "), linewidth=1.6)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Goal reached (%)")
    ax.set_title("Q-learning success rate during training")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "success_rates.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
