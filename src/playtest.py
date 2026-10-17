"""
Bot playtesting: scoring a generated scene by how it plays.

Geometric checks say whether a scene is *built* correctly -- nothing floats,
nothing overlaps. They cannot say whether it is any good to move through. A
level designer finds that out by playing it; this module does the same thing
with a reinforcement-learning agent and a few measurements taken along the
route it learns.

What comes out is a set of gameplay-flow metrics, each with a raw value, a
0-100 score and a one-line reading, plus an overall flow score. The pipeline
uses the overall score to choose between candidate layouts -- generate
several, playtest each, keep the one that plays best -- which closes the loop
from "generate a scene" to "generate a scene a player can actually use".

    reachability   share of open ground a player can get to from the start
    connected      whether the far end of the road can be reached at all
    directness     straight-line distance over the walking distance
    learnability   how quickly and reliably a Q-learning bot learns the route
    pacing         the longest stretch of the route with nothing to look at
    chokepoints    places along the route narrower than a single cell
    openness       how much of the map is open to explore
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .rl.agents import make_agent
from .rl.gridworld import ACTIONS, env_from_scene
from .rl.train import evaluate, greedy_rollout, train

# Weights of each metric in the overall flow score. Learnability carries the
# most because it is the one no geometric check can stand in for.
WEIGHTS = {
    "reachability": 0.20,
    "learnability": 0.25,
    "pacing": 0.20,
    "directness": 0.10,
    "chokepoints": 0.10,
    "openness": 0.15,
}

# Things a player notices on the way past: the landmarks that pace a route.
LANDMARK_CATEGORIES = {"building", "structure", "light_source"}
LANDMARK_REACH = 14.0          # metres from the route to count as "passed"

AGENT_PARAMS = dict(alpha=0.5, gamma=0.95, epsilon=1.0,
                    epsilon_min=0.05, epsilon_decay=0.985)


@dataclass
class Metric:
    key: str
    label: str
    value: float               # the raw measurement
    unit: str
    score: float               # 0-100, higher is better
    note: str                  # what the number means for this scene

    def to_dict(self) -> dict:
        return {"key": self.key, "label": self.label,
                "value": round(float(self.value), 3), "unit": self.unit,
                "score": round(float(self.score), 1), "note": self.note}


@dataclass
class PlaytestReport:
    flow_score: float
    connected: bool
    metrics: List[Metric]
    grid: List[str] = field(default_factory=list)
    optimal_route: List[Tuple[int, int]] = field(default_factory=list)
    bot_route: List[Tuple[int, int]] = field(default_factory=list)
    landmarks: List[Tuple[int, int]] = field(default_factory=list)
    start: Tuple[int, int] = (0, 0)
    goal: Tuple[int, int] = (0, 0)
    learning_curve: List[float] = field(default_factory=list)
    steps_curve: List[float] = field(default_factory=list)
    optimal_steps: Optional[int] = None
    episodes: int = 0
    seconds: float = 0.0
    cell_metres: float = 0.0

    @property
    def grade(self) -> str:
        s = self.flow_score
        return ("A" if s >= 85 else "B" if s >= 70 else "C" if s >= 55
                else "D" if s >= 40 else "F")

    def metric(self, key: str) -> Optional[Metric]:
        return next((m for m in self.metrics if m.key == key), None)

    def to_dict(self) -> dict:
        return {
            "flow_score": round(self.flow_score, 1),
            "grade": self.grade,
            "connected": self.connected,
            "metrics": [m.to_dict() for m in self.metrics],
            "weights": WEIGHTS,
            "grid": self.grid,
            "optimal_route": [list(c) for c in self.optimal_route],
            "bot_route": [list(c) for c in self.bot_route],
            "landmarks": [list(c) for c in self.landmarks],
            "start": list(self.start),
            "goal": list(self.goal),
            "learning_curve": [round(v, 3) for v in self.learning_curve],
            "steps_curve": [round(v, 1) for v in self.steps_curve],
            "optimal_steps": self.optimal_steps,
            "episodes": self.episodes,
            "seconds": round(self.seconds, 2),
            "cell_metres": round(self.cell_metres, 2),
        }

    def summary(self) -> str:
        parts = [f"flow {self.flow_score:.0f}/100 ({self.grade})"]
        for m in self.metrics:
            parts.append(f"{m.key} {m.score:.0f}")
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def _bfs(walkable: np.ndarray, start) -> np.ndarray:
    rows, cols = walkable.shape
    dist = np.full(walkable.shape, -1, dtype=int)
    if not walkable[start]:
        return dist
    dist[start] = 0
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dc, dr in ACTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and walkable[nr, nc] \
                    and dist[nr, nc] < 0:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def _trace(dist: np.ndarray, goal) -> List[Tuple[int, int]]:
    """Walk back down a BFS distance field from the goal to the start."""
    if dist[goal] < 0:
        return []
    rows, cols = dist.shape
    route = [goal]
    cell = goal
    while dist[cell] > 0:
        r, c = cell
        for dc, dr in ACTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols \
                    and dist[nr, nc] == dist[cell] - 1:
                cell = (nr, nc)
                break
        route.append(cell)
    return route[::-1]


def _clearance(walkable: np.ndarray) -> np.ndarray:
    """Distance in cells from every walkable cell to the nearest obstacle.

    The map edge counts as an obstacle: a corridor along the border of the
    world is as narrow as one between two walls.
    """
    rows, cols = walkable.shape
    dist = np.full(walkable.shape, np.inf)
    q = deque()
    for r in range(rows):
        for c in range(cols):
            if not walkable[r, c]:
                dist[r, c] = 0
                q.append((r, c))
    for r in range(rows):
        for c in (0, cols - 1):
            if walkable[r, c] and dist[r, c] > 1:
                dist[r, c] = 1
                q.append((r, c))
    for c in range(cols):
        for r in (0, rows - 1):
            if walkable[r, c] and dist[r, c] > 1:
                dist[r, c] = 1
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dc, dr in ACTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols \
                    and dist[nr, nc] > dist[r, c] + 1:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


# ---------------------------------------------------------------------------
# The playtest
# ---------------------------------------------------------------------------

def playtest(scene, resolution: int = 40, episodes: int = 300,
             seeds: Tuple[int, ...] = (0, 1), train_bot: bool = True
             ) -> PlaytestReport:
    """Play a generated scene and score how it plays.

    With `train_bot` off only the structural metrics run, which takes
    milliseconds -- enough to rank candidate layouts. The winner is then
    played properly.
    """
    t0 = time.time()
    env = env_from_scene(scene, resolution=resolution, shaping=1.0)
    walkable = env.walkable
    rows, cols = walkable.shape
    cell_m = scene.terrain.size / resolution
    start, goal = env.start, env.goal

    dist = _bfs(walkable, start)
    connected = dist[goal] >= 0
    optimal = int(dist[goal]) if connected else None
    route = _trace(dist, goal) if connected else []

    metrics: List[Metric] = []

    # -- reachability -------------------------------------------------------
    open_cells = int(walkable.sum())
    reached = int((dist >= 0).sum())
    reach = reached / max(open_cells, 1)
    metrics.append(Metric(
        "reachability", "Reachability", reach * 100, "%",
        reach * 100,
        "every part of the map can be reached" if reach > 0.97 else
        f"{(1 - reach) * 100:.0f}% of open ground is cut off from the start"))

    # -- directness ----------------------------------------------------------
    manhattan = abs(goal[0] - start[0]) + abs(goal[1] - start[1])
    if optimal:
        direct = manhattan / optimal
        # A little winding is good -- a road that bends shows the player
        # more -- so the score peaks short of a perfectly straight line.
        d_score = 100 * min(1.0, direct / 0.85)
        note = (f"walking route is {optimal * cell_m:.0f} m for "
                f"{manhattan * cell_m:.0f} m as the crow flies")
    else:
        direct, d_score, note = 0.0, 0.0, "the goal cannot be reached"
    metrics.append(Metric("directness", "Route directness", direct * 100,
                          "%", d_score, note))

    # -- chokepoints -----------------------------------------------------------
    # A pinch is a cell of the route walled in on both opposite sides -- a
    # bridge deck, a gap between two houses. One or two are good design (a
    # bridge is a chokepoint on purpose); a route that is all pinches is a
    # corridor.
    def blocked(r, c):
        return not (0 <= r < rows and 0 <= c < cols) or not walkable[r, c]

    def pinched(cell):
        r, c = cell
        return ((blocked(r, c - 1) and blocked(r, c + 1))
                or (blocked(r - 1, c) and blocked(r + 1, c)))

    pinches, prev, narrow = 0, False, 0
    for c in route:
        now = pinched(c)
        narrow += now
        if now and not prev:
            pinches += 1
        prev = now
    c_score = max(0.0, 100 - 20 * max(0, pinches - 2)) if route else 0.0
    metrics.append(Metric(
        "chokepoints", "Chokepoints", pinches, "",
        c_score,
        "no bottlenecks on the main route" if pinches == 0 else
        f"{pinches} pinch{'es' if pinches != 1 else ''} on the route "
        f"({narrow * cell_m:.0f} m single-file)"
        + (" -- a bridge or gate is a good one" if pinches <= 2 else "")))

    # -- openness -------------------------------------------------------------
    open_frac = open_cells / (rows * cols)
    # Most of a game map should be walkable, but not all of it: a map with
    # nothing in the way has nothing to navigate. Best between 60 and 85%.
    if open_frac < 0.6:
        o_score = 100 * open_frac / 0.6
    elif open_frac <= 0.85:
        o_score = 100.0
    else:
        o_score = max(40.0, 100 - (open_frac - 0.85) * 400)
    metrics.append(Metric(
        "openness", "Open ground", open_frac * 100, "%", o_score,
        "balanced open ground and obstacles" if 0.6 <= open_frac <= 0.85
        else ("crowded: little room to move" if open_frac < 0.6
              else "very open: little to navigate around")))

    # -- pacing -----------------------------------------------------------------
    landmarks = []
    for inst in scene.instances:
        if inst.category in LANDMARK_CATEGORIES and inst.radius >= 1.0 \
                and getattr(inst, "role", "") != "bridge":
            x, _y, z = inst.position
            landmarks.append((x, z))

    landmark_cells = sorted({(int(min(z / cell_m, rows - 1)),
                              int(min(x / cell_m, cols - 1)))
                             for x, z in landmarks})

    if route and landmarks:
        pts = np.array([((c + 0.5) * cell_m, (r + 0.5) * cell_m)
                        for r, c in route])
        lm = np.array(landmarks)
        near = np.min(np.hypot(pts[:, None, 0] - lm[None, :, 0],
                               pts[:, None, 1] - lm[None, :, 1]), axis=1)
        seen = near <= LANDMARK_REACH
        longest = run = 0
        for s in seen:
            run = 0 if s else run + 1
            longest = max(longest, run)
        gap = longest / len(route)
        p_score = 100 * (1 - gap)
        passed = int(seen.sum())
        p_note = (f"longest stretch with nothing to see is "
                  f"{longest * cell_m:.0f} m of a {len(route) * cell_m:.0f} m "
                  f"route")
        p_value = gap * 100
    else:
        p_score, p_value, passed = 0.0, 100.0, 0
        p_note = "no landmarks along the route"
    metrics.append(Metric("pacing", "Pacing", p_value, "% empty", p_score,
                          p_note))

    # -- learnability (the bot) -------------------------------------------------
    learning_curve: List[float] = []
    steps_curve: List[float] = []
    bot_route: List[Tuple[int, int]] = []
    if train_bot and connected:
        successes, evals, learn_eps = [], [], []
        histories = []
        for seed in seeds:
            env_s = env_from_scene(scene, resolution=resolution, seed=seed,
                                   shaping=1.0)
            agent = make_agent("q_learning", env_s.n_states, env_s.n_actions,
                               seed=seed, **AGENT_PARAMS)
            h = train(env_s, agent, episodes=episodes)
            histories.append(h)
            ev = evaluate(env_s, agent, episodes=5)
            evals.append(ev)
            successes.append(ev["success_rate"])
            # First episode after which the bot succeeds 9 times in 10.
            ok = np.convolve(np.asarray(h.successes, float),
                             np.ones(10) / 10, mode="valid")
            hit = np.flatnonzero(ok >= 0.9)
            learn_eps.append(int(hit[0] + 10) if len(hit) else episodes)
            if not bot_route:
                bot_route = greedy_rollout(env_s, agent,
                                           limit=resolution * 6)

        succ = float(np.mean(successes))
        ep = float(np.mean(learn_eps))
        bot_steps = float(np.mean([e["mean_steps"] for e in evals]))
        efficiency = (optimal / bot_steps) if (optimal and bot_steps) else 0
        # Reliable, quick to learn and close to the shortest route.
        l_score = succ * (0.55 + 0.25 * (1 - ep / episodes)
                          + 0.20 * min(1.0, efficiency))
        l_note = (f"bot learned the route in ~{ep:.0f} episodes, "
                  f"{succ:.0f}% success, {bot_steps:.0f} steps vs "
                  f"{optimal} optimal")

        n = min(len(h.successes) for h in histories)
        succ_arr = np.mean([np.asarray(h.successes[:n], float)
                            for h in histories], axis=0)
        step_arr = np.mean([np.asarray(h.steps[:n], float)
                            for h in histories], axis=0)
        w = 10
        learning_curve = list(np.convolve(succ_arr, np.ones(w) / w,
                                          mode="valid"))
        steps_curve = list(np.convolve(step_arr, np.ones(w) / w,
                                       mode="valid"))
        metrics.append(Metric("learnability", "Learnability (bot)", ep,
                              "episodes", l_score, l_note))
    elif train_bot:
        metrics.append(Metric("learnability", "Learnability (bot)",
                              episodes, "episodes", 0.0,
                              "the bot cannot reach the goal"))

    # -- overall -------------------------------------------------------------------
    present = {m.key: m.score for m in metrics if m.key in WEIGHTS}
    total_w = sum(WEIGHTS[k] for k in present)
    flow = sum(WEIGHTS[k] * v for k, v in present.items()) / total_w
    if not connected:
        flow = min(flow, 30.0)

    grid = ["".join("." if walkable[r, c] else "#" for c in range(cols))
            for r in range(rows)]

    return PlaytestReport(
        flow_score=float(flow), connected=bool(connected), metrics=metrics,
        grid=grid, optimal_route=route, bot_route=bot_route,
        landmarks=landmark_cells, start=start, goal=goal,
        learning_curve=learning_curve, steps_curve=steps_curve,
        optimal_steps=optimal, episodes=episodes if train_bot else 0,
        seconds=time.time() - t0, cell_metres=cell_m)
