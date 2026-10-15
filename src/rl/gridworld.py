"""
A navigation MDP built from a generated scene.

The environment generator produces terrain, water and a few thousand placed
objects. This turns one of those scenes into a discrete Markov Decision
Process an agent can be trained on, so that how well an agent learns to cross
a layout becomes a measurement of the layout itself.

The point is not navigation for its own sake. A scene a policy learns to cross
quickly and reliably is well connected; one where it thrashes has chokepoints,
dead ends or an unreachable objective. Training an agent is therefore a way of
asking whether a generated place is actually good to move through -- which is
something no purely geometric check can answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# The four cardinal moves. Diagonals are excluded so that step count is a
# clean proxy for path length and the transition model stays simple.
ACTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
ACTION_NAMES = ["north", "east", "south", "west"]

# Reward structure.
#
# A small negative reward per step is what makes the agent prefer short
# routes: with no step cost every path that eventually reaches the goal is
# worth the same, and the agent has no reason to learn a good one.
STEP_COST = -1.0
GOAL_REWARD = 100.0
WALL_PENALTY = -5.0      # bumping into something wastes a step and hurts more


@dataclass
class EpisodeResult:
    total_reward: float
    steps: int
    reached_goal: bool
    visited: int


class SceneNavigationEnv:
    """A discrete navigation environment derived from a generated scene.

    The state is the agent's cell. The action space is the four cardinal
    moves. An episode ends when the agent reaches the goal cell or runs out of
    steps, which makes this an episodic, finite MDP with deterministic
    transitions -- except that a move into an occupied cell leaves the agent
    where it was, which is a perfectly ordinary part of the dynamics rather
    than a special case.
    """

    def __init__(self, walkable: np.ndarray, start: Tuple[int, int],
                 goal: Tuple[int, int], max_steps: int = 600,
                 slip: float = 0.0, seed: int = 0, shaping: float = 0.0):
        self.walkable = walkable
        self.rows, self.cols = walkable.shape
        self.start = start
        self.goal = goal
        self.max_steps = max_steps
        # Probability that a move goes somewhere other than intended. Zero
        # gives a deterministic MDP; a small value makes it stochastic, which
        # is closer to a character controller on uneven ground.
        self.slip = slip
        self.rng = np.random.default_rng(seed)

        self.n_states = self.rows * self.cols
        self.n_actions = len(ACTIONS)

        self.position = start
        self.steps = 0
        self._visited = set()

        # Potential-based reward shaping (Ng, Harada and Russell, 1999).
        #
        # With one goal cell in sixteen hundred, a random walk reaches it in
        # about one episode in a hundred, so an agent exploring randomly sees
        # almost no reward and has nothing to learn from. Shaping adds
        # gamma*phi(s') - phi(s) to every transition, where phi is the
        # negated distance to the goal. Because the added term is the
        # difference of a potential, the optimal policy is provably unchanged
        # -- it makes the signal dense without changing what is being solved.
        self.shaping = shaping
        self._potential = self._distance_field() if shaping else None

    # -- state helpers ----------------------------------------------------
    def state_index(self, position: Optional[Tuple[int, int]] = None) -> int:
        r, c = position if position is not None else self.position
        return r * self.cols + c

    def index_to_cell(self, index: int) -> Tuple[int, int]:
        return divmod(index, self.cols)

    def is_walkable(self, r: int, c: int) -> bool:
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return False
        return bool(self.walkable[r, c])

    # -- the MDP ----------------------------------------------------------
    def reset(self) -> int:
        self.position = self.start
        self.steps = 0
        self._visited = {self.start}
        return self.state_index()

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """Take one action. Returns (state, reward, done, info)."""
        self.steps += 1

        if self.slip > 0 and self.rng.random() < self.slip:
            # Slip to a neighbouring action rather than a random one: a
            # character that means to go north and drifts is far likelier to
            # end up north-east than south.
            action = (action + int(self.rng.choice([-1, 1]))) % self.n_actions

        dc, dr = ACTIONS[action]
        r, c = self.position
        nr, nc = r + dr, c + dc

        if self.is_walkable(nr, nc):
            self.position = (nr, nc)
            reward = STEP_COST
        else:
            reward = STEP_COST + WALL_PENALTY

        if self.shaping:
            # gamma is folded into the caller's discount; using 1.0 here keeps
            # the shaping term a pure difference of potentials, which is the
            # form that guarantees the optimal policy is preserved.
            reward += self.shaping * (self._phi(self.position) - self._phi((r, c)))

        self._visited.add(self.position)

        done = False
        if self.position == self.goal:
            reward += GOAL_REWARD
            done = True
        elif self.steps >= self.max_steps:
            done = True

        return self.state_index(), reward, done, {
            "reached_goal": self.position == self.goal,
            "visited": len(self._visited),
        }

    def _distance_field(self) -> np.ndarray:
        """BFS distance from every cell to the goal, in steps."""
        from collections import deque

        dist = np.full((self.rows, self.cols), np.inf)
        if not self.is_walkable(*self.goal):
            return dist

        dist[self.goal] = 0.0
        queue = deque([self.goal])
        while queue:
            r, c = queue.popleft()
            for dc, dr in ACTIONS:
                nr, nc = r + dr, c + dc
                if self.is_walkable(nr, nc) and np.isinf(dist[nr, nc]):
                    dist[nr, nc] = dist[r, c] + 1
                    queue.append((nr, nc))
        return dist

    def _phi(self, cell: Tuple[int, int]) -> float:
        if self._potential is None:
            return 0.0
        d = self._potential[cell]
        if np.isinf(d):
            return -float(self.rows * self.cols)
        return -float(d)

    # -- reference solution ------------------------------------------------
    def shortest_path_length(self) -> Optional[int]:
        """Breadth-first distance from start to goal.

        Not used for training. It is the optimum an agent is measured
        against, and it also tells us whether the goal is reachable at all --
        a scene where it isn't should be rejected before any training runs.
        """
        from collections import deque

        if not self.is_walkable(*self.start) or not self.is_walkable(*self.goal):
            return None

        seen = {self.start}
        queue = deque([(self.start, 0)])
        while queue:
            (r, c), d = queue.popleft()
            if (r, c) == self.goal:
                return d
            for dc, dr in ACTIONS:
                nxt = (r + dr, c + dc)
                if nxt not in seen and self.is_walkable(*nxt):
                    seen.add(nxt)
                    queue.append((nxt, d + 1))
        return None

    def reachable_fraction(self) -> float:
        """Share of walkable cells the agent could ever get to.

        A layout can be perfectly valid geometrically and still be split in
        two by a river or a wall of trees.
        """
        from collections import deque

        total = int(self.walkable.sum())
        if total == 0:
            return 0.0

        seen = {self.start}
        queue = deque([self.start])
        while queue:
            r, c = queue.popleft()
            for dc, dr in ACTIONS:
                nxt = (r + dr, c + dc)
                if nxt not in seen and self.is_walkable(*nxt):
                    seen.add(nxt)
                    queue.append(nxt)
        return len(seen) / total


# ---------------------------------------------------------------------------
# Building an environment from a generated scene
# ---------------------------------------------------------------------------

def walkability_from_scene(scene, resolution: int = 48,
                           max_slope: float = 0.45) -> np.ndarray:
    """Mark which cells of a scene a character could stand in.

    A cell is blocked by water, by ground too steep to walk, or by an object
    whose footprint covers it. Vegetation is treated as passable: a player
    walks through undergrowth, and treating every grass tuft as a wall would
    make the map almost entirely solid.
    """
    terrain = scene.terrain
    size = terrain.size
    cell = size / resolution

    walkable = np.ones((resolution, resolution), dtype=bool)

    # Terrain: water and steep ground.
    for r in range(resolution):
        for c in range(resolution):
            x = (c + 0.5) * cell
            z = (r + 0.5) * cell
            if terrain.on_bridge(x, z, margin=cell * 0.5):
                # The deck spans the water and the steep banks either side.
                continue
            if terrain.is_water(x, z, margin=cell * 0.4):
                walkable[r, c] = False
            elif terrain.slope_at(x, z) > max_slope:
                walkable[r, c] = False

    # Objects with a real footprint. Small scatter is ignored: a pebble is
    # not an obstacle, and marking it as one would fragment the map.
    solid = {"building", "structure", "rock"}
    for inst in scene.instances:
        if inst.category not in solid:
            continue
        if getattr(inst, "role", "") == "bridge":
            continue
        if inst.radius < cell * 0.35:
            continue

        x, _y, z = inst.position
        reach = int(np.ceil(inst.radius / cell))
        cc, rr = int(x / cell), int(z / cell)
        for r in range(max(0, rr - reach), min(resolution, rr + reach + 1)):
            for c in range(max(0, cc - reach), min(resolution, cc + reach + 1)):
                px, pz = (c + 0.5) * cell, (r + 0.5) * cell
                if (px - x) ** 2 + (pz - z) ** 2 <= inst.radius ** 2:
                    walkable[r, c] = False

    return walkable


def env_from_scene(scene, resolution: int = 48, seed: int = 0,
                   slip: float = 0.0, shaping: float = 0.0,
                   max_steps: Optional[int] = None) -> SceneNavigationEnv:
    """Build a navigation MDP from a generated scene.

    Start and goal are placed at opposite ends of the scene's road, so the
    task is to cross the place the way a player would, rather than to solve
    an arbitrary maze.
    """
    walkable = walkability_from_scene(scene, resolution=resolution)
    size = scene.terrain.size
    cell = size / resolution

    def nearest_walkable(x: float, z: float) -> Tuple[int, int]:
        cc, rr = int(np.clip(x / cell, 0, resolution - 1)), \
            int(np.clip(z / cell, 0, resolution - 1))
        if walkable[rr, cc]:
            return rr, cc
        for radius in range(1, resolution):
            for r in range(max(0, rr - radius), min(resolution, rr + radius + 1)):
                for c in range(max(0, cc - radius),
                               min(resolution, cc + radius + 1)):
                    if walkable[r, c]:
                        return r, c
        return rr, cc

    if scene.path:
        start = nearest_walkable(*scene.path[0])
        goal = nearest_walkable(*scene.path[-1])
    else:
        start = nearest_walkable(cell, cell)
        goal = nearest_walkable(size - cell, size - cell)

    if max_steps is None:
        # Generous enough that a poor policy still terminates, tight enough
        # that wandering is penalised.
        max_steps = resolution * 12

    return SceneNavigationEnv(walkable, start, goal, max_steps=max_steps,
                              slip=slip, seed=seed, shaping=shaping)


def render_ascii(env: SceneNavigationEnv,
                 path: Optional[List[Tuple[int, int]]] = None) -> str:
    """A text picture of the map, for reports and for sanity-checking."""
    marks = {env.start: "S", env.goal: "G"}
    if path:
        for cell in path:
            marks.setdefault(cell, "·")

    lines = []
    for r in range(env.rows):
        row = []
        for c in range(env.cols):
            if (r, c) in marks:
                row.append(marks[(r, c)])
            else:
                row.append(" " if env.walkable[r, c] else "#")
        lines.append("".join(row))
    return "\n".join(lines)
