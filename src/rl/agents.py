"""
Agents for the scene-navigation MDP.

Three of them, deliberately:

  RandomAgent     the floor. Any learning method has to beat this, and it also
                  gives an honest sense of how hard a given layout is -- a
                  scene a random walk solves is not measuring anything.

  QLearningAgent  off-policy temporal-difference control. Learns the value of
                  the greedy action regardless of what it actually did, so it
                  can explore aggressively while still converging on the
                  optimal policy.

  SarsaAgent      the on-policy counterpart. Learns the value of what it
                  actually does, including the exploratory mistakes, which
                  makes it more conservative near hazards.

Both learners are tabular. The state space here is a grid of at most a few
thousand cells, which a table covers exactly -- no approximation error, no
network to tune, and the learned values can be read directly. Function
approximation would add machinery without adding anything to measure.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class BaseAgent:
    name = "base"

    def __init__(self, n_states: int, n_actions: int, seed: int = 0):
        self.n_states = n_states
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def act(self, state: int, greedy: bool = False) -> int:
        raise NotImplementedError

    def learn(self, state, action, reward, next_state, next_action,
              done) -> None:
        pass

    def end_episode(self) -> None:
        pass


class RandomAgent(BaseAgent):
    """Uniform random action selection. Learns nothing, by design."""

    name = "random"

    def act(self, state: int, greedy: bool = False) -> int:
        return int(self.rng.integers(0, self.n_actions))


class TabularTDAgent(BaseAgent):
    """Shared machinery for the two temporal-difference learners."""

    def __init__(self, n_states: int, n_actions: int, alpha: float = 0.1,
                 gamma: float = 0.95, epsilon: float = 1.0,
                 epsilon_min: float = 0.05, epsilon_decay: float = 0.995,
                 seed: int = 0, optimistic: float = 0.0):
        super().__init__(n_states, n_actions, seed)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Optimistic initialisation encourages early exploration on its own:
        # every unvisited action looks better than it is, so the agent tries
        # it once. Zero-initialised here by default so that exploration is
        # attributable to epsilon alone and the comparison stays clean.
        self.q = np.full((n_states, n_actions), optimistic, dtype=np.float64)

    def act(self, state: int, greedy: bool = False) -> int:
        """Epsilon-greedy action selection.

        Ties are broken randomly rather than by argmax's first-index rule.
        With a zero-initialised table every action in an unvisited state ties,
        and always taking the first one makes the agent walk north until it
        hits something.
        """
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))

        values = self.q[state]
        best = np.flatnonzero(values == values.max())
        return int(self.rng.choice(best))

    def end_episode(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    @property
    def policy(self) -> np.ndarray:
        return self.q.argmax(axis=1)

    def value_function(self) -> np.ndarray:
        return self.q.max(axis=1)


class QLearningAgent(TabularTDAgent):
    """Off-policy TD control.

        Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

    The max over next actions is what makes it off-policy: the update assumes
    the best action will be taken next, whatever the agent actually does. It
    therefore learns the optimal policy while still behaving exploratively.
    """

    name = "q_learning"

    def learn(self, state, action, reward, next_state, next_action,
              done) -> None:
        target = reward
        if not done:
            target += self.gamma * self.q[next_state].max()
        self.q[state, action] += self.alpha * (target - self.q[state, action])


class SarsaAgent(TabularTDAgent):
    """On-policy TD control.

        Q(s,a) <- Q(s,a) + alpha * [ r + gamma * Q(s',a') - Q(s,a) ]

    Uses the action actually taken next rather than the best one, so the
    exploratory moves are part of what is learned. The result is a policy that
    accounts for its own randomness -- it keeps further from anything costly,
    because it knows it sometimes slips.
    """

    name = "sarsa"

    def learn(self, state, action, reward, next_state, next_action,
              done) -> None:
        target = reward
        if not done and next_action is not None:
            target += self.gamma * self.q[next_state, next_action]
        self.q[state, action] += self.alpha * (target - self.q[state, action])


def make_agent(kind: str, n_states: int, n_actions: int,
               seed: int = 0, **kwargs) -> BaseAgent:
    if kind == "random":
        return RandomAgent(n_states, n_actions, seed=seed)
    if kind == "q_learning":
        return QLearningAgent(n_states, n_actions, seed=seed, **kwargs)
    if kind == "sarsa":
        return SarsaAgent(n_states, n_actions, seed=seed, **kwargs)
    raise ValueError(f"unknown agent {kind!r}")
