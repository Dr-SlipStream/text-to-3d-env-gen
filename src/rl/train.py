"""
Training and evaluation for the scene-navigation agents.

Everything here reports across several random seeds. A single training run of
a tabular method on a stochastic environment can look like anything; the
spread across seeds is the part worth reading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .agents import BaseAgent, make_agent
from .gridworld import SceneNavigationEnv


@dataclass
class RunHistory:
    """Per-episode record of one training run."""

    returns: List[float] = field(default_factory=list)
    steps: List[int] = field(default_factory=list)
    successes: List[bool] = field(default_factory=list)
    epsilons: List[float] = field(default_factory=list)

    def moving_average(self, window: int = 50) -> np.ndarray:
        if not self.returns:
            return np.array([])
        r = np.asarray(self.returns, dtype=float)
        window = min(window, len(r))
        kernel = np.ones(window) / window
        return np.convolve(r, kernel, mode="valid")

    def success_rate(self, last: int = 100) -> float:
        if not self.successes:
            return 0.0
        tail = self.successes[-last:]
        return 100.0 * sum(tail) / len(tail)

    def episodes_to_threshold(self, threshold: float,
                              window: int = 50) -> Optional[int]:
        """First episode at which the moving average crosses `threshold`.

        Reported as a convergence speed. None means it never got there, which
        is a result in itself rather than a missing number.
        """
        avg = self.moving_average(window)
        hits = np.flatnonzero(avg >= threshold)
        return int(hits[0] + window) if len(hits) else None


def train(env: SceneNavigationEnv, agent: BaseAgent, episodes: int = 800
          ) -> RunHistory:
    """Run one training session and record what happened."""
    history = RunHistory()

    for _ in range(episodes):
        state = env.reset()
        action = agent.act(state)

        total, steps, reached = 0.0, 0, False
        done = False
        while not done:
            next_state, reward, done, info = env.step(action)
            next_action = None if done else agent.act(next_state)

            agent.learn(state, action, reward, next_state, next_action, done)

            state, action = next_state, next_action
            total += reward
            steps += 1
            reached = info["reached_goal"]

            if action is None and not done:
                action = agent.act(state)

        agent.end_episode()
        history.returns.append(total)
        history.steps.append(steps)
        history.successes.append(reached)
        history.epsilons.append(getattr(agent, "epsilon", 0.0))

    return history


def evaluate(env: SceneNavigationEnv, agent: BaseAgent,
             episodes: int = 100) -> Dict[str, float]:
    """Measure a trained agent with exploration switched off.

    Training returns are depressed by the exploration the agent is still
    doing; this is what the learned policy is actually worth.
    """
    returns, steps, successes = [], [], []

    for _ in range(episodes):
        state = env.reset()
        total, count, reached = 0.0, 0, False
        done = False
        while not done:
            action = agent.act(state, greedy=True)
            state, reward, done, info = env.step(action)
            total += reward
            count += 1
            reached = info["reached_goal"]
        returns.append(total)
        steps.append(count)
        successes.append(reached)

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_steps": float(np.mean(steps)),
        "success_rate": 100.0 * sum(successes) / len(successes),
    }


def run_experiment(env_factory, agent_kind: str, seeds: List[int],
                   episodes: int = 800, **agent_kwargs
                   ) -> Tuple[List[RunHistory], List[Dict[str, float]]]:
    """Train one method across several seeds.

    The environment is rebuilt per seed as well as the agent, so the reported
    spread covers both the randomness in learning and any randomness in the
    environment itself.
    """
    histories, evaluations = [], []

    for seed in seeds:
        env = env_factory(seed)
        agent = make_agent(agent_kind, env.n_states, env.n_actions,
                           seed=seed, **agent_kwargs)
        histories.append(train(env, agent, episodes=episodes))
        evaluations.append(evaluate(env, agent))

    return histories, evaluations


def aggregate(histories: List[RunHistory], window: int = 50
              ) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of the smoothed return, across seeds."""
    curves = [h.moving_average(window) for h in histories]
    if not curves:
        return np.array([]), np.array([])
    length = min(len(c) for c in curves)
    stacked = np.vstack([c[:length] for c in curves])
    return stacked.mean(axis=0), stacked.std(axis=0)


def summarise(evaluations: List[Dict[str, float]]) -> Dict[str, float]:
    """Collapse per-seed evaluations into means and spreads."""
    if not evaluations:
        return {}
    keys = evaluations[0].keys()
    out = {}
    for key in keys:
        values = [e[key] for e in evaluations]
        out[key] = float(np.mean(values))
        out[key + "_std"] = float(np.std(values))
    return out


def greedy_rollout(env: SceneNavigationEnv, agent: BaseAgent,
                   limit: int = 400) -> List[Tuple[int, int]]:
    """The route a trained agent actually takes, for plotting."""
    env.reset()
    path = [env.position]
    for _ in range(limit):
        action = agent.act(env.state_index(), greedy=True)
        _, _, done, _ = env.step(action)
        path.append(env.position)
        if done:
            break
    return path
