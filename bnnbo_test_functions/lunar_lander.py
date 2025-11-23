__credits__ = ["Andrea PIERRÉ"]

import math
import warnings
from multiprocessing import Pool
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from botorch.test_functions.base import BaseTestProblem
from gymnasium.envs.box2d.lunar_lander import LunarLander

if TYPE_CHECKING:
    import pygame



def heuristic_Controller(s, w):
    angle_targ = s[0] * w[0] + s[2] * w[1]
    if angle_targ > w[2]:
        angle_targ = w[2]
    if angle_targ < -w[2]:
        angle_targ = -w[2]
    hover_targ = w[3] * np.abs(s[0])

    angle_todo = (angle_targ - s[4]) * w[4] - (s[5]) * w[5]
    hover_todo = (hover_targ - s[1]) * w[6] - (s[3]) * w[7]

    if s[6] or s[7]:
        angle_todo = w[8]
        hover_todo = -(s[3]) * w[9]

    a = 0
    if hover_todo > np.abs(angle_todo) and hover_todo > w[10]:
        a = 2
    elif angle_todo < -w[11]:
        a = 3
    elif angle_todo > +w[11]:
        a = 1
    return a


def simulate_lunar_rover(p):
    x, seed = p
    # print(x)
    env = LunarLander()
    total_reward = 0
    steps = 0
    s, info = env.reset(seed=seed)
    while True:
        a = heuristic_Controller(s, x)
        s, r, terminated, truncated, info = env.step(a)
        total_reward += r

        # if steps % 20 == 0 or terminated or truncated:
        #     print("observations:", " ".join([f"{x:+0.2f}" for x in s]))
        #     print(f"step {steps} total_reward {total_reward:+0.2f}")
        steps += 1
        if terminated or truncated:
            break
        if steps > 5000:
            # total_reward -= 100
            break
    if steps > 500:
        print(steps, total_reward)
    return total_reward


class LunarLanderProblem(BaseTestProblem):
    dim: int = 12
    num_objectives = 1

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = False
    ) -> None:
        self._bounds = np.repeat([[0], [2]], 12, axis=1).T
        super().__init__(
            noise_std=noise_std,
            negate=negate,
        )

    def get_reward(self, x):
        n = 50
        reward = 0.0
        # x = np.array([0.5, 1.0, 0.4, 0.55, 0.5, 1.0, 0.5, 0.5, 0, 0.5, 0.05, 0.05])
        with Pool() as p:
            params = [(x, i) for i in range(n)]
            reward = torch.tensor(p.map(simulate_lunar_rover, params)).sum()
        return reward / n

    def _evaluate_true(self, X):
        n_envs = 50
        params = []
        for x in X:
            for env_seed in range(n_envs):
                params.append((x.cpu(), env_seed))
        with Pool(16) as p:
            reward = p.map(simulate_lunar_rover, params)
        # convert to len(X) x n_envs
        reward = torch.tensor(reward).view(-1, n_envs) 
        # average over envs
        reward = reward.mean(1) 
        return reward.to(X)

