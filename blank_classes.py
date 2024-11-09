import numpy as np
import random as rn
import pandas as pd
from copy import deepcopy
from collections import deque, namedtuple
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from grid_vis_new import run_grid

# Define the DQN network with increased capacity and proper initialization
class DQN(nn.Module):
    pass

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return rn.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    def __str__(self):
        frame = pd.DataFrame(self.memory, columns=['state', 'action', 'next_state', 'reward'])
        return str(frame)


class DLGrid():
    def __init__(self):
        self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else
                    "mps" if torch.backends.mps.is_available() else
                    "cpu")

    
    def calc_reward(self, choice):
        pass
    
    def next_state(self, choice):
        pass

    def is_done(self):
        pass
    
    def pick_action(self):
        sample = rn.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.EPS_CURRENT = eps_threshold

        if sample > eps_threshold:
            # Take deliberate action
            pass
        else:
            # Take random action
            pass

    def optimize_model(self):
        pass

    def step(self, current_grid):
        pass

    def reset(self):
        pass

    def run_n_episodes(self, n, tick_rate = 0, vis=False):
        for i in range(n):
            if vis:
                run_grid(self.board, update_func=self.step, tick_rate=tick_rate)
            else:
                self.reset()
                while not self.finished:
                    self.step(None)


if __name__ == "__main__":
    pass