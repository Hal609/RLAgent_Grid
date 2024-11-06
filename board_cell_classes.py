import numpy as np
import random as rn
from time import sleep
from copy import deepcopy
from collections import deque, namedtuple
import math
import torch
import torch.nn as nn
import torch.optim as optim

from grid_vis_new import run_grid
    
class Grid():
    def __init__(self, size, treat_ratio = 0.7, trap_ratio = 0.3):
        self.size = size
        self.height = size[0]
        self.width = size[1]
        self.score = 0
        self.finished = False
        self.treat_ratio = treat_ratio
        self.trap_ratio = trap_ratio
        self.num_traps = 0

        self.colour_dict = {"treat": "ba0d8e66", "trap": "110c0baa", "agent": "db2046"}

        self.memory = deque(maxlen=10000)

        self.init_board()
        self.initial_setup = deepcopy(self.board)
        

    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    def init_board(self):
        self.board = np.zeros(self.size, dtype='U8')
        for y, row in enumerate(self.board):
            for x in range(len(row)):
                self.board[y][x] = "000000FF"
        treats = 0
        num_treats = self.height * self.width * self.treat_ratio
        while treats < num_treats:
            x = rn.randint(0, self.width - 1)
            y = rn.randint(0, self.height - 1)
            if self.board[y][x] != 1:
                self.board[y][x] = self.colour_dict["treat"]
                treats += 1
        traps = 0
        self.num_traps = (self.height * self.width - num_treats) * self.trap_ratio
        while traps < self.num_traps:
            x = rn.randint(0, self.width - 1)
            y = rn.randint(0, self.height - 1)
            if self.board[y][x] != 1:
                self.board[y][x] = self.colour_dict["trap"]
                traps += 1

        self.agent_pos = torch.tensor((self.height//2, self.width//2))
        self.board[self.agent_pos[0]][self.agent_pos[1]] = self.colour_dict["agent"]

    def is_inbounds(self, pos):
        if -1 in pos or self.width <= pos[0] or self.height <= pos[1]:
            return False
        return True
    
    def get_next(self):
        directions = torch.tensor(((0, 1), (0, -1), (1, 0), (-1, 0)))
        nexts = []
        for entry in (self.agent_pos + directions):
            if self.is_inbounds(entry):
                nexts.append(entry)
        return torch.stack(nexts)
    
    def calc_reward(self, action):
        next_val = self.board[action[0]][action[1]]
        if next_val == self.colour_dict["treat"]:
            return 1
        elif next_val == self.colour_dict["trap"]:
            return -3
        return 0

    def is_done(self):
        # if int(torch.count_nonzero(self.board)) == self.num_traps + 1: return True
        if self.initial_setup[self.agent_pos[0]][self.agent_pos[1]] == self.colour_dict["trap"]: return True
        return False
    
    def next_state(self, next_pos):
        next = deepcopy(self.board)
        next[self.agent_pos[0]][self.agent_pos[1]] = "000000"
        self.agent_pos = next_pos
        next[next_pos[0]][next_pos[1]] = self.colour_dict["agent"]

        return next

    def pick_action(self):
        return rn.choice(self.get_next())
    
    def step(self):
        action = self.pick_action()
        reward = self.calc_reward(action)
        self.score += reward
        next_state = self.next_state(action)
        self.remember(self.board, action, reward, next_state, self.finished)
        self.board = next_state
        self.finished = self.is_done()

# Define the Q-Network
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_observations, 128) 
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)  # Output is the Q-value for each action (up, down, left, right)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

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
    

class DLGrid(Grid):
    def __init__(self, size, treat_ratio=0.7, trap_ratio=0.3):
        super().__init__(size, treat_ratio, trap_ratio)

        self.BATCH_SIZE = 128  # BATCH_SIZE is the number of transitions sampled from the replay buffer
        self.GAMMA = 0.99  # GAMMA is the discount factor as mentioned in the previous section
        self.EPS_START = 0.9  # EPS_START is the starting value of epsilon
        self.EPS_END = 0.05  # EPS_END is the final value of epsilon
        self.EPS_DECAY = 1000  # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
        self.TAU = 0.005  # TAU is the update rate of the target network
        self.LR = 1e-4  # LR is the learning rate of the ``AdamW`` optimizer

        self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else
                    "mps" if torch.backends.mps.is_available() else
                    "cpu")
        
        n_actions = 4
        n_observations = self.height * self.width

        self.policy_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)

        self.steps_done = 0
    
    def pick_action(self):
        sample = rn.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            # Take deliberate action
            return rn.choice(self.get_next())
        else:
            # Take random action
            return rn.choice(self.get_next())

    def replay(self):
        rn.sample(self.memory, self.BATCH_SIZE)

    def step(self, current_grid):
        if self.finished: return None
        action = self.pick_action()
        reward = self.calc_reward(action)
        self.score += reward
        next_state = self.next_state(action)
        # state, action, reward, next state, is terminal?
        self.remember(self.board, action, reward, next_state, self.finished)
        self.board = next_state
        self.finished = self.is_done()

        return next_state

    def run_episode(self):
        while not self.finished:
            print(self.board)
            sleep(1)
            self.step()
        
        print(self.score)

    def run_n_episodes(self, n):
        for i in range(n):
            self.run_episode()
            self.reset()

    def reset(self):
        self.score = 0
        self.board = self.initial_setup
        self.finished = False
        self.init_board()
        self.initial_setup = deepcopy(self.board)


if __name__ == "__main__":
    grid =  DLGrid((4, 6))
    run_grid(grid.board, update_func=grid.step, tick_rate=1)