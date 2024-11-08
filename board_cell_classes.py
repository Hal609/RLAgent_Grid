import numpy as np
import random as rn
import pandas as pd
from copy import deepcopy
from collections import deque, namedtuple
import math
import torch
import torch.nn as nn
import torch.optim as optim

from grid_vis_new import run_grid

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
    
    def __str__(self):
        frame = pd.DataFrame(self.memory, columns=['state', 'action', 'next_state', 'reward'])
        return str(frame)


class DLGrid():
    def __init__(self, size, treat_ratio=0.7, trap_ratio=0.3):
        self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else
                    "mps" if torch.backends.mps.is_available() else
                    "cpu")
        self.size = size
        self.height = size[0]
        self.width = size[1]
        self.score = 0
        self.finished = False
        self.treat_ratio = treat_ratio
        self.trap_ratio = trap_ratio
        self.num_traps = 0
        self.action_map = {
            0 : torch.tensor((0, 1), device=self.device), 
            1 : torch.tensor((0, -1), device=self.device), 
            2 : torch.tensor((1, 0), device=self.device), 
            3 : torch.tensor((-1, 0), device=self.device)}
        
        self.colour_dict = {"treat": "ba0d8e66", "trap": "110c0baa", "agent": "db2046"}
        self.board_val_dict = {"000000ff": 0, "ffffffff": 1, "ba0d8e66": 2, "110c0baa": 3, "db2046": 4}

        self.init_board()
        self.state = self.board_to_tensor(self.board)
        self.initial_setup = deepcopy(self.board)

        self.BATCH_SIZE = 64  # BATCH_SIZE is the number of transitions sampled from the replay buffer
        self.GAMMA = 0.95  # GAMMA is the discount factor
        self.EPS_START = 0.9  # EPS_START is the starting value of epsilon
        self.EPS_CURRENT = self.EPS_START
        self.EPS_END = 0.05  # EPS_END is the final value of epsilon
        self.EPS_DECAY = 1000  # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
        self.TAU = 0.010  # TAU is the update rate of the target network
        self.LR = 2e-4  # LR is the learning rate of the ``AdamW`` optimizer


        n_actions = 4
        # n_observations = self.height * self.width
        n_observations = 2

        self.policy_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(10000)

        self.steps_done = 0

    def init_board_random(self):
        self.board = np.zeros(self.size, dtype='U8')
        for y, row in enumerate(self.board):
            for x in range(len(row)):
                if y == 0 or y == len(self.board) - 1 or x == 0 or x == len(row) - 1:
                     self.board[y][x] = "ffffffff"
                else:
                    self.board[y][x] = "000000ff"
        treats = 0
        num_treats = self.height * self.width * self.treat_ratio
        while treats < num_treats:
            x = rn.randint(1, self.width - 2)
            y = rn.randint(1, self.height - 2)
            if self.board[y][x] != 1:
                self.board[y][x] = self.colour_dict["treat"]
                treats += 1
        traps = 0
        self.num_traps = (self.height * self.width - num_treats) * self.trap_ratio
        while traps < self.num_traps:
            x = rn.randint(1, self.width - 2)
            y = rn.randint(1, self.height - 2)
            if self.board[y][x] != 1:
                self.board[y][x] = self.colour_dict["trap"]
                traps += 1

        self.agent_pos = torch.tensor((self.height//2, self.width//2), dtype=torch.float32, device=self.device)
        self.board[int(self.agent_pos[0])][int(self.agent_pos[1])] = self.colour_dict["agent"]

    def init_board(self):
        self.board = np.zeros(self.size, dtype='U8')
        for y, row in enumerate(self.board):
            for x in range(len(row)):
                if y == 0 or y == len(self.board) - 1 or x == 0 or x == len(row) - 1:
                    self.board[y][x] = "ffffffff"
                else:
                    self.board[y][x] = "000000ff"
        treats = 0
        x, y = 0, 0
        num_treats = self.height * self.width * self.treat_ratio
        while treats < num_treats:
            x = int(1 + ((x + 1.5) % (self.width - 2)))
            y = int(1 + ((y + 2) % (self.height - 2)))
            if self.board[y][x] != 1:
                self.board[y][x] = self.colour_dict["treat"]
                treats += 1
        traps = 0
        self.num_traps = (self.height * self.width - num_treats) * self.trap_ratio
        while traps < self.num_traps:
            x = int(1 + ((x + 2) % (self.width - 2)))
            y = int(1 + ((y + 3) % (self.height - 2)))
            if self.board[y][x] != 1:
                self.board[y][x] = self.colour_dict["trap"]
                traps += 1

        self.agent_pos = torch.tensor((self.height//2, self.width//2), dtype=torch.float32, device=self.device)
        self.board[int(self.agent_pos[0])][int(self.agent_pos[1])] = self.colour_dict["agent"]
    
    def is_inbounds(self, pos):
        if 0 in pos or self.width-1 <= pos[1] or self.height-1 <= pos[0]:
            return False
        return True
    
    def calc_reward(self, choice):
        new_pos = self.agent_pos + self.action_map[int(choice)]
        next_val = self.board[int(new_pos[0])][int(new_pos[1])]
        if next_val == self.colour_dict["treat"]:
            return 5.0
        elif next_val == self.colour_dict["trap"]:
            return -10.0
        return -1.0
    
    def next_state(self, choice):
        next_pos = self.agent_pos + self.action_map[int(choice)]
        if not self.is_inbounds(next_pos): return self.board
        next = deepcopy(self.board)
        next[int(self.agent_pos[0])][int(self.agent_pos[1])] = "000000ff"
        self.agent_pos = next_pos
        next[int(next_pos[0])][int(next_pos[1])] = self.colour_dict["agent"]

        return next

    def board_to_tensor(self, board):
        tensor_board = torch.zeros((self.board.shape)).to(self.device)
        for y, row in enumerate(self.board):
            for x, cell in enumerate(row):
                tensor_board[y][x] = torch.tensor(self.board_val_dict[cell.lower()], device=self.device)
        return tensor_board.flatten().unsqueeze(0)
        
    def is_done(self):
        if self.score < -50: return True
        if int(torch.count_nonzero(self.board_to_tensor(self.board))) == self.num_traps + 1: return True
        if self.initial_setup[int(self.agent_pos[0])][int(self.agent_pos[1])] == self.colour_dict["trap"]: return True
        return False
    
    def pick_action(self):
        sample = rn.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.EPS_CURRENT = eps_threshold
        self.steps_done += 1
        if sample > eps_threshold:
            # Take deliberate action
            with torch.no_grad():
                return torch.argmax(self.policy_net(self.agent_pos.unsqueeze(0))).unsqueeze(0).unsqueeze(0)
                # return torch.argmax(self.policy_net(self.state)).unsqueeze(0).unsqueeze(0)
        else:
            # Take random action
            return torch.tensor([[rn.randint(0, 3)]], device=self.device)

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        # print("state batch shape", state_batch.shape)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch


        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def step(self, current_grid):
        if self.finished: self.reset()

        start_pos = self.agent_pos.unsqueeze(0)
        action = self.pick_action()
        reward = self.calc_reward(action)
        reward = torch.tensor([reward], device=self.device)
        self.score += reward
        next_board = self.next_state(action)
        self.finished = self.is_done()

        if self.finished:
            next_state = None
        else:
            next_state = self.board_to_tensor(next_board)

        self.state = self.board_to_tensor(self.board)
        self.memory.push(start_pos, action, self.agent_pos.unsqueeze(0), reward)
        self.board = next_board

        # Perform one step of the optimization (on the policy network)
        self.optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
        self.target_net.load_state_dict(target_net_state_dict)

        return next_board

    def run_n_episodes(self, n, tick_rate = 0, vis=False):
        for i in range(n):
            if vis:
                run_grid(self.board, update_func=self.step, tick_rate=tick_rate)
            else:
                self.reset()
                while not self.finished:
                    self.step(None)

    def reset(self):
        # print(int(self.score))
        self.score = 0
        self.finished = False
        self.init_board()
        self.state = self.board_to_tensor(self.board)
        self.initial_setup = deepcopy(self.board)

if __name__ == "__main__":
    grid =  DLGrid((18, 18))
    grid.run_n_episodes(n=500, vis=False)