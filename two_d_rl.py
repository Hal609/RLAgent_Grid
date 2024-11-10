import numpy as np
import random as rn
import pandas as pd
from time import sleep
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
    def __init__(self, n_inputs):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 256)  # Now input size is 2 (x and y coordinates)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 4)  # Output size is 4 (up, down, left, right)
        self.apply(self.init_weights)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

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
        frame = pd.DataFrame(self.memory, columns=['state', 'action', 'next_state', 'reward', 'done'])
        return str(frame)


class DLGrid():
    def __init__(self, size=11):
        self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else
                    # "mps" if torch.backends.mps.is_available() else
                    "cpu")
        
        num_inputs = size ** 2
        
        # Initialize the policy and target networks
        self.policy_net = DQN(num_inputs).to(self.device)
        self.target_net = DQN(num_inputs).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set target network to evaluation mode

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)

        # Parameters
        self.gamma = 0.95  # Discount factor

        # Epsilon parameters for decay
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.9991
        self.epsilon = self.epsilon_start

        self.target_update = 10  # How often to update the target network
        self.max_steps_per_episode = 50
        self.episode_steps = 0
        self.visualise_every_n = 100
        self.grid_size = size # Grid size, defaults to (11x11)
        self.hazards = [(x, y) for x in range(1, size-1) for y in range(1, size-1) if rn.random() < 0.03]
        self.goal_states = [(x, y) for x in range(1, size-1) for y in range(1, size-1) if rn.random() < 0.03]  # Target position the agent should reach
        self.remaining_goals = deepcopy(self.goal_states)
        self.done = False

        # Replay buffer
        self.replay_buffer = ReplayMemory(10000)
        self.batch_size = 64

        # Action space
        self.actions = ['up', 'down', 'left', 'right']

        # State
        # self.position = (rn.randint(1, self.grid_size - 2), rn.randint(1, self.grid_size - 2))  # Random start position
        self.position = (size//2, size//2)

        # Grid
        self.colour_map = {"empty": "000000ff", "wall": "ffffffff", "goal": "aa00ff44", "agent": "ff3300ff", "hazard": "333333aa"}
        self.colour_to_float = {"000000ff": 0.0, "ffffffff": 0.0, "aa00ff44": 0.5, "ff3300ff": 1.0, "333333aa": 0.2}
        self.grid = self.init_grid()

    def init_grid(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype='U8')
        for y, row in enumerate(grid):
            for x, cell in enumerate(row):
                grid[y][x] = self.colour_map["empty"]
                if y == 0 or y == self.grid_size - 1 or x == 0 or x == self.grid_size - 1:
                    grid[y][x] = self.colour_map["wall"]
                else:
                    grid[y][x] == self.colour_map["empty"]
        for goal in self.goal_states:
            grid[goal[0]][goal[1]] = self.colour_map["goal"]
        for hazard in self.hazards:
            grid[hazard[0]][hazard[1]] = self.colour_map["hazard"]

        return grid
    
    def grid_to_state(self, grid):
        grid = list(grid.flatten())
        for i in range(len(grid)):
            grid[i] = self.colour_to_float[grid[i]]

        return tuple(grid)

    def calc_reward(self, state):
        reward = 0
        x, y = state
        # distance_to_goal = abs(x - self.goal_state[0]) + abs(y - self.goal_state[1])  # Manhattan distance
        if state in self.remaining_goals:
            self.remaining_goals.remove(state)
            print("Goals", self.goal_states)
            print("Remaining goals", self.remaining_goals)
            print("position", state)
            reward += 3.0  # Positive reward for reaching the goal
        if state in self.hazards:
            reward -= 1.0
        reward +=  -0.1
        return reward / (2 * (self.grid_size - 1))  # Normalize reward between -1 and 0
    
    def next_state(self, action):
        x, y = self.position
        if action == 0:  # Up
            y = y - 1
        elif action == 1:  # Down
            y = y + 1
        elif action == 2:  # Left
            x = x - 1
        elif action == 3:  # Right
            x = x + 1
        # Keep within grid boundaries
        x = max(1, min(self.grid_size - 2, x))
        y = max(1, min(self.grid_size - 2, y))

        return (x, y)

    def is_done(self, state):
        if self.episode_steps >= self.max_steps_per_episode: return True
        if state in self.hazards: return True
        if len(self.remaining_goals) <= 0:
            return True
        else:
            return False
    
    def pick_action(self, normalised_state):
        if rn.random() < self.epsilon:
            return rn.randint(0, 3)  # Random action (exploration)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor([normalised_state], dtype=torch.float32, device=self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()  # Best action (exploitation)

    def optimize_model(self):
        # Sample a batch of experiences
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions_batch, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_batch = torch.tensor(actions_batch, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Compute Q(s,a)
        q_values = self.policy_net(states).gather(1, actions_batch).squeeze()

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss
        loss = F.smooth_l1_loss(q_values, targets)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def step(self, current_grid):
        self.episode_steps += 1

        # Normalize state for input to network
        normalized_state = self.grid_to_state(self.grid)
        action = self.pick_action(normalized_state)

        next_state = self.next_state(action)
        reward = self.calc_reward(self.position)
        self.done = self.is_done(self.position)

        self.total_reward += reward

        self.grid[self.position] = self.colour_map["empty"]
        self.grid[next_state] = self.colour_map["agent"]

        # Store experience in replay buffer
        normalized_next_state = self.grid_to_state(self.grid)
        self.replay_buffer.push(normalized_state, action, reward, normalized_next_state, float(self.done))

        self.position = next_state

        if len(self.replay_buffer) >= self.batch_size:
            self.optimize_model()

        if self.done: self.reset()
        
        return self.grid


    def visualise_step(self, current_grid):
        if self.episode >= self.total_episodes:
            return None

        while self.episode % self.visualise_every_n != 0 or self.episode == 0:
            self.step(None)

        if self.episode % self.visualise_every_n == 0: return self.step(None)
        
        return self.grid
        
    def reset(self):
        if self.episode % self.visualise_every_n == 0: sleep(0.2)
        self.update_values()
        self.position = (rn.randint(1, self.grid_size - 2), rn.randint(1, self.grid_size - 2))  # Random start position
        self.grid = self.init_grid()
        self.total_reward = 0
        self.episode_steps = 0
        self.done = False
        self.remaining_goals = deepcopy(self.goal_states)
        self.episode += 1

    def update_values(self):
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Update the target network periodically
        if self.episode % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Optionally print the total reward every 10 episodes
        if self.episode % 10 == 0:
            print(f"Episode {self.episode}, Total Reward: {self.total_reward:.2f}, Epsilon: {self.epsilon:.2f}")

    def run_n_episodes(self, num_episodes = 1000, vis=False):
        # Training loop
        self.episode = 0
        self.total_episodes = num_episodes
        for episode in range(num_episodes):
            self.total_reward = 0
            if vis:
                run_grid(self.grid, update_func=self.visualise_step, tick_rate=0.2)
            else:
                for t in range(self.max_steps_per_episode):
                    self.step(None)
                    if self.done:
                        break
                self.reset()

        self.show_results()

    def show_results(self):
        print("Training complete")

        # Visualize the learned policy
        print("\nLearned Policy (for x=5):")
        for y in range(self.grid_size):
            state = (5, y)
            normalized_state = (state[0] / (self.grid_size - 1), state[1] / (self.grid_size - 1))
            state_tensor = torch.tensor([normalized_state], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                best_action = q_values.argmax().item()
                action_str = self.actions[best_action]
                print(f"State (5, {y}): Best Action: {action_str}, Q-Values: {q_values.numpy()}")
        print("\nLearned Policy (for y=5):")
        for x in range(self.grid_size):
            state = (x, 5)
            normalized_state = (state[0] / (self.grid_size - 1), state[1] / (self.grid_size - 1))
            state_tensor = torch.tensor([normalized_state], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                best_action = q_values.argmax().item()
                action_str = self.actions[best_action]
                print(f"State ({x}, 5): Best Action: {action_str}, Q-Values: {q_values.numpy()}")

if __name__ == "__main__":
    env = DLGrid(size=11)
    env.run_n_episodes(3000, vis=True)