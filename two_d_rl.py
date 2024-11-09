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
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(2, 32)  # Now input size is 2 (x and y coordinates)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 4)  # Output size is 4 (up, down, left, right)
        self.apply(self.init_weights)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

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
                    "mps" if torch.backends.mps.is_available() else
                    "cpu")
        
        # Initialize the policy and target networks
        self.policy_net = DQN()
        self.target_net = DQN()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set target network to evaluation mode

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)

        # Parameters
        self.gamma = 0.9  # Discount factor

        # Epsilon parameters for decay
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.999
        self.epsilon = self.epsilon_start

        self.target_update = 10  # How often to update the target network
        self.max_steps_per_episode = 50
        self.episode_steps = 0
        self.grid_size = size # Grid size, defaults to (11x11)
        self.goal_states = [(size//2, size//2)]  # Target position the agent should reach
        self.remaining_goals = deepcopy(self.goal_states)
        self.done = False

        # Replay buffer
        self.replay_buffer = ReplayMemory(10000)
        self.batch_size = 64

        # Action space
        self.actions = ['up', 'down', 'left', 'right']

        # State
        self.state = (rn.randint(1, self.grid_size - 2), rn.randint(1, self.grid_size - 2))  # Random start position

        # Grid
        self.grid = self.init_grid()

    def init_grid(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype='U8')
        for y, row in enumerate(grid):
            for x, cell in enumerate(row):
                grid[y][x] = "000000ff"
                if y == 0 or y == self.grid_size - 1 or x == 0 or x == self.grid_size - 1:
                    grid[y][x] = "ffffffff"
                else:
                    grid[y][x] == "000000ff"
        for goal in self.goal_states:
            grid[goal[0]][goal[1]] = "ff000044"

        return grid
    
    def calc_reward(self, state):
        reward = 0
        x, y = state
        # distance_to_goal = abs(x - self.goal_state[0]) + abs(y - self.goal_state[1])  # Manhattan distance
        if state in self.remaining_goals:
            self.remaining_goals.remove(state)
            reward += 1.0  # Positive reward for reaching the goal
        reward +=  -0.1
        return reward / (2 * (self.grid_size - 1))  # Normalize reward between -1 and 0
    
    def next_state(self, action):
        x, y = self.state
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

    def is_done(self, next_state):
        if self.episode_steps >= self.max_steps_per_episode: return True
        if len(self.remaining_goals) == 0:
            return True
        else:
            return False
    
    def pick_action(self, normalised_state):
        if rn.random() < self.epsilon:
            return rn.randint(0, 3)  # Random action (exploration)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor([normalised_state], dtype=torch.float32)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()  # Best action (exploitation)

    def optimize_model(self):
        # Sample a batch of experiences
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions_batch, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(states, dtype=torch.float32)
        actions_batch = torch.tensor(actions_batch, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

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
        normalized_state = (self.state[0] / (self.grid_size - 1), self.state[1] / (self.grid_size - 1))
        action = self.pick_action(normalized_state)

        next_state = self.next_state(action)
        reward = self.calc_reward(self.state)
        self.done = self.is_done(self.state)

        self.total_reward += reward

        # Store experience in replay buffer
        normalized_next_state = (next_state[0] / (self.grid_size - 1), next_state[1] / (self.grid_size - 1))
        self.replay_buffer.push(normalized_state, action, reward, normalized_next_state, float(self.done))

        self.grid[self.state] = "000000"
        self.grid[next_state] = "FFFFFF"

        self.state = next_state

        if len(self.replay_buffer) >= self.batch_size:
            self.optimize_model()

        if self.done: self.reset()
        
        return self.grid


    def visualise_step(self, current_grid):
        if self.episode % 100 == 0: return self.step(None)

        while self.episode % 100 != 0:
            self.step(None)

        if self.episode >= self.total_episodes:
            return None
        
        return self.grid
        
    def reset(self):
        self.update_values()
        self.state = (rn.randint(1, self.grid_size - 2), rn.randint(1, self.grid_size - 2))  # Random start position
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
                run_grid(self.grid, update_func=self.visualise_step)
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
            state_tensor = torch.tensor([normalized_state], dtype=torch.float32)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                best_action = q_values.argmax().item()
                action_str = self.actions[best_action]
                print(f"State (5, {y}): Best Action: {action_str}, Q-Values: {q_values.numpy()}")
        print("\nLearned Policy (for y=5):")
        for x in range(self.grid_size):
            state = (x, 5)
            normalized_state = (state[0] / (self.grid_size - 1), state[1] / (self.grid_size - 1))
            state_tensor = torch.tensor([normalized_state], dtype=torch.float32)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                best_action = q_values.argmax().item()
                action_str = self.actions[best_action]
                print(f"State ({x}, 5): Best Action: {action_str}, Q-Values: {q_values.numpy()}")

if __name__ == "__main__":
    env = DLGrid(size=11)
    env.run_n_episodes(2000, vis=True)