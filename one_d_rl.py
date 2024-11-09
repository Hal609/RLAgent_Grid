import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque

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

# Initialize the policy and target networks
policy_net = DQN()
target_net = DQN()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()  # Set target network to evaluation mode

optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)

# Parameters
num_episodes = 1000
gamma = 0.9  # Discount factor

# Epsilon parameters for decay
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
epsilon = epsilon_start

target_update = 10  # How often to update the target network
max_steps_per_episode = 50
grid_size = 11  # Grid size (11x11)
goal_state = (5, 5)  # Target position the agent should reach

# Replay buffer
replay_buffer = deque(maxlen=10000)
batch_size = 64

# Action space
actions = ['up', 'down', 'left', 'right']

# Function to select an action based on epsilon-greedy policy with decay
def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 3)  # Random action (exploration)
    else:
        with torch.no_grad():
            state_tensor = torch.tensor([state], dtype=torch.float32)
            q_values = policy_net(state_tensor)
            return q_values.argmax().item()  # Best action (exploitation)

# Environment step function with modified reward
def env_step(state, action):
    x, y = state
    if action == 0:  # Up
        y = y - 1
    elif action == 1:  # Down
        y = y + 1
    elif action == 2:  # Left
        x = x - 1
    elif action == 3:  # Right
        x = x + 1
    # Keep within grid boundaries
    x = max(0, min(grid_size - 1, x))
    y = max(0, min(grid_size - 1, y))
    next_state = (x, y)
    distance_to_goal = abs(x - goal_state[0]) + abs(y - goal_state[1])  # Manhattan distance
    reward = -distance_to_goal / (2 * (grid_size - 1))  # Normalize reward between -1 and 0
    if next_state == goal_state:
        reward += 1.0  # Positive reward for reaching the goal
        done = True
    else:
        done = False
    return next_state, reward, done

# Training loop
for episode in range(num_episodes):
    state = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))  # Random start position
    total_reward = 0
    for t in range(max_steps_per_episode):
        # Normalize state for input to network
        normalized_state = (state[0] / (grid_size - 1), state[1] / (grid_size - 1))
        action = select_action(normalized_state, epsilon)
        next_state, reward, done = env_step(state, action)
        total_reward += reward

        # Store experience in replay buffer
        normalized_next_state = (next_state[0] / (grid_size - 1), next_state[1] / (grid_size - 1))
        replay_buffer.append((normalized_state, action, reward, normalized_next_state, float(done)))

        state = next_state

        if len(replay_buffer) >= batch_size:
            # Sample a batch of experiences
            batch = random.sample(replay_buffer, batch_size)
            states, actions_batch, rewards, next_states, dones = zip(*batch)
            
            states = torch.tensor(states, dtype=torch.float32)
            actions_batch = torch.tensor(actions_batch, dtype=torch.int64).unsqueeze(1)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)

            # Compute Q(s,a)
            q_values = policy_net(states).gather(1, actions_batch).squeeze()

            # Compute target Q-values
            with torch.no_grad():
                next_q_values = target_net(next_states).max(1)[0]
                targets = rewards + gamma * next_q_values * (1 - dones)

            # Compute loss
            loss = F.smooth_l1_loss(q_values, targets)

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    # Decay epsilon
    epsilon = max(epsilon_end, epsilon * epsilon_decay)

    # Update the target network periodically
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Optionally print the total reward every 10 episodes
    if episode % 10 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

print("Training complete")

# Visualize the learned policy
print("\nLearned Policy (for x=5):")
for y in range(grid_size):
    state = (5, y)
    normalized_state = (state[0] / (grid_size - 1), state[1] / (grid_size - 1))
    state_tensor = torch.tensor([normalized_state], dtype=torch.float32)
    with torch.no_grad():
        q_values = policy_net(state_tensor)
        best_action = q_values.argmax().item()
        action_str = actions[best_action]
        print(f"State (5, {y}): Best Action: {action_str}, Q-Values: {q_values.numpy()}")
print("\nLearned Policy (for y=5):")
for x in range(grid_size):
    state = (x, 5)
    normalized_state = (state[0] / (grid_size - 1), state[1] / (grid_size - 1))
    state_tensor = torch.tensor([normalized_state], dtype=torch.float32)
    with torch.no_grad():
        q_values = policy_net(state_tensor)
        best_action = q_values.argmax().item()
        action_str = actions[best_action]
        print(f"State (5, {y}): Best Action: {action_str}, Q-Values: {q_values.numpy()}")