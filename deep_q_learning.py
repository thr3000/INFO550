import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import namedtuple, deque

EXPERIENCE = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))

class DQN(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=24, output_dim=3):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(EXPERIENCE(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size

    def __len__(self):
        return len(self.memory)

class DeepQLearningAgent:
    def __init__(self, alpha, gamma, epsilon):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = 32
        self.policy_net = DQN(input_dim=5, hidden_dim=24, output_dim=3)
        self.target_net = DQN(input_dim=5, hidden_dim=24, output_dim=3)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=alpha)
        self.memory = ReplayMemory(10000)

    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.policy_net.fc3.out_features)]], dtype=torch.long)

    def optimize_model(self):
        if not self.memory.can_provide_sample(self.batch_size):
            return

        transitions = self.memory.sample(self.batch_size)
        batch = EXPERIENCE(*zip(*transitions))

        non_final_mask = torch.tensor([s is not None for s in batch.next_state], dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        predictions = self.policy_net(state_batch)
        state_action_values = predictions.gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=state_batch.device)
        if non_final_next_states.size(0) > 0:
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        expected_state_action_values = expected_state_action_values.unsqueeze(1)

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, filename):
        torch.save(self.policy_net.state_dict(), filename)

    def load_model(self, filename):
        self.policy_net.load_state_dict(torch.load(filename))
        self.update_target_net()
