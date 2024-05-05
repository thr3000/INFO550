import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import namedtuple, deque

LEARNING_RATE = 0.001
GAMMA = 0.99
BATCH_SIZE = 32
CAPACITY = 10000
EPSILON = 0.1
EXPERIENCE = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(5, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, next_state, reward):
        self.memory.append(EXPERIENCE(state, action, next_state, reward))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size

    def __len__(self):
        return len(self.memory)


def select_action(state, policy_net, n_actions=3):
    if random.random() > EPSILON:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)

def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = EXPERIENCE(*zip(*transitions))

    non_final_mask = torch.tensor([s is not None for s in batch.next_state], dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    if action_batch.dim() > 2:
        action_batch = action_batch.squeeze(-1)
    elif action_batch.dim() < 2:
        action_batch = action_batch.unsqueeze(-1)

    predictions = policy_net(state_batch)
    state_action_values = predictions.gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=state_batch.device)
    if non_final_next_states.size(0) > 0:
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    expected_state_action_values = expected_state_action_values.unsqueeze(1)

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



policy_net = DQN()
target_net = DQN()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayMemory(CAPACITY)

def update_target_net():
    target_net.load_state_dict(policy_net.state_dict())

__all__ = ['select_action', 'optimize_model', 'update_target_net', 'policy_net', 'target_net', 'optimizer', 'memory', 'ReplayMemory']
