import numpy as np

class QLearningAgent:
    def __init__(self, num_states, num_actions, alpha, gamma, epsilon, state_bounds, num_bins):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((num_states, num_actions))

        self.state_bounds = state_bounds
        self.num_bins = num_bins
        self.bin_edges = [np.linspace(b[0], b[1], n + 1) for b, n in zip(state_bounds, num_bins)]

    def discretize_state(self, observation):
        indices = []
        for i, (obs, edges) in enumerate(zip(observation, self.bin_edges)):
            idx = np.digitize(obs, edges) - 1
            idx = min(idx, len(edges) - 2)
            indices.append(idx)
        state_index = 0
        multiplier = 1
        for i in reversed(range(len(indices))):
            state_index += indices[i] * multiplier
            multiplier *= self.num_bins[i]
        return state_index

    def get_state(self, observation):
        return self.discretize_state(observation)

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.Q[state])

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.gamma * self.Q[next_state][best_next_action]
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

    def save_q_table(self, file_name):
        np.save(file_name, self.Q)

    def load_q_table(self, file_name):
        self.Q = np.load(file_name)
