import numpy as np

class ReplayBuffer:
    def __init__(self,
                 capacity,
                 batch_size,
                 state_shape,
                 action_shape):
        self.states = np.zeros((capacity, state_shape))
        self.actions = np.zeros((capacity, action_shape))
        self.next_states = np.zeros((capacity, state_shape))
        self.rewards = np.zeros(capacity)
        self.is_terminals = np.zeros(capacity)
        self.capacity = capacity
        self.batch_size = batch_size
        self.idx = 0
        self.size = 0

    def store(self,
              state,
              action,
              reward,
              next_state,
              is_failure):
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.is_terminals[self.idx] = 1 - is_failure
        self.idx += 1
        self.idx = self.idx % self.capacity
        self.size += 1
        self.size = min(self.size, self.capacity)

    def get_batch(self, batch_size=None):
        batch_size = self.batch_size if batch_size==None else batch_size
        batch = np.random.choice(self.size, batch_size)

        states = self.states[batch]
        actions = self.actions[batch]
        rewards = self.rewards[batch]
        next_states = self.next_states[batch]
        is_terminals = self.is_terminals[batch]

        return states, actions, rewards, next_states, is_terminals
