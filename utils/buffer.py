from collections import namedtuple, deque
import random


PolicyTransition = namedtuple('Transition', ('state', 'action', 'logprobs', 'reward', 'next_state', 'dones'))


class ExpReplay:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def store(self, *args):
        """Save a transition"""
        self.memory.append(PolicyTransition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def clear(self):
        return self.memory.clear()


class Transition:
    def __init__(self):
        self.tuple = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class DQNMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def store(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def clear(self):
        return self.memory.clear()
