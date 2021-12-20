from collections import namedtuple, deque
import random


class ExpReplay:
    def __init__(self, capacity, transition):
        self.memory = deque([], maxlen=capacity)
        self.record = transition

    def __len__(self):
        return len(self.memory)

    def store(self, *args):
        """Save a transition"""
        self.memory.append(self.record(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def clear(self):
        return self.memory.clear()
