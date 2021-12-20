from abc import ABCMeta, abstractmethod


class Algorithm(metaclass=ABCMeta):
    """
    Constructs an abstract based class Algorithm.
    This will be the main bone of all Reinforcement learning algorithms
    contained in this repository. This class gives a clear structure of what needs
    to be implemented for all reinforcement algorithm in general.
    If there is other extra methods to be specified, child class will inherit the existing methods as well as
    add new methods to it.
    """
    def __init__(self, env, disc_rate, learning_rate):
        self.env = env
        self.gamma = disc_rate
        self.lr = learning_rate

    @abstractmethod
    def act(self):
        pass

    @abstractmethod
    def store(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def train(self):
        pass