class ExpReplay:
    def __init__(self):
        self.actions  = []
        self.states   = []
        self.logprobs = []
        self.rewards  = []
        self.dones    = []

    def store(self, state_tuple):
        state, action, log_prob, reward, done = state_tuple

        self.actions.append(action)
        self.states.append(state)
        self.rewards.append(reward)
        self.logprobs.append(log_prob)
        self.dones.append(done)

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]


class DQNMemory:
    def __init__(self):
        self.states   = []
        self.actions  = []
        self.rewards  = []
        self.next_states  = []
        self.next_actions = []

    def store(self, state_tuple):
        state, action, reward, next_state, next_action = state_tuple

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.next_actions.append(next_action)

    def clear(self):
        del self.states
        del self.actions
        del self.rewards
        del self.next_states
        del self.next_actions
