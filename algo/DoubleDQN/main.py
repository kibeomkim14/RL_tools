import gym
from networks import ValueNet
from model import DoubleDQN

###############################################
############# Parameter Setting ###############
###############################################

NUM_EPISODES  = 1200
NUM_TIMESTEP  = 100
LEARNING_RATE = 1e-4
BATCH_SIZE    = 120
GAMMA         = 0.99
EPSILON       = 0.05
TAU           = 0.02
EPSILON_start = 1
EPSILON_end = 0.05
DECAY = 0.999
UPDATE = 4

###############################################
############## MODEL TRAINING #################
###############################################


def main():
    # set up the environment and agent
    env = gym.make('LunarLander-v2')
    agent = DoubleDQN(env, ValueNet, LEARNING_RATE, GAMMA, BATCH_SIZE, TAU)
    agent.reset()
    epsilon = EPSILON_start

    for episode in range(NUM_EPISODES):
        # reset state
        state = env.reset()
        total_reward = 0
        # set epsilon value before the training
        done = False

        while not done:
            # take action given state
            action = agent.act(state, epsilon)

            # take next step of the environment
            next_state, reward, done, _ = env.step(action)

            # record interaction between environment and the agent
            agent.store(state, action, reward, next_state, done)
            env.render()

            state = next_state
            total_reward += reward
            if done:
                break

        if episode % UPDATE:
            agent.train()
        epsilon = max(epsilon * DECAY, EPSILON_end)
        solved  = total_reward > 195.0
        print(f'Episode {episode}, total_reward: {total_reward}, solved: {solved}')
        if episode % UPDATE == 1:
            agent.train()

        if episode % 200 == 1:
            print(epsilon)


if __name__ == '__main__':
    main()
