import gym
from networks import ValueNet
from model import DoubleDQN

###############################################
############# Parameter Setting ###############
###############################################

NUM_EPISODES  = 2000
NUM_TIMESTEP  = 100
LEARNING_RATE = 1e-4
BATCH_SIZE    = 500
GAMMA         = 0.90
EPSILON       = 0.05
TAU           = 0.02
UPDATE_T      = 100

###############################################
############## MODEL TRAINING #################
###############################################


def main():
    # set up the environment and agent
    env = gym.make('CartPole-v0')
    agent = DoubleDQN(env, ValueNet, LEARNING_RATE, GAMMA, EPSILON, BATCH_SIZE, TAU)
    agent.reset()

    for episode in range(NUM_EPISODES):
        # reset state
        state = env.reset()
        total_reward = 0
        for t in range(NUM_TIMESTEP):
            # take action given state
            action = agent.act(state)

            # take next step of the environment
            next_state, reward, done, _ = env.step(action)

            # record interaction between environment and the agent
            agent.store(state, action, reward, next_state, done)
            env.render()

            total_reward += reward
            if done:
                break

        if episode % UPDATE_T:
            agent.train()
        solved = total_reward > 195.0
        agent.reset()
        print(f'Episode {episode}, total_reward: {total_reward}, solved: {solved}')


if __name__ == '__main__':
    main()
