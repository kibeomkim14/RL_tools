import gym
from networks import ValueNet
from model import DQN

###############################################
############# Parameter Setting ###############
###############################################

NUM_EPISODES = 300
LEARNING_RATE = 1e-3
GAMMA = 0.9
BATCH_SIZE = 100
EPSILON = 0.15

###############################################
############## MODEL TRAINING #################
###############################################


def main():
    # set up the environment and agent
    env = gym.make('CartPole-v0')
    agent = DQN(env, ValueNet, LEARNING_RATE, GAMMA, EPSILON, BATCH_SIZE)
    agent.reset()

    for episode in range(NUM_EPISODES):
        # reset state
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            # take action given state
            action = agent.act(state)

            # take next step of the environment
            next_state, reward, done, _ = env.step(action)

            # record interaction between environment and the agent
            agent.store(state, action, reward, next_state, done)
            env.render()

            total_reward += reward
            state = next_state
            if done:
                break

        agent.train()
        solved = total_reward > 195.0
        print(f'Episode {episode}, total_reward: {total_reward}, solved: {solved}')


if __name__ == '__main__':
    main()
