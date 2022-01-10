import gym
from networks import ValueNet
from model import SARSA

###############################################
############# Parameter Setting ###############
###############################################

NUM_EPISODES = 1250
LEARNING_RATE = 1e-4
GAMMA = 0.99
BATCH_SIZE = 64
EPSILON = 0.1
DECAY = 0.999
UPDATE = 4

###############################################
############## MODEL TRAINING #################
###############################################


def main():
    # set up the environment and agent
    env = gym.make('LunarLander-v2')
    env.seed(0)
    agent = SARSA(env, ValueNet, LEARNING_RATE, GAMMA, EPSILON, BATCH_SIZE)
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
        solved = total_reward > 195.0
        print(f'Episode {episode}, total_reward: {total_reward}, solved: {solved}')
        if episode % UPDATE == 1:
            agent.train()

        # epsilon = max(epsilon * DECAY, EPSILON_end)
        # if episode % 200 == 1:
        #     print(epsilon)


if __name__ == '__main__':
    main()
