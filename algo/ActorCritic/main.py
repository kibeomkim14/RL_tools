import gym
from algo.REINFORCE.networks import MLP
from model import AC
import pyglet

###############################################
############# Parameter Setting ###############
###############################################

NUM_EPISODES = 2000
NUM_TIMESTEP = 200
LEARNING_RATE = 1e-3
BATCH_SIZE = 100
GAMMA = 0.90


def main():
    # set up the environment and agent
    env = gym.make('MountainCar-v0')
    agent = AC(env, MLP, LEARNING_RATE, GAMMA, BATCH_SIZE)
    agent.reset()

    for episode in range(NUM_EPISODES):
        # reset state
        state = env.reset()
        total_reward = 0
        for t in range(NUM_TIMESTEP):
            # take action given state
            action, logprob = agent.act(state)

            # take next step of the environment
            next_state, reward, done, _ = env.step(action)

            # record interaction between environment and the agent
            agent.store(state, action, logprob, reward, next_state, done)
            env.render()

            total_reward += reward
            if done:
                break

        agent.train()
        solved = total_reward > 195.0
        agent.reset()
        print(f'Episode {episode}, total_reward: {total_reward}, solved: {solved}')


if __name__ == '__main__':
    main()