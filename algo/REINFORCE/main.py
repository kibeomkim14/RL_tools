import gym
from algo.REINFORCE.networks import MLP
from model import REINFORCE


###############################################
############# Parameter Setting ###############
###############################################

NUM_EPISODES = 1000
NUM_TIMESTEP = 200
LEARNING_RATE = 0.001
GAMMA = 0.99


def main():
    # set up the environment and agent
    env = gym.make('MountainCar-v0')
    agent = REINFORCE(env, MLP, LEARNING_RATE, GAMMA)

    for episode in range(NUM_EPISODES):
        # reset state
        state = env.reset()
        total_reward = 0

        for t in range(NUM_TIMESTEP):
            # take action given state
            action, logprob = agent.act(state)

            # take next step of the environment
            state, reward, done, _ = env.step(action)

            # record interaction between environment and the agent
            agent.store(state, action, logprob, reward, done)
            env.render()
            total_reward += reward

            if done:
                break

        agent.train()
        solved = total_reward > 195.0
        if solved:
            print(f'solved!! (at episode {episode}, reward {total_reward}')
            break
        agent.reset()
        print(f'Episode {episode}, total_reward: {total_reward}, solved: {solved}')


if __name__ == '__main__':
    main()
