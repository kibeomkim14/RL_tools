import gym
from utils.networks import MLP
from REINFORCE import REINFORCE


###############################################
############# Parameter Setting ###############
###############################################

NUM_EPISODES = 100
NUM_TIMESTEP = 200
LEARNING_RATE = 0.001
GAMMA = 0.99


def main():
    # set up the environment and agent
    env = gym.make('CartPole-v0')
    agent = REINFORCE(env, MLP, LEARNING_RATE, GAMMA)

    for episode in range(NUM_EPISODES):
        # reset state
        state = env.reset()
        for t in range(NUM_TIMESTEP):
            # take action given state
            action, logprob = agent.act(state)

            # take next step of the environment
            state, reward, done, _ = env.step(action)

            # record interaction between environment and the agent
            agent.store([state, action, logprob, reward, done])

            env.render()

            if done:
                break

        agent.train()
        total_reward = sum(agent.buffer.rewards)
        solved = total_reward > 195.0
        agent.reset()
        print(f'Episode {episode}, total_reward: {total_reward}, solved: {solved}')


if __name__ == '__main__':
    main()
