import gym
import os
import argparse
import torch
from gym import envs
from algo.ac import *
from algo.ddpg import *
from algo.pg import *
from algo.valuebased import *
from network.networks import *

###############################################
############# Parameter Setting ###############
###############################################


agents = {'a2c':A2C, 'ac':AC, 'ddpg':DDPG, 'ddqn':DoubleDQN, 
          'dqn':DQN, 'dqnper':DoubleDQN_PER, 'ppo':PPO, 'pg':REINFORCE, 
          'sac':SAC, 'td3':TD3}

environments = [spec.id for spec in envs.registry.all()]
networks = {'mlp':MLP, 'cnn':CNN}

if __name__ == '__main__':
    # initialize ArgumentParser class of argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--env"   , help="Environment", default="MountainCar-v0", type=str, choices=environments)
    parser.add_argument("--agent" , help="RL agent", default="DQN", type=str, choices=agents)
    parser.add_argument("--net"   , help="type of network", default='MLP', type=str)
    parser.add_argument("--gamma" , help="discount rate", default=0.99, type=float)
    parser.add_argument("--epsilon", help="exploration coeff", default=0.05, type=float)
    parser.add_argument("--n_episode", help="number of episodes", default=100, type=int)
    parser.add_argument("--n_maxstep", help="maximum steps per episode", default=500, type=int)
    parser.add_argument("--learning_rate", help="learning rate", default=1e-3, type=float)
    parser.add_argument("--netkwargs", nargs='+'), 

    # read the arguments from the command line
    args = parser.parse_args()
    print(args.netkwargs)
    # set up the environment and agent
    env = gym.make(args.env)
    agent = agents[args.agent](env, args.net , args.learning_rate, args.gamma, args.netkwargs)

    for episode in range(args.n_episode):
        # reset state
        state = env.reset()
        total_reward = 0

        for t in range(args.n_maxstep):
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


