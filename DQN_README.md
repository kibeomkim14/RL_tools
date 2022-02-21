# DQN Algorithm

In this section we will be looking at Deep Q-Network algorithm. This is the start of value based methods that uses deep learning as the function approximator of Q value, $Q(s,a)$. 

## Q-Learning 
Before we start, we need to recap the idea of what Q learning is. Assume that we have a tabular representation of Q values. i.e. it is a rectangular data sheet that contains Q-values given each combination of $(s,a)$. Using the $\epsilon$-greedy algorithm, we select the action, $a_t$.

$\epsilon-greedy\, algorithm$

$\begin{align} 
a_t = \underset{a_t}{argmax}\, Q_{\theta}(s_t,a_t)  
\end{align}$

The equation above tells us that the agent chooses a 'greedy' action that maximizes the Q-value given the state $s_t$. Otherwise, action is chosen randomly over predefined action space.

## Deep Neural Network as an function apporoximator

On top of the idea of Q learning, we add another idea of Deep Neural Network. Neural Network(NN) is already known as a universal function approximator of non-linear functions. Hence, we can calculate Q values like:

$\begin{align} 
Q_{\theta}(s_t,a_t) = NeuralNet(s_t)
\end{align}$

where\, $NeuralNet : \mathcal{S} \rightarrow [0, 1]^{n_a}$, $n_a =$ no of availabe actions

In this case, we set output nodes of neural network to be the probability of selecting the corresponding action. Hence it is important to note that **DQN algorithm only works in discrete action space environment**!!


## Loss function
Unlike REINFORCE, the loss function in this algorithm is rather a simple one. Define a loss function, $L(\theta)$


$\begin{align} 
L(\theta) = \mathbb{E}[y_t - Q_{\theta}(s_t,a_t)]
\end{align}$

where

$\begin{align} 
y_t = r_t + \gamma\, \underset{a_t'}{max}\, Q_{\theta}(s_t', a_t')
\end{align}$

the TD estimate of $Q(s, a)$ and $y_t$ is called a **target**. Note that we take maximal $Q(s',a')$ for calculating the target and the estimation is done with NN, this bring us to the idea of "bootstrapping". Simply, bootstrapping means that we use other estimate to calculate an estimate. But the problem with this, is that we tend to overestimate Q value compared to its true value. Hence the training could be very unstable.

## Implementation

Now we will have a look on implementation. Note that this implementation was done solely on PyTorch. Again, I will implement the same algorithm using JAX.(tentative). 

First we make a DQN class. For the simplicity, we will use a simple MLP to make the implementation simpler (shown as a  `Net`). We also specify `env` input in order to extract the size of state and action spaces from specified gym environment. Discount rate, $\gamma$ is shown as `discount_rate`, and $\alpha$ is shown as `learning_rate` in the code below. We use Adam optimizer as our key optimizer. Also, to store data from the interaction, we would need to define a replay buffer to store these data.

Same as `REINFORCE`, `DQN` inherits from `Algorithm` class, we inherit some properties and methods like:

- act
- store
- reset
- train

For a detailed explanation of these methods, see the README.md of `REINFORCE`. 

```
class DQN(Algorithm):
    def __init__(self, env, Net, learning_rate, disc_rate, batch_size):
        self.env = env
        self.dim_in  = self.env.observation_space.shape[0]
        self.dim_out = self.env.action_space.n

        self.QNet = Net(self.dim_in, self.dim_out)

        self.transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'dones'))
        self.buffer     = ExpReplay(40000, self.transition)
        self.optimizer  = optim.Adam(self.QNet.parameters(), lr=learning_rate)

        self.gamma      = disc_rate
        self.batch_size = batch_size
        self.loss_function = nn.HuberLoss()
```

Now we define `act` method. Note that we are assuming the action space to be **discrete**. Unlike `REINFORCE`, this time `Net` outputs Q-values for each available actions. By $\epsilon$-greedy algorithm, agent takes an action with maximum Q-value given a state $s_t$ with a probability $1-\epsilon$ or a random action with probability $\epsilon$.

```
    def act(self, state, epsilon):
        if random.random() > epsilon:
            x = torch.tensor(state.astype(np.float32))  
            Q_values = self.QNet(x)
            action = torch.argmax(Q_values).item() 
        else:
            action = self.env.action_space.sample()
        return action
```

(We won't go through `store` and `reset` as it is very trivial to setup.)

Finally, we define `train` method. Training an agent is done in following.

1. sample a batch of transitions from experience buffer.
2. calculate a target, $y_t$
3. subtract $Q_{\theta}(s_t,a_t)$ to get a TD error, $\delta_t$
4. perform a gradient descent using Adam optimizer with respect to parameter, $\theta$.


```
def train(self):
   # if the number of sample collected is lower than the batch size,
        # we do not train the model.
        if self.buffer.len() < self.batch_size:
            return

        # else, we begin training.
        # calculate return of all times in the episode
        transitions = self.buffer.sample(self.batch_size)
        batch = self.transition(*zip(*transitions))

        states = torch.tensor(batch.state).view(self.batch_size, self.dim_in)
        actions = torch.tensor(batch.action).view(self.batch_size, 1)
        rewards = torch.tensor(batch.reward).view(self.batch_size, 1)
        dones = torch.tensor(batch.dones).view(self.batch_size, 1).long()
        next_states = torch.tensor(batch.next_state).view(self.batch_size, self.dim_in)

        # calculate loss of policy
        # Below advantage function is the TD estimate of Q(s,a).
        y = rewards + self.gamma * (1 - dones) * torch.max(self.QNet(next_states), 1)[0]
        Q = self.QNet(states).gather(1, actions)
        loss = (y - Q).pow(2).sum()

        # do gradient ascent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```


This is it! We have implemented DQN algorithm. Now then, we will need this algorithm get going! `main.py` file will do this job.

```
import gym
from networks import ValueNet
from model import DQN

###############################################
############# Parameter Setting ###############
###############################################

NUM_EPISODES = 1250
LEARNING_RATE = 1e-4
GAMMA = 0.99
BATCH_SIZE = 64
EPSILON_start = 1
EPSILON_end = 0.01
DECAY = 0.999
UPDATE = 4

###############################################
############## MODEL TRAINING #################
###############################################


def main():
    # set up the environment and agent
    env = gym.make('CartPole-v1')
    env.seed(0)
    agent = DQN(env, ValueNet, LEARNING_RATE, GAMMA, BATCH_SIZE)
    agent.reset()

    # set epsilon value before the training
    epsilon = EPSILON_start

    for episode in range(NUM_EPISODES):
        # reset state
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            # take action given state
            action = agent.act(state, epsilon)

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

        epsilon = max(epsilon * DECAY, EPSILON_end)
        #if episode % 200 == 1:
            #print(epsilon)


if __name__ == '__main__':
    main()
```

In this post, we went through the theory and implementation of DQN algorithm. In the bloodline of value-based algorithms, DQN is the first algorithm to utilize NN. As we have discussed that this algorithm is not so stable hence a few attempts have been made to mitigate this problem. For example, `DoubleDQN` or `Rainbow` are the one. These are achieved by adding a target network and prioritized experience buffer to speed up the learning as well as stabilize the target, $y_t$. 
