# REINFORCE Algorithm

In this section we will be looking at REINFORCE algorithm which was introduced in year 1992 by Ronald J. Williams. Unlike model-based learning algorithms, this algorithm uses model-free approach. In other words, we don't have a prior knowledge about the transition probability, $P(s',r|s,a)$.

Key idea of REINFORCE is that we instead learn a policy function $\pi(s_t)$ which takes the state, $s_t$ and outputs an action, $a_t$. During the training, actions that resulted in good outcomes should become more probable - i.e these actions are 'reinforced'.  

## Policy function

Define a policy function $\pi : \mathcal{S} \rightarrow \mathcal{A}$, where it is a mapping that maps states to actions. 

$\begin{align} 
a \sim \pi(s)  
\end{align}$

Since we are interested in making a policy which directs to a good return, $G_t$ where it is formulated as:

$\begin{align} 
G_t = \Sigma^{T}_{j = 0} \gamma^{t} R_{t + j} 
\end{align}$

This means we are maximizing the return, which we can see the problem as an optimization problem. With the goal specified by now, we can view the policy function as an function approximator, which we will use deep learning algorithm.

Suppose we have a trajectory $\tau$, the sequence taken out from Markov Decision Process. The example of trajectory is like one as below.

$\begin{align} 
\tau = s_0, a_0, r_0, ... , s_T, a_T, r_T
\end{align}$

Given the trajectory, we define the objective function, $J(\pi_\theta)$.

$\begin{align} 
J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \mathbb{E}_{\tau \sim \pi_\theta}[\Sigma^{T}_{t=0} \gamma^{t} R_{t} ]
\end{align}$

>**Intuition Note**\
Above expression means that expectation is calculated by taking the simple average of returns genereated by many trajectories, $\tau$. If there was exactly 1000 trajectories available, then we could get the approximation of $J(\pi_\theta)$ by calculating the sample mean of 1000 returns. Due to the central limit theorem, as we have more samples, the approximated value will converge to the real value.


## Loss function
Just now, we defined the objective funtion $J(\pi_\theta)$. To trigger the learning, we would need a loss function and its associated loss gradients to perform a gradient descent. In REINFORCE, we solve the following problem:


$\begin{align} 
\underset{\theta}{\max}\, J(\pi_\theta) = \underset{\theta}{\max}\, \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] 
\end{align}$
$\begin{align} 
\theta \leftarrow \theta + \alpha \nabla_{\theta} J(\pi_\theta) 
\end{align}$

(**NOTE** This is gradient ASCENT not gradient DESCENT!)
where $\theta$ is parameters of $\pi$, $\alpha$ is the learning rate, which controls the rate of update. This is called **policy gradient**.

$\begin{align} 
\nabla_{\theta} J(\pi_\theta) &= \nabla_{\theta} \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] \\
&= \nabla_{\theta} \int  R(\tau) \, \pi(\tau|\theta) \, d\tau\\
&= \int \nabla_{\theta}(R(\tau) \, \pi(\tau|\theta)) \, d\tau\\
&= \int \nabla_{\theta}R(\tau) \, \pi(\tau|\theta) +  R(\tau) \, \nabla_{\theta} \pi(\tau|\theta) \,d\tau\\
&= \int R(\tau) \, \nabla_{\theta} \pi(\tau|\theta) \,d\tau\\
&= \int R(\tau) \, \pi(\tau|\theta)\frac {\nabla_{\theta} \pi(\tau|\theta)}{\pi(\tau|\theta)} \,d\tau\\
&= \int R(\tau) \, \pi(\tau|\theta)\nabla_{\theta} log\, \pi(\tau|\theta)\,d\tau\\
&= \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)\nabla_{\theta} log\, \pi(\tau|\theta)] \\
\end{align}$

We have used few tricks to rephrase the equation (7) to (14). Note that in (14), we have a policy function with trajectory, $\tau$ as an input. We will have to disect this quantity into a combination of actions, $a_t$.

$\begin{align} 
\nabla_{\theta}\, log p(\tau | \theta) = \nabla_{\theta} \Sigma_{t \geq 0}\, log \pi(a_t|s_t) \\
\end{align}$

Combining (14) and (15) together, we finally get:

$\begin{align}
\nabla_{\theta} J(\pi_\theta) &= \mathbb{E}_{\tau \sim \pi_\theta}[\Sigma^{T}_{t = 0} R_t\, \nabla_{\theta}\, log\, \pi(a_t|s_t)]   \\
\end{align}$

## Implementation

Now we will have a look on implementation. Note that this implementation was done solely on PyTorch. Soon later, I will implement the same algorithm using JAX.(tentative). 

First we make a REINFORCE class. For the simplicity, we will use a simple MLP to make the implementation simpler (shown as a  `Net`). We also specify `env` input in order to extract the size of state and action spaces from specified gym environment. Discount rate, $\gamma$ is shown as `discount_rate`, and $\alpha$ is shown as `learning_rate` in the code below. We use Adam optimizer as our key optimizer. Also, to store data from the interaction, we would need to define a replay buffer to store these data.

Starting from now, all reinforcement learning algorithms will inherit the characteristic from parent class `Algorithm`. (For more details see `utils.Algorithm` file.) The reason for this is,an `Algorithm` class guides us to make sure all necessary methods are introduced in order to ensure all RL algorithms work, which makes this class to be perceived as a blue print of all RL algorithms. Also, `Algorithm` class already has a parameter `self.buffer` in it. However, due to the difference of inputs between all RL algorithm, we would need to introduce the structure of the data we want to input each time we define the class. See `self.transition` to gain more understanding of the comment above.

By inheriting `Algorithm` class, we inherit some properties and methods. Especially,

- act
- store
- reset
- train.

`Algorithm.act` method defines the agent taking an action given a state as an input. Next two methods `Algorithm.store` & `Algorithm.reset` are used in the context of utilizing **Experience Buffer**. This will become much more clearer as we move towards more advanced algorithms. Lastly, `Algorithm.train` is a method of training an agent using gradient descent or any other equivalents. And easily, this method is by far the most trickiest part to implement.

```
class REINFORCE(Algorithm):
    def __init__(self, env, Net, learning_rate, disc_rate):
        self.dim_in  = env.observation_space.shape[0]
        self.dim_out = env.action_space.n
        self.policy  = Net(self.dim_in, self.dim_out)
        self.gamma   = disc_rate
        self.transition = namedtuple('Transition', ('state', 'action', 'logprobs', 'reward', 'dones'))
        self.buffer = ExpReplay(10000, self.transition)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
    
    def reset(self):
        ...

    def act(self, state):
        ...
    
    def store(self, *args):
        ...
    
    def train(self):
        ...
```

Now we define `act` method. Note that we are assuming the action space to be **discrete**. This doesn't mean all RL algorithms work on discrete action spaces - there are continuous action space problems. If this is the case, then the `Net` should output parameters of the distribution of one's choice. Then these parameters are used to specify the distribution and sample an action from it. For now, we will review discrete action space. 

`Net` outputs probabilities for each action in the action space. Then `torch.distributions.Categorical` is used to set up the distribution. We then take out an action and its log_probability, which will be stored and used in the training.



```
    def act(self, state):
        x = torch.from_numpy(state.astype(np.float32))  
        prob = self.policy.forward(x)
        dist = Categorical(prob)

        action = dist.sample()  
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
```

(We won't go through `store` and `reset` as it is very trivial to setup.)

Finally, we define `train` method. Training an agent is done in following.

1. sample a batch of transitions from experience buffer.
2. calculate a return, $G_t$
3. multiply return $G_t$ with its log probability
4. perform a gradient ascent using Adam optimizer


```
    def train(self):
        transitions = self.buffer.sample(self.buffer.__len__())
        batch = self.transition(*zip(*transitions))
        reward_list = batch.reward

        # calculate return of all times in the episode
        T = len(reward_list)
        returns = np.empty(T, dtype=np.float32)
        future_return = 0.0

        # calculate returns recursively
        for t in reversed(range(T)):
            future_return = reward_list[t] + self.gamma * future_return
            returns[t] = future_return

        # calculate loss of policy
        returns = torch.tensor(returns)
        log_probs = torch.stack(batch.logprobs)
        loss = - log_probs * returns
        loss = torch.sum(loss)

        # do gradient ascent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```


This is it! We have implemented REINFORCE algorithm. Now then, we will need this algorithm get going! `main.py` file will do this job.

```
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

```

In this post, we went through the theory and implementation of REINFORCE algorithm. This algorithm is the father of all policy based algorithms such as Actor Critic and its other variants - i.e PPO. It is important to understand why we use log probability to calculate the gradient of the objective function $J$. 

Although it is simple to understand and use, REINFORCE algorithm has some drawbacks. For example, learning could be very unstable since the value $R_t$ in $\nabla J(\pi_\theta)$ could fluctuate a lot. This makes an optimization process to be of high variance hence makes the training very unstable. To mitigate this problem, a few suggestions made on replacing $G_t$ with other alternatives, such as an **Advantage** function. This will be discussed in later posts. 
