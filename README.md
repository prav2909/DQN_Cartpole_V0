# DQN-on-CartPole-Environemnt-from-openai
DQN with replay buffer on Cartpole environment.

Deep Q Network(DQN)
Q learning, a ground breaking development in reinforcement learning, was created from an off-policy temporal difference control algorithm. It attempts to dictate the best possible action to take, given the state in which the robot is. In its simplest form, Q-learning algorithm stores something called Q-Value. Q value of a state is the sum of maximum possible future rewards that the robot can get if it goes in that state. Q-learning motivates the robots to always choose the states that have maximum Q value. The Q- Values are learned and updated as the robot explores the environment using the famous Bellman equation, which is represented in a simplified manner in Equation below.

Q(s) = E[Rt+1 + γq(St+1)|St = s]
Where,
Rt+1 = Immediate reward for Rt+1
γ ∶ discount factor, immediate reward is more important than future rewards
q(St+1
): Successor state rewards
s: State
E[ ]: Expected value

Bellman equation describes the expected value of sum of immediate reward and discounted value of maximum future rewards. Deep Q Network is a combination of Deep Learning and Q Learning. Remembering Q-value for all state action pair, as done in Q Learning, is a memory extensive task. Hence, neural networks are used to estimate Q values instead of memorizing them. In a way, the neural networks approximate a heuristic function that rates the next step.
However, in classic Deep Learning problems, the labels for same input values do not change over time, this creates a stable condition for learning. This is not the case with reinforcement learning, as the robot explores the environment, it starts to know the environment better, and the labels for the same inputs get updated. This makes the
training unstable and is referred as unstable target problem. Two strategies are used to solve this problem, Experience Replay and Separating Training and Target network. In Experience Replay, instead of using the entire learned data for training, only a small buffer is used for creating batches, this creates an input dataset stable enough for training. The second strategy keeps the neural network weights of training network and target network separate, which makes sure that target does not change during the training. The training network weights are copied to the target after some iterations.

To Run the code, clone the repository, intall requirements.txt and run main.py 

For any questions, contact:
https://www.linkedin.com/in/praveen-kumar-b2096391/
