###############################################
#Author: Praveen Kumar
#Date: 05/12/2019
#For queries raise issue on github repository
###############################################
#########################################
#       Library Imports                 #
#########################################
import gym
import tensorflow as tf
#########################################
#       Helper Functions                #
#########################################
from Replay_Buffer import ReplayBuffer
from DQN_Agent import DqnAgent

MAX_EPISODES = 100

def evaluate_training_result(env, agent):
    """
    evaluate_training_result
    :param env: Training Environment - CartpoleV0
    :param agent: DQN Agent for evaluation
    :return: Average reward gained by DQN Agent during evaluation
    """
    #Initialized
    total_reward = 0.0
    #Fixed number of episodes for evaluation
    episodes_to_play = 10 
    for i in range(episodes_to_play):
        state = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action = agent.policy(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        total_reward += episode_reward
    average_reward = total_reward / episodes_to_play
    return average_reward

def collect_gameplay_experiences(env, agent, buffer):
    """
    collect gameplay experiences
    :param env: the game environemnt
    :param agent: the DQN Agent
    :param buffer: the replay buffer
    :return: None
    """
    state = env.reset()
    done = False
    while not done:
        action = agent.collect_policy(state)
        next_state, reward, done, _ = env.step(action)
        if done:
            reward = -1.0
        buffer.store_gameplay_experience(state, next_state,
                                         reward, action, done)
        state = next_state

def main(max_episodes=MAX_EPISODES):
    """
    main: Train the model
    :param max_episodes: max number episoedes to train model
    :return: None
    """
    agent = DqnAgent()
    buffer = ReplayBuffer()
    env = gym.make('CartPole-v0')
    for _ in range(100):
        collect_gameplay_experiences(env, agent, buffer)
    for episode_cnt in range(max_episodes):
        collect_gameplay_experiences(env, agent, buffer)
        gameplay_experience_batch = buffer.sample_gameplay_batch()
        loss = agent.train(gameplay_experience_batch)
        avg_reward = evaluate_training_result(env, agent)
        
        #Uncomment to view cartpole progress
        #env.render() 
        print('Episode {0}/{1} and so far the performance is {2} and loss is {3}'.format(episode_cnt, max_episodes, avg_reward, loss[0]))
        
        #Update network after every 20 episodes
        if episode_cnt % 20 == 0:
            agent.update_target_network()
    env.close()

# Start the training
if __name__ == "__main__":
    main()
