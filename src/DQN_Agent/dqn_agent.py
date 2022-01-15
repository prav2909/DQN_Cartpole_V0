#########################################
#       Library Imports                 #
#########################################
import numpy as np
from tensorflow import convert_to_tensor
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

class DqnAgent:
    """
    DQN Agent
    """
    def __init__(self):
        self.q_net = self._build_dqn_model()
        self.target_q_net = self._build_dqn_model()

    @staticmethod
    def _build_dqn_model():
        """
        Builds DQN a deep nn to predict Q values for all possible actions in that state
        :return: Q network
        """
        q_net = Sequential()
        q_net.add(Dense(64, input_dim=4, activation='relu',
                        kernel_initializer='he_uniform', dtype='float32'))
        q_net.add(Dense(32, activation='relu',
                        kernel_initializer='he_uniform'))
        q_net.add(Dense(2, activation='linear', kernel_initializer='he_uniform'))
        q_net.compile(optimizer="Adam",
                      loss='mse')
        return q_net
    def random_policy(self, state):
        """
        create a random action
        :param state:
        :return: action
        """
        return np.random.randint(0,2)

    def collect_policy(self, state):
        """
        Policy with some random actions to encourage exploration
        :param state: game state
        :return: action
        """
        if np.random.random()<0.3:
            return self.random_policy(state)
        return self.policy(state)
    def policy(self, state):
        """
        takes state and returns action based on output of q_net i.e. action that has highest predicted q value
        :param state:
        :return: action
        """
        state_input = convert_to_tensor(state[None, :], dtype='float32')
        action_q = self.q_net(state_input)
        action = np.argmax(action_q.numpy()[0], axis= 0)
        return action
    def update_target_network(self):
        """
        Update target network with batch of gameplay
        :return: None
        """
        self.target_q_net.set_weights(self.q_net.get_weights())
    def train(self, batch):
        """
        Trains the underlying network with batch of gameplays
        :param batch: batch size
        :return:loss
        """
        state_batch, next_state_batch, action_batch, reward_batch, done_batch = batch
        current_q = self.q_net(state_batch).numpy()
        target_q = np.copy(current_q)
        next_q = self.target_q_net(next_state_batch).numpy()
        max_next_q = np.amax(next_q, axis = 1)
        for i in range(state_batch.shape[0]):
            target_q_val = reward_batch[i]
            if not done_batch[i]:
                target_q_val += 0.95 * max_next_q[i]
            target_q[i][action_batch[i]] = target_q_val
        training_history = self.q_net.fit(x=state_batch, y = target_q, verbose = 0)
        loss = training_history.history['loss']
        return loss