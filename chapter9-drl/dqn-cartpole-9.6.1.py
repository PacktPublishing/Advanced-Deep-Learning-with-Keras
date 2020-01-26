"""Trains a DQN/DDQN to solve CartPole-v0 problem


"""

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from collections import deque
import numpy as np
import random
import argparse
import gym
from gym import wrappers, logger

class DQNAgent:
    def __init__(self,
                 state_space, 
                 action_space, 
                 episodes=500):
        """DQN Agent on CartPole-v0 environment

        Arguments:
            state_space (tensor): state space
            action_space (tensor): action space
            episodes (int): number of episodes to train
        """
        self.action_space = action_space

        # experience buffer
        self.memory = []

        # discount rate
        self.gamma = 0.9

        # initially 90% exploration, 10% exploitation
        self.epsilon = 1.0
        # iteratively applying decay til 
        # 10% exploration/90% exploitation
        self.epsilon_min = 0.1
        self.epsilon_decay = self.epsilon_min / self.epsilon
        self.epsilon_decay = self.epsilon_decay ** \
                             (1. / float(episodes))

        # Q Network weights filename
        self.weights_file = 'dqn_cartpole.h5'
        # Q Network for training
        n_inputs = state_space.shape[0]
        n_outputs = action_space.n
        self.q_model = self.build_model(n_inputs, n_outputs)
        self.q_model.compile(loss='mse', optimizer=Adam())
        # target Q Network
        self.target_q_model = self.build_model(n_inputs, n_outputs)
        # copy Q Network params to target Q Network
        self.update_weights()

        self.replay_counter = 0

    
    def build_model(self, n_inputs, n_outputs):
        """Q Network is 256-256-256 MLP

        Arguments:
            n_inputs (int): input dim
            n_outputs (int): output dim

        Return:
            q_model (Model): DQN
        """
        inputs = Input(shape=(n_inputs, ), name='state')
        x = Dense(256, activation='relu')(inputs)
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(n_outputs,
                  activation='linear', 
                  name='action')(x)
        q_model = Model(inputs, x)
        q_model.summary()
        return q_model


    def save_weights(self):
        """save Q Network params to a file"""
        self.q_model.save_weights(self.weights_file)


    def update_weights(self):
        """copy trained Q Network params to target Q Network"""
        self.target_q_model.set_weights(self.q_model.get_weights())


    def act(self, state):
        """eps-greedy policy
        Return:
            action (tensor): action to execute
        """
        if np.random.rand() < self.epsilon:
            # explore - do random action
            return self.action_space.sample()

        # exploit
        q_values = self.q_model.predict(state)
        # select the action with max Q-value
        action = np.argmax(q_values[0])
        return action


    def remember(self, state, action, reward, next_state, done):
        """store experiences in the replay buffer
        Arguments:
            state (tensor): env state
            action (tensor): agent action
            reward (float): reward received after executing
                action on state
            next_state (tensor): next state
        """
        item = (state, action, reward, next_state, done)
        self.memory.append(item)


    def get_target_q_value(self, next_state, reward):
        """compute Q_max
           Use of target Q Network solves the 
            non-stationarity problem
        Arguments:
            reward (float): reward received after executing
                action on state
            next_state (tensor): next state
        Return:
            q_value (float): max Q-value computed
        """
        # max Q value among next state's actions
        # DQN chooses the max Q value among next actions
        # selection and evaluation of action is 
        # on the target Q Network
        # Q_max = max_a' Q_target(s', a')
        q_value = np.amax(\
                     self.target_q_model.predict(next_state)[0])

        # Q_max = reward + gamma * Q_max
        q_value *= self.gamma
        q_value += reward
        return q_value


    def replay(self, batch_size):
        """experience replay addresses the correlation issue 
            between samples
        Arguments:
            batch_size (int): replay buffer batch 
                sample size
        """
        # sars = state, action, reward, state' (next_state)
        sars_batch = random.sample(self.memory, batch_size)
        state_batch, q_values_batch = [], []

        # fixme: for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for state, action, reward, next_state, done in sars_batch:
            # policy prediction for a given state
            q_values = self.q_model.predict(state)
            
            # get Q_max
            q_value = self.get_target_q_value(next_state, reward)

            # correction on the Q value for the action used
            q_values[0][action] = reward if done else q_value

            # collect batch state-q_value mapping
            state_batch.append(state[0])
            q_values_batch.append(q_values[0])

        # train the Q-network
        self.q_model.fit(np.array(state_batch),
                         np.array(q_values_batch),
                         batch_size=batch_size,
                         epochs=1,
                         verbose=0)

        # update exploration-exploitation probability
        self.update_epsilon()

        # copy new params on old target after 
        # every 10 training updates
        if self.replay_counter % 10 == 0:
            self.update_weights()

        self.replay_counter += 1

    
    def update_epsilon(self):
        """decrease the exploration, increase exploitation"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        

class DDQNAgent(DQNAgent):
    def __init__(self,
                 state_space, 
                 action_space, 
                 episodes=500):
        super().__init__(state_space, 
                         action_space, 
                         episodes)
        """DDQN Agent on CartPole-v0 environment

        Arguments:
            state_space (tensor): state space
            action_space (tensor): action space
            episodes (int): number of episodes to train
        """

        # Q Network weights filename
        self.weights_file = 'ddqn_cartpole.h5'
        print("-------------DDQN------------")

    def get_target_q_value(self, next_state, reward):
        """compute Q_max
           Use of target Q Network solves the 
            non-stationarity problem
        Arguments:
            reward (float): reward received after executing
                action on state
            next_state (tensor): next state
        Returns:
            q_value (float): max Q-value computed
        """
        # max Q value among next state's actions
        # DDQN
        # current Q Network selects the action
        # a'_max = argmax_a' Q(s', a')
        action = np.argmax(self.q_model.predict(next_state)[0])
        # target Q Network evaluates the action
        # Q_max = Q_target(s', a'_max)
        q_value = self.target_q_model.predict(\
                                      next_state)[0][action]

        # Q_max = reward + gamma * Q_max
        q_value *= self.gamma
        q_value += reward
        return q_value



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id',
                        nargs='?',
                        default='CartPole-v0',
                        help='Select the environment to run')
    parser.add_argument("-d",
                        "--ddqn",
                        action='store_true',
                        help="Use Double DQN")
    parser.add_argument("-r",
                        "--no-render",
                        action='store_true',
                        help="Disable rendering (for env w/o graphics")
    args = parser.parse_args()

    # the number of trials without falling over
    win_trials = 100

    # the CartPole-v0 is considered solved if 
    # for 100 consecutive trials, he cart pole has not 
    # fallen over and it has achieved an average 
    # reward of 195.0 
    # a reward of +1 is provided for every timestep 
    # the pole remains upright
    win_reward = { 'CartPole-v0' : 195.0 }

    # stores the reward per episode
    scores = deque(maxlen=win_trials)

    logger.setLevel(logger.ERROR)
    env = gym.make(args.env_id)

    outdir = "/tmp/dqn-%s" % args.env_id
    if args.ddqn:
        outdir = "/tmp/ddqn-%s" % args.env_id

    if args.no_render:
        env = wrappers.Monitor(env,
                               directory=outdir,
                               video_callable=False,
                               force=True)
    else:
        env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)

    # instantiate the DQN/DDQN agent
    if args.ddqn:
        agent = DDQNAgent(env.observation_space, env.action_space)
    else:
        agent = DQNAgent(env.observation_space, env.action_space)

    # should be solved in this number of episodes
    episode_count = 3000
    state_size = env.observation_space.shape[0]
    batch_size = 64

    # by default, CartPole-v0 has max episode steps = 200
    # you can use this to experiment beyond 200
    # env._max_episode_steps = 4000

    # Q-Learning sampling and fitting
    for episode in range(episode_count):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        total_reward = 0
        while not done:
            # in CartPole-v0, action=0 is left and action=1 is right
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            # in CartPole-v0:
            # state = [pos, vel, theta, angular speed]
            next_state = np.reshape(next_state, [1, state_size])
            # store every experience unit in replay buffer
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward


        # call experience relay
        if len(agent.memory) >= batch_size:
            agent.replay(batch_size)
    
        scores.append(total_reward)
        mean_score = np.mean(scores)
        if mean_score >= win_reward[args.env_id] \
                and episode >= win_trials:
            print("Solved in episode %d: \
                   Mean survival = %0.2lf in %d episodes"
                  % (episode, mean_score, win_trials))
            print("Epsilon: ", agent.epsilon)
            agent.save_weights()
            break
        if (episode + 1) % win_trials == 0:
            print("Episode %d: Mean survival = \
                   %0.2lf in %d episodes" %
                  ((episode + 1), mean_score, win_trials))

    # close the env and write monitor result info to disk
    env.close() 
