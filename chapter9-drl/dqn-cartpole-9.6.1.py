"""Trains a DQN/DDQN to solve CartPole-v0 problem


"""

from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from collections import deque
import heapq
import numpy as np
import random
import argparse
import sys
import gym
from gym import wrappers, logger


class DQNAgent(object):
    def __init__(self, state_space, action_space, args):
        self.action_space = action_space
        self.state_space = state_space
        self.memory = []

        # discount rate
        self.gamma = 0.9

        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99

        # Q Network weights filename
        self.weights_file = 'dqn_cartpole.h5'
        # Q Network for training
        self.q_model = self.build_model()
        self.q_model.compile(loss='mse', optimizer=Adam())
        # Target Q Network
        self.target_q_model = self.build_model()
        # copy Q Network params to target Q Network
        self.update_weights()

        self.replay_counter = 0
        self.ddqn = True if args.ddqn else False
        if self.ddqn:
            print("DDQN---------------------------------------------------")
        else:
            print("----------------------------------------------------DQN")

    
    # Q Network is 256-256-256 MLP
    def build_model(self):
        inputs = Input(shape=(self.state_space.shape[0], ), name='state')
        x = Dense(256, activation='relu')(inputs)
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(self.action_space.n, activation='linear', name='action')(x)
        q_model = Model(inputs, x)
        q_model.summary()
        return q_model


    # save Q Network params to a file
    def save_weights(self):
        self.q_model.save_weights(self.weights_file)


    # copy Q Network params to target Q Network
    def update_weights(self):
        self.target_q_model.set_weights(self.q_model.get_weights())


    # eps-greedy policy
    def act(self, state):
        if np.random.rand() < self.epsilon:
            # explore - do random action
            return self.action_space.sample()

        # exploit
        q_values = self.q_model.predict(state)
        # select the action with max Q-value
        return np.argmax(q_values[0])


    # store experiences in the replay buffer
    def remember(self, state, action, reward, next_state, done):
        item = (state, action, reward, next_state, done)
        self.memory.append(item)


    # compute Qmax
    def get_target_q_value(self, next_state):
       # TD(0) Q-value using Bellman equation
       # to deal with non-stationarity, model weights are fixed
        if self.ddqn:
            # DDQN
            # current Q Network selects the action
            action = np.argmax(self.q_model.predict(next_state)[0])
            # target Q Network evaluates the action
            q_value = self.target_q_model.predict(next_state)[0][action]
        else:
            # DQN chooses the Q value of the action with max value
            q_value = np.amax(self.target_q_model.predict(next_state)[0])

        q_value *= self.gamma
        q_value += reward
        return q_value


    # experience replay addresses the correlation issue between samples
    def replay(self, batch_size):
        # sars = state, action, reward, state' (next_state)
        sars_batch = random.sample(self.memory, batch_size)
        state_batch, q_values_batch = [], []
        index = 0

        for state, action, reward, next_state, done in sars_batch:
            # policy prediction for a given state
            q_values = self.q_model.predict(state)
            
            q_value = self.get_target_q_value(next_state)

            # correction on the Q-value for the given action
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
        if self.replay_counter % 4 == 0:
            self.update_epsilon()

        # copy new params on old target after every x training updates
        if self.replay_counter % 2 == 0:
            self.update_weights()

        self.replay_counter += 1

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        

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
    args = parser.parse_args()

    if args.ddqn:
        print("Using DDQN")
    else:
        print("Using default DQN")


    # the number of trials without falling over
    win_trials = 100

    # the CartPole-v0 is considered solved if for 100 consecutive trials,
    # the cart pole has not fallen over and it has achieved an average 
    # reward of 195.0
    # a reward of +1 is provided for every timestep the pole remains
    # upright
    win_reward = { 'CartPole-v0' : 195.0 }
    # stores the reward per episode
    scores = deque(maxlen=win_trials)

    logger.setLevel(logger.ERROR)
    env = gym.make(args.env_id)

    outdir = "/tmp/dqn-%s" % args.env_id
    if args.ddqn:
        outdir = "/tmp/ddqn-%s" % args.env_id

    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = DQNAgent(env.observation_space, env.action_space, args)

    episode_count = 3000
    state_size = env.observation_space.shape[0]
    batch_size = 64

    # by default, CartPole-v0 has max episode steps = 200
    # you can use this to experiment beyond 200
    # env._max_episode_steps = 4000

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
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        if len(agent.memory) >= batch_size:
            agent.replay(batch_size)

        scores.append(total_reward)
        mean_score = np.mean(scores)
        if mean_score >= win_reward[args.env_id] and episode >= win_trials:
            print("Solved in episode %d: Mean survival = %0.2lf in %d episodes"
                  % (episode, mean_score, win_trials))
            print("Epsilon: ", agent.epsilon)
            agent.save_weights()
            break
        if episode % win_trials == 0:
            print("Episode %d: Mean survival = %0.2lf in %d episodes" %
                  (episode, mean_score, win_trials))

    # close the env and write monitor result info to disk
    env.close() 
