"""Trains a DQN/DDQN to solve CartPole-v0 problem


"""

from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam, RMSprop
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
        self.build_model()
        self.memory = []
        self.gamma = 0.9    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99
        self.q_model = self.build_model()
        optimizer = Adam()
        self.weights_file = 'dqn_cartpole.h5'
        self.q_model = self.build_model()
        self.q_model.compile(loss='mse', optimizer=optimizer)
        self.target_q_model = self.build_model()
        self.update_weights()
        self.replay_counter = 0
        self.enable_ddqn = True if args.enable_ddqn else False
        self.prioritized_replay = True if args.prioritized_replay else False
        if self.enable_ddqn:
            print("DDQN---------------------------------------------------")
        else:
            print("----------------------------------------------------DQN")

        if self.prioritized_replay:
            print("PRIORITIZED REPLAY-------------------------------------")
        self.priority = 0 

    def build_model(self):
        inputs = Input(shape=(self.state_space.shape[0], ), name='state')
        x = Dense(256, activation='relu')(inputs)
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(self.action_space.n, activation='linear', name='action')(x)
        q_model = Model(inputs, x)
        q_model.summary()
        return q_model


    def save_weights(self):
        self.q_model.save_weights(self.weights_file)


    def update_weights(self):
        self.target_q_model.set_weights(self.q_model.get_weights())


    def act(self, state):
        if np.random.rand() < self.epsilon:
            # explore - do random action
            return self.action_space.sample()

        # exploit
        q_values = self.q_model.predict(state)
        # select the action with max acc reward (Q-value)
        return np.argmax(q_values[0])


    def get_td_error(self, next_state):
        eps = random.uniform(1e-4, 1e-3)
        q_value = self.get_target_q_value(next_state)
        q_value -= self.q_model.predict(state)[0][action]
        return abs(q_value) + eps


    def remember(self, state, action, reward, next_state, done):
        # self.memory.append([state, action, reward, next_state, done])
        self.priority += 1
        if self.prioritized_replay:
            self.priority = self.get_td_error(next_state)

        item = (self.priority, state, action, reward, next_state, done)
        heapq.heappush(self.memory, item)


    def get_target_q_value(self, next_state):
       # TD(0) Q-value using Bellman equation
       # to deal with non-stationarity, model weights are fixed
        if self.enable_ddqn:
            # DDQN
            # current q network selects the action
            action = np.argmax(self.q_model.predict(next_state)[0])
            # target q network evaluate the action
            q_value = self.target_q_model.predict(next_state)[0][action]
        else:
            # DQN chooses the q value of the action with max value
            q_value = np.amax(self.target_q_model.predict(next_state)[0])

        q_value *= self.gamma
        q_value += reward
        return q_value


    def replay(self, batch_size):
        """Experience replay removes correlation between samples that
        is causing the neural network to diverge
        
        """
        # get a random batch of sars from replay memory
        # sars = state, action, reward, state' (next_state)
        if self.prioritized_replay:
            self.memory = heapq.nlargest(len(self.memory), self.memory, key=lambda m:m[0])
            indexes = np.random.choice(min(len(self.memory), 16*batch_size), batch_size, replace=False)
            sars_batch = []
            for index in indexes:
                sars_batch.append(self.memory[index])
        else:
            sars_batch = random.sample(self.memory, batch_size)

        state_batch, q_values_batch = [], []
        index = 0
        for _, state, action, reward, next_state, done in sars_batch:
            # policy prediction for a given state
            q_values = self.q_model.predict(state)
            
            q_value = self.get_target_q_value(next_state)

            # correction on the Q-value for the given action
            q_values[0][action] = reward if done else q_value

            # collect batch state-q_value mapping
            state_batch.append(state[0])
            q_values_batch.append(q_values[0])

            if self.prioritized_replay:
                priority = self.get_td_error(next_state)
                i = indexes[index]
                self.memory[i] = (priority, state, action, reward, next_state, done)
                index += 1
            
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


    win_trials = 100
    win_reward = { 'CartPole-v0' : 195.0 }
    scores = deque(maxlen=win_trials)

    # You can set the level to logging.DEBUG or logging.WARN if you
    # want to change the amount of output.
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

    # by default, CartPole-v0 has max episode count = 200
    # you can use this to experiment beyond 200
    # env._max_episode_steps = 4000

    for episode in range(episode_count):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        episode = 0
        done = False
        while not done:
            # in CartPole, action=0 is left and action=1 is right
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            # in CartPole:
            # state = [pos, vel, theta, angular speed]
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            t += 1

        if len(agent.memory) >= batch_size:
            agent.replay(batch_size)

        scores.append(t)
        mean_score = np.mean(scores)
        if mean_score >= win_reward[args.env_id] and i >= win_trials:
            print("Solved in episode %d: Mean survival = %0.2lf in %d episodes"
                  % (i, mean_score, win_trials))
            print("Epsilon: ", agent.epsilon)
            agent.save_weights()
            break
        if i % win_trials == 0:
            print("Episode %d: Mean survival = %0.2lf in %d episodes" %
                  (i, mean_score, win_trials))

    # close the env and write monitor result info to disk
    env.close() 
