
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from collections import deque
import numpy as np
import random
import argparse
import logging
import sys
import gym
from gym import wrappers, logger


class DQNAgent(object):
    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space
        self.build_model()
        self.memory = deque()
        self.gamma = 0.9    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999
        
    def build_model(self):
        inputs = Input(shape=(self.state_space.shape[0], ), name='state')
        x = Dense(128, activation='relu')(inputs)
        x = Dense(128, activation='relu')(x)
        x = Dense(self.action_space.n, activation='linear', name='action')(x)
        self.model = Model(inputs, x)
        self.model.compile(loss='mse', optimizer=Adam())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # explore
            act = self.action_space.sample()
            return act

        # exploit
        act = self.model.predict(state)
        act = np.argmax(act[0])
        return act

    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    def replay(self, batch_size):
        mini_batch = random.sample(self.memory, batch_size)
        x_batch, y_batch = [], []
        for state, action, reward, next_state, done in mini_batch:
            y_target = self.model.predict(state)
            q_value = self.gamma * np.amax(self.model.predict(next_state)[0])
            y_target[0][action] = reward if done else reward + q_value
            x_batch.append(state[0])
            y_batch.append(y_target[0])

        self.model.fit(np.array(x_batch),
                       np.array(y_batch),
                       batch_size=batch_size,
                       epochs=1,
                       verbose=0)
        self.update_epsilon()

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id',
                        nargs='?',
                        default='CartPole-v0',
                        help='Select the environment to run')
    args = parser.parse_args()

    win_trials = 100
    win_reward = { 'CartPole-v0' : 195.0 }
    scores = deque(maxlen=win_trials)

    # You can set the level to logging.DEBUG or logging.WARN if you
    # want to change the amount of output.
    logger.setLevel(logging.ERROR)

    env = gym.make(args.env_id)

    outdir = "/tmp/dqn-%s" % args.env_id
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = DQNAgent(env.observation_space, env.action_space)

    episode_count = 3000
    state_size = env.observation_space.shape[0]
    batch_size = 4

    for i in range(episode_count):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        t = 0
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

        scores.append(t)
        mean_score = np.mean(scores)
        if mean_score >= win_reward[args.env_id] and i >= win_trials:
            print("Solved after %d episodes" % i)
            exit(0)
        if i % win_trials == 0:
            print("Episode %d: Mean survival in the last %d episodes: %0.2lf" %
                  (i, win_trials, mean_score))
                  
        if len(agent.memory) >= batch_size:
            agent.replay(batch_size)
            
    # close the env and write monitor result info to disk
    env.close() 
