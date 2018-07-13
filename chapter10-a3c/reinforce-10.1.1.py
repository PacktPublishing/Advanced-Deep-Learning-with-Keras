"""Trains a Policy Estimator to solve
MountainCarCountinuous-v0 problem


"""

from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

from scipy.stats import norm
from collections import deque
import numpy as np
import random
import argparse
import gym
from gym import wrappers, logger


class PolicyAgent():
    def __init__(self, state_space, action_space, args, episodes=500):

        self.action_space = action_space

        # experience buffer
        self.memory = []

        # discount rate
        self.gamma = 0.9
        self.t = 0
        self.action = 0
        self.action_prob = 0

        # weights filename
        self.weights_file = 'reinforce_mountaincarcontinuous.h5'
        n_inputs = state_space.shape[0]
        n_outputs = action_space.shape[0]
        self.pi_model = self.build_model(n_inputs, n_outputs)
        self.pi_model.compile(loss=self.loss, optimizer=Adam())

    def reset_time(self):
        self.t = 0
    
    def loss(self, action_mean, reward):
        discount_rate = self.gamma**self.t
        loss = -K.log(self.action_prob + K.epsilon()) * reward
        loss *= discount_rate
        self.t += 1
        return loss

    # network is 256-256-256 MLP
    def build_model(self, n_inputs, n_outputs):
        inputs = Input(shape=(n_inputs, ), name='state')
        x = Dense(256, activation='relu')(inputs)
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(n_outputs, activation='linear', name='action')(x)
        model = Model(inputs, x)
        model.summary()
        return model


    # policy
    def action_sample(self, state):
        mean = self.pi_model.predict(state)
        stddev = 1.0
        self.action = np.random.normal(loc=mean, scale=stddev)
        self.action_prob = norm.pdf(self.action, mean[0][0], 1.0)
        return self.action


    def remember(self, reward):
        self.memory.append(reward)


    def train(self, state, reward):
        # self.pi_model.train_on_batch(np.array(state), np.array([reward]))
        # return
        self.pi_model.fit(np.array(state),
                         np.array([reward]),
                         batch_size=1,
                         epochs=1,
                         verbose=1)


def sample(mean=0.0, stddev=1.0):
    sample = K.random_normal(shape=(1, 1), mean=mean, stddev=stddev)
    return sample

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id',
                        nargs='?',
                        default='MountainCarContinuous-v0',
                        help='Select the environment to run')
    args = parser.parse_args()

    logger.setLevel(logger.ERROR)
    env = gym.make(args.env_id)

    outdir = "/tmp/reinforce-%s" % args.env_id

    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)

    # instantiate agent
    agent = PolicyAgent(env.observation_space, env.action_space, args)

    # should be solved in this number of episodes
    episode_count = 3000
    state_size = env.observation_space.shape[0]
    batch_size = 64

    # sampling and fitting
    for episode in range(episode_count):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        agent.reset_time()
        t = 0
        while not done:
            action = agent.action_sample(state)
            # print(action)
            env.render()
            next_state, reward, done, _ = env.step(action)
            agent.train(state, reward)
            state = next_state
            state = np.reshape(state, [1, state_size])
            t += 1
            # in -v0:
            # state = [pos, vel, theta, angular speed]

        print("Done at t=%d" % t)


    # close the env and write monitor result info to disk
    env.close() 
