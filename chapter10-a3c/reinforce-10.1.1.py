"""Trains a Policy Estimator to solve
MountainCarCountinuous-v0 problem


"""

from keras.layers import Dense, Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.layers.merge import concatenate
import tensorflow as tf

from collections import deque
import numpy as np
import random
import argparse
import gym
from gym import wrappers, logger


class PolicyAgent():
    def __init__(self, env):

        self.env = env
        # experience buffer
        self.memory = []

        # discount rate
        self.gamma = 0.99
        self.action = 0.0
        self.log_prob = 0.0

        # weights filename
        self.weights_file = 'reinforce_mountaincarcontinuous.h5'
        n_inputs = env.observation_space.shape[0]
        self.pi_model = self.build_model(n_inputs)
        self.pi_model.compile(loss=self.loss, optimizer=Adam(lr=1e-5))

    def reset_memory(self):
        self.memory = []
    
    def loss(self, y_true, y_pred):

        # loss = -y_pred[0][1] * y_true[0][1]
        # mean = y_pred[0][0]
        # stddev = y_pred[0][1]
        # action = self.action_sample([mean, stddev])
        # action_log_prob = self.action_log_prob([mean, stddev, self.action])
        loss = -y_pred[0][0][1] * y_true[0][0][1]

        discount_rate = self.gamma**y_true[0][0][0]
        loss *= discount_rate
        return loss

    def action_sample(self, args):
        mean, stddev = args
        dist = tf.distributions.Normal(loc=mean, scale=stddev)
        action = dist.sample(1)
        action = K.clip(action, self.env.action_space.low[0], self.env.action_space.high[0])
        log_prob = dist.log_prob(action)
        outputs = concatenate([action, log_prob])
        return outputs

    def action_log_prob(self, args):
        mean, stddev, action = args
        dist = tf.distributions.Normal(loc=mean, scale=stddev)
        log_prob = dist.log_prob(action)
        return log_prob

    def build_model(self, n_inputs):
        inputs = Input(shape=(n_inputs, ), name='state')
        x = Dense(64, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        mean = Dense(1, activation='linear', name='action_mean')(x)
        stddev = Dense(1, activation='softplus', name='action_stddev')(x)
        # action = Lambda(self.action_sample, output_shape=(1,), name='action')([mean, stddev])
        # log_prob = Lambda(self.action_log_prob, output_shape=(1,), name='action_log_prob')([mean, stddev, action])
        # outputs = concatenate([mean, stddev])
        outputs = Lambda(self.action_sample, name='action')([mean, stddev])
        model = Model(inputs, outputs)
        model.summary()
        return model


    def act(self, state):
        outputs = self.pi_model.predict(state)
        action = outputs[0][0][0]
        # stddev = outputs[0][1]
        # action = self.action_sample([mean, stddev])
        # self.log_prob = self.action_log_prob([mean, stddev, action])
        # action = K.eval(action)
        # print(action)
        return [action]

    def remember(self, item):
        self.memory.append(item)

    def train_once(self, t, state, reward):
        target = np.array([t, reward])
        target = np.reshape(target, [1, 1, 2])
        if t == 998:
            verbose = 1
        else:
            verbose = 0
        self.pi_model.fit(np.array(state),
                          target,
                          batch_size=1,
                          epochs=1,
                          verbose=verbose)

    def train(self):
        for item in self.memory:
            t, state, reward = item
            target = np.array([t, reward])
            target = np.reshape(target, [1, 2])
            self.pi_model.fit(np.array(state),
                              target,
                              batch_size=1,
                              epochs=1,
                              verbose=0)


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
    agent = PolicyAgent(env)

    # should be solved in this number of episodes
    episode_count = 1000
    state_size = env.observation_space.shape[0]

    # sampling and fitting
    for episode in range(episode_count):
        state = env.reset()
        # state is car [position, speed]
        state = np.reshape(state, [1, state_size])
        done = False
        agent.reset_memory()
        t = 0
        total_reward = 0
        while not done:
            # [min, max] action = [-1.0, 1.0]
            action = agent.act(state)
            env.render()
            next_state, reward, done, _ = env.step(action)
            # item = (t, state, reward)
            # agent.remember(item)
            agent.train_once(t, state, reward)
            state = next_state
            state = np.reshape(state, [1, state_size])
            total_reward += reward
            t += 1

        print("Done at t=%d. Action=%f, Reward=%f, Total_Reward=%f" % (t, action[0], reward, total_reward))
        # agent.train()


    # close the env and write monitor result info to disk
    env.close() 
