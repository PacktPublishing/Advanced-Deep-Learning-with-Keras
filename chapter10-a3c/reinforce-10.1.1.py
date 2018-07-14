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
import sys
import csv


class PolicyAgent():
    def __init__(self, env, args):

        self.env = env
        self.args = args
        # experience buffer
        self.memory = []

        # discount rate
        self.gamma = 0.99

        # weights filename
        self.weights_file = 'reinforce_mntcar.h5'
        n_inputs = env.observation_space.shape[0]
        self.action_model, self.log_prob_model, self.value_model = self.build_model(n_inputs)
        self.log_prob_model.compile(loss=self.loss, optimizer=Adam(lr=1e-5))
        self.value_model.compile(loss='mse', optimizer=Adam(lr=1e-5))

    def reset_memory(self):
        self.memory = []
   
    def loss(self, y_true, y_pred):
        loss = K.mean(-y_pred * y_true)
        return loss

    def action_sample(self, args):
        mean, stddev = args
        dist = tf.distributions.Normal(loc=mean, scale=stddev)
        action = dist.sample(1)
        action = K.clip(action, self.env.action_space.low[0], self.env.action_space.high[0])
        # log_prob = dist.log_prob(action)
        # outputs = concatenate([action, log_prob])
        return action

    def action_log_prob(self, args):
        mean, stddev, action = args
        dist = tf.distributions.Normal(loc=mean, scale=stddev)
        log_prob = dist.log_prob(action)
        return log_prob

    def build_model(self, n_inputs):
        inputs = Input(shape=(n_inputs, ), name='state')
        x = Dense(256, activation='relu')(inputs)
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        y = Dense(128, activation='relu')(x)
        value = Dense(1, activation='linear', name='value')(y)
        mean = Dense(1, activation='linear', name='action_mean')(x)
        stddev = Dense(1, activation='softplus', name='action_stddev')(x)
        action = Lambda(self.action_sample, output_shape=(1,), name='action')([mean, stddev])
        log_prob = Lambda(self.action_log_prob, output_shape=(1,), name='action_log_prob')([mean, stddev, action])
        action_model = Model(inputs, action)
        action_model.summary()
        log_prob_model = Model(inputs, log_prob)
        log_prob_model.summary()
        value_model = Model(inputs, value)
        value_model.summary()
        return action_model, log_prob_model, value_model

    def act(self, state):
        action = self.action_model.predict(state)
        return action[0]

    def value(self, state):
        value = self.value_model.predict(state)
        return value[0]

    def remember(self, item):
        self.memory.append(item)

    def train(self, step, state, next_state, reward):
        discount_factor = self.gamma**step
        if self.args.baseline:
            value_target = reward
            value_target = np.reshape(value_target, [1, 1])
            reward -= self.value(state)[0] 
        elif self.args.actor_critic:
            next_value = self.value(next_state)[0]
            value_target = reward + self.gamma*next_value
            reward = value_target - self.value(state)[0] 
            value_target = np.reshape(value_target, [1, 1])
        # reward *= discount_factor
        target = np.array([reward])
        target = np.reshape(target, [1, 1])
        if step == 997:
            verbose = 1
        else:
            verbose = 0
        self.log_prob_model.fit(np.array(state),
                                target,
                                batch_size=1,
                                epochs=1,
                                verbose=verbose)

        if self.args.baseline or self.args.actor_critic:
            self.value_model.fit(np.array(state),
                                 value_target,
                                 batch_size=1,
                                 epochs=1,
                                 verbose=verbose)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id',
                        nargs='?',
                        default='MountainCarContinuous-v0',
                        help='Select the environment to run')
    parser.add_argument("-b",
                        "--baseline",
                        action='store_true',
                        help="Reinforce with baseline")
    parser.add_argument("-a",
                        "--actor_critic",
                        action='store_true',
                        help="Actor-Critic")
    args = parser.parse_args()
    args = parser.parse_args()

    logger.setLevel(logger.ERROR)
    env = gym.make(args.env_id)

    postfix = ''
    if args.baseline:
        postfix = "-baseline"
    elif args.actor_critic:
        postfix = "-actor-critic"
    outdir = "/tmp/reinforce%s-%s" % (postfix, args.env_id)
    csvfilename = "reinforce%s.csv" % postfix
    csvfile = open(csvfilename, 'w', 1)
    writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    writer.writerow(['Episode','Step','Total Reward','Number of Episodes Solved'])

    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    # print(env._max_episode_steps)

    # instantiate agent
    agent = PolicyAgent(env, args)

    # should be solved in this number of episodes
    episode_count = 100
    state_size = env.observation_space.shape[0]
    n_solved = 0 

    # sampling and fitting
    for episode in range(episode_count):
        state = env.reset()
        # state is car [position, speed]
        state = np.reshape(state, [1, state_size])
        step = 0
        total_reward = 0
        done = False
        while not done:
            # [min, max] action = [-1.0, 1.0]
            action = agent.act(state)
            env.render()
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            total_reward += reward
            if done:
                if not args.baseline and not args.actor_critic:
                    step += 1
                    break
            agent.train(step, state, next_state, reward)
            state = next_state
            step += 1

        if reward > 0:
            n_solved += 1
        fmt = "Episode=%d, Step=%d. Action=%f, Reward=%f, Total_Reward=%f"
        print(fmt % (episode, step, action[0], reward, total_reward))
        writer.writerow([episode, step, total_reward, n_solved])

    # close the env and write monitor result info to disk
    csvfile.close()
    env.close() 
