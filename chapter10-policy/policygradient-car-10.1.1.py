"""Trains a Policy Estimator to solve
MountainCarCountinuous-v0 problem


"""

from keras.layers import Dense, Input, Lambda, Activation
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.layers.merge import concatenate
from keras.models import load_model

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

        # discount rate
        self.gamma = 0.99
        self.state = env.reset()
        state_size = env.observation_space.shape[0]
        self.state = np.reshape(self.state, [1, state_size])

        n_inputs = env.observation_space.shape[0]
        self.actor_model, self.logprob_model, self.entropy_model, self.value_model = self.build_models(n_inputs)
        loss = self.logprob_loss(self.get_entropy(self.state))
        self.logprob_model.compile(loss=loss, optimizer=Adam(lr=1e-3))
        self.value_model.compile(loss=self.value_loss, optimizer=Adam(lr=1e-6, clipvalue=0.5))


    def logprob_loss(self, entropy, beta=0.0):
        def loss(y_true, y_pred):
            return K.mean((-y_pred * y_true) - (beta * entropy), axis=-1)

        return loss


    def value_loss(self, y_true, y_pred):
        return K.mean(-y_pred * y_true, axis=-1)


    def action(self, args):
        mean, stddev = args
        dist = tf.distributions.Normal(loc=mean, scale=stddev)
        action = dist.sample(1)
        action = K.clip(action, self.env.action_space.low[0], self.env.action_space.high[0])
        return action


    def logprob(self, args):
        mean, stddev, action = args
        dist = tf.distributions.Normal(loc=mean, scale=stddev)
        logprob = dist.log_prob(action)
        return logprob


    def entropy(self, args):
        mean, stddev = args
        dist = tf.distributions.Normal(loc=mean, scale=stddev)
        entropy = dist.entropy()
        return entropy


    def build_models(self, n_inputs):
        inputs = Input(shape=(n_inputs, ), name='state')
        kernel_initializer = 'zeros'
        x = Dense(256, activation='relu', kernel_initializer=kernel_initializer)(inputs)
        x = Dense(256, activation='relu', kernel_initializer=kernel_initializer)(x)
        x = Dense(256, activation='relu', kernel_initializer=kernel_initializer)(x)

        value = Dense(128, activation='relu', kernel_initializer=kernel_initializer)(x)
        value = Dense(1, activation='linear', name='value', kernel_initializer=kernel_initializer)(value)
        value_model = Model(inputs, value)
        value_model.summary()

        mean = Dense(1, activation='linear', kernel_initializer=kernel_initializer, name='action_mean')(x)
        stddev = Dense(1, activation='softplus', kernel_initializer=kernel_initializer, name='action_stddev')(x)
        action = Lambda(self.action, output_shape=(1,), name='action')([mean, stddev])
        actor_model = Model(inputs, action)
        actor_model.summary()

        logprob = Lambda(self.logprob, output_shape=(1,), name='logprob')([mean, stddev, action])
        logprob_model = Model(inputs, logprob)
        logprob_model.summary()

        entropy = Lambda(self.action, output_shape=(1,), name='entropy')([mean, stddev])
        entropy_model = Model(inputs, entropy)
        entropy_model.summary()

        return actor_model, logprob_model, entropy_model, value_model


    def save_actor_weights(self, filename):
        self.actor_model.save_weights(filename)


    def load_actor_weights(self, filename):
        self.actor_model.load_weights(filename)


    def act(self, state):
        action = self.actor_model.predict(state)
        return action[0]


    def value(self, state):
        value = self.value_model.predict(state)
        return value[0]


    def get_entropy(self, state):
        entropy = self.entropy_model.predict(state)
        return entropy[0]


    def train(self, step, state, next_state, reward):
        self.state = state
        discount_factor = self.gamma**step
        delta = reward
        if self.args.baseline:
            delta -= self.value(state)[0] 
        elif self.args.actor_critic:
            next_value = self.value(next_state)[0]
            delta += self.gamma*next_value
            delta -= self.value(state)[0] 
        discounted_delta = delta * discount_factor
        discounted_delta = np.reshape(discounted_delta, [1, 1])
        verbose = 1 if step == 0 else 0
        if self.args.baseline or self.args.actor_critic:
            self.value_model.fit(np.array(state),
                                 discounted_delta,
                                 batch_size=1,
                                 epochs=1,
                                 verbose=verbose)
        self.logprob_model.fit(np.array(state),
                               discounted_delta,
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
    parser.add_argument("-r",
                        "--random",
                        action='store_true',
                        help="Random action policy")
    parser.add_argument("-w",
                        "--weights",
                        help="Load pre-trained actor model weights")
    args = parser.parse_args()

    logger.setLevel(logger.ERROR)
    env = gym.make(args.env_id)

    postfix = 'reinforce'
    if args.baseline:
        postfix = "reinforce-baseline"
    elif args.actor_critic:
        postfix = "actor-critic"
    elif args.random:
        postfix = "random"

    filename = "actor_weights-%s-%s.h5" % (postfix, args.env_id)
    outdir = "/tmp/%s-%s" % (postfix, args.env_id)
    csvfilename = "%s.csv" % postfix
    csvfile = open(csvfilename, 'w', 1)
    writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    writer.writerow(['Episode','Step','Total Reward','Number of Episodes Solved'])

    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    
    # instantiate agent
    agent = PolicyAgent(env, args)
    train = True
    if args.weights:
        agent.load_actor_weights(args.weights)
        train = False

    # should be solved in this number of episodes
    episode_count = 1000
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
            if args.random:
                action = env.action_space.sample()
            else:
                action = agent.act(state)
            env.render()
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            total_reward += reward
            # if done:
            #    step += 1
            #    break
            if not args.random and train:
                agent.train(step, state, next_state, reward)
            state = next_state
            step += 1

        if reward > 0:
            n_solved += 1
        fmt = "Episode=%d, Step=%d. Action=%f, Reward=%f, Total_Reward=%f"
        print(fmt % (episode, step, action[0], reward, total_reward))
        writer.writerow([episode, step, total_reward, n_solved])

    if not args.weights and not args.random:
        agent.save_actor_weights(filename)

    # close the env and write monitor result info to disk
    csvfile.close()
    env.close() 
