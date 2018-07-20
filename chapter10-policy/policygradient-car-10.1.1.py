"""Trains a policy network to solve
MountainCarCountinuous-v0 problem


"""

from keras.layers import Dense, Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

import tensorflow as tf

import numpy as np
import argparse
import gym
from gym import wrappers, logger
import sys
import csv


class PolicyAgent():
    def __init__(self, env, args):

        self.env = env
        self.args = args

        self.memory = []
        self.state = env.reset()
        state_size = env.observation_space.shape[0]
        self.state = np.reshape(self.state, [1, state_size])

        actor, logp, entropy, value = self.build_models(state_size)
        self.actor_model = actor
        self.logp_model = logp
        self.entropy_model = entropy
        self.value_model = value
        beta = 0.01 if self.args.a2c else 0.0
        loss = self.logp_loss(self.get_entropy(self.state))
        self.logp_model.compile(loss=loss,
                                   optimizer=Adam(lr=1e-3))
        loss = 'mse' if self.args.a2c else self.value_loss
        self.value_model.compile(loss=loss,
                                 optimizer=Adam(lr=1e-5))


    def reset_memory(self):
        self.memory = []


    def remember(self, item):
        self.memory.append(item)


    def logp_loss(self, entropy, beta=0.0):
        def loss(y_true, y_pred):
            return K.mean((-y_pred * y_true) - (beta * entropy), axis=-1)

        return loss


    def value_loss(self, y_true, y_pred):
        return K.mean(-y_pred * y_true, axis=-1)


    def action(self, args):
        mean, stddev = args
        dist = tf.distributions.Normal(loc=mean, scale=stddev)
        action = dist.sample(1)
        action = K.clip(action,
                        self.env.action_space.low[0],
                        self.env.action_space.high[0])
        return action


    def logp(self, args):
        mean, stddev, action = args
        dist = tf.distributions.Normal(loc=mean, scale=stddev)
        logp = dist.log_prob(action)
        return logp


    def entropy(self, args):
        mean, stddev = args
        dist = tf.distributions.Normal(loc=mean, scale=stddev)
        entropy = dist.entropy()
        return entropy


    def build_models(self, n_inputs):
        inputs = Input(shape=(n_inputs, ), name='state')
        kernel_initializer = 'zeros'
        x = Dense(256,
                  activation='relu',
                  kernel_initializer=kernel_initializer)(inputs)
        x = Dense(256,
                  activation='relu',
                  kernel_initializer=kernel_initializer)(x)
        x = Dense(256,
                  activation='relu',
                  kernel_initializer=kernel_initializer)(x)

        value = Dense(128,
                      activation='relu',
                      kernel_initializer=kernel_initializer)(x)
        value = Dense(1,
                      activation='linear',
                      name='value',
                      kernel_initializer=kernel_initializer)(value)
        value_model = Model(inputs, value)
        value_model.summary()

        mean = Dense(1,
                     activation='linear',
                     kernel_initializer=kernel_initializer,
                     name='mean')(x)
        stddev = Dense(1,
                       activation='softplus',
                       kernel_initializer=kernel_initializer,
                       name='stddev')(x)
        action = Lambda(self.action,
                        output_shape=(1,),
                        name='action')([mean, stddev])
        actor_model = Model(inputs, action)
        actor_model.summary()

        logp = Lambda(self.logp,
                         output_shape=(1,),
                         name='logp')([mean, stddev, action])
        logp_model = Model(inputs, logp)
        logp_model.summary()

        entropy = Lambda(self.entropy,
                         output_shape=(1,),
                         name='entropy')([mean, stddev])
        entropy_model = Model(inputs, entropy)
        entropy_model.summary()

        return actor_model, logp_model, entropy_model, value_model


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


    def train_by_episode(self, last_value=0):
        gamma = 1.0
        r = last_value
        i = len(self.memory) - 1 
        for item in self.memory[::-1]:
            step, state, next_state, reward, done = item
            r = reward + gamma*r 
            discounted_item = (step, state, next_state, r, done)
            self.memory[i] = item
            i -= 1

        for item in self.memory:
            self.train(item)


    def train(self, item):
        step, state, next_state, reward, done = item
        self.state = state
        gamma = 1.0
        discount_factor = gamma**step
        delta = reward
        critic = False
        if self.args.baseline:
            critic = True
            delta -= self.value(state)[0] 
        elif self.args.actor_critic or self.args.a2c:
            critic = True
            if done:
                next_value = 0.0
            else:
                next_value = self.value(next_state)[0]
            delta += gamma*next_value
            # if actor-critic,
            # delta is a multiplier to loss not a target
            if self.args.actor_critic:
                delta -= self.value(state)[0] 

        discounted_delta = delta * discount_factor
        discounted_delta = np.reshape(discounted_delta, [-1, 1])
        verbose = 1 if done else 0
        if critic:
            self.value_model.fit(np.array(state),
                                 discounted_delta,
                                 batch_size=1,
                                 epochs=1,
                                 verbose=verbose)
        self.logp_model.fit(np.array(state),
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
    parser.add_argument("-c",
                        "--a2c",
                        action='store_true',
                        help="Advantage-Actor-Critic")
    parser.add_argument("-r",
                        "--random",
                        action='store_true',
                        help="Random action policy")
    parser.add_argument("-e",
                        "--train_by_episode",
                        action='store_true',
                        help="Complete 1 episode before training")
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
    elif args.a2c:
        postfix = "a2c"
    elif args.random:
        postfix = "random"

    by_episode = False
    if args.train_by_episode:
        print("Train by episode.....")
        postfix += "-episode" 
        by_episode = True

    filename = "actor_weights-%s-%s.h5" % (postfix, args.env_id)
    outdir = "/tmp/%s-%s" % (postfix, args.env_id)
    csvfilename = "%s.csv" % postfix
    csvfile = open(csvfilename, 'w', 1)
    writer = csv.writer(csvfile,
                        delimiter=',',
                        quoting=csv.QUOTE_NONNUMERIC)
    writer.writerow(['Episode',
                    'Step',
                    'Total Reward',
                    'Number of Episodes Solved'])

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
        if by_episode:
            agent.reset_memory()
        while not done:
            # [min, max] action = [-1.0, 1.0]
            if args.random:
                action = env.action_space.sample()
            else:
                action = agent.act(state)
            env.render()
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            item = (step, state, next_state, reward, done)
            if by_episode:
                agent.remember(item)
            total_reward += reward
            if not args.random and train:
                if by_episode and done:
                    v = 0 if reward > 0 else agent.value(next_state)[0]
                    agent.train_by_episode(last_value=v)
                else:
                    agent.train(item)
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
