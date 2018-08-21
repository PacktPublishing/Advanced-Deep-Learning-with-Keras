"""Code implementation of Policy Gradient Methods as solution
to MountainCarCountinuous-v0 problem

Methods implemented:
    1) REINFORCE
    2) REINFORCE with Baseline
    3) Actor-Critic
    4) A2C

References:
    1) Sutton and Barto, Reinforcement Learning: An Introduction
    (2017)

    2) Mnih, et al. Asynchronous Methods for Deep Reinforcement
    Learning. Intl Conf on Machine Learning. 2016

"""

from keras.layers import Dense, Input, Lambda, Activation
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.utils import plot_model

import tensorflow as tf

import numpy as np
import argparse
import gym
from gym import wrappers, logger
import sys
import csv
import time
import os
import datetime
import math


# some implementations use a modified softplus to ensure that
# the stddev is never zero
def softplusk(x):
    return K.softplus(x) + 1e-10


# implements the models and training of Policy Gradient
# Methods
class PolicyAgent():
    def __init__(self, env, args):

        self.env = env
        self.args = args
        
        # s,a,r,s' are stored in memory
        self.memory = []

        # for computation of input size
        self.state = env.reset()
        self.state_dim = env.observation_space.shape[0]
        self.state = np.reshape(self.state, [1, self.state_dim])
        self.build_autoencoder()


    # clear the memory before the start of every episode
    def reset_memory(self):
        self.memory = []


    # remember every s,a,r,s' in every step of the episode
    def remember(self, item):
        self.memory.append(item)


    # given mean and stddev, sample an action, clip and return
    # we assume Gaussian distribution of probability of selecting an
    # action given a state
    def action(self, args):
        mean, stddev = args
        dist = tf.distributions.Normal(loc=mean, scale=stddev)
        action = dist.sample(1)
        action = K.clip(action,
                        self.env.action_space.low[0],
                        self.env.action_space.high[0])
        return action


    # given mean, stddev, and action compute
    # the log probability of the Gaussian distribution
    def logp(self, args):
        mean, stddev, action = args
        dist = tf.distributions.Normal(loc=mean, scale=stddev)
        logp = dist.log_prob(action)
        return logp


    # given the mean and stddev compute the Gaussian dist entropy
    def entropy(self, args):
        mean, stddev = args
        dist = tf.distributions.Normal(loc=mean, scale=stddev)
        entropy = dist.entropy()
        return entropy


    # autoencoder to convert states into features
    def build_autoencoder(self):
        # first build the encoder model
        inputs = Input(shape=(self.state_dim, ), name='state')
        feature_size = 32
        x = Dense(256, activation='relu')(inputs)
        x = Dense(128, activation='relu')(x)
        feature = Dense(feature_size, name='feature_vector')(x)

        # instantiate encoder model
        self.encoder = Model(inputs, feature, name='encoder')
        self.encoder.summary()
        plot_model(self.encoder, to_file='encoder.png', show_shapes=True)

        # build the decoder model
        feature_inputs = Input(shape=(feature_size,), name='decoder_input')
        x = Dense(128, activation='relu')(feature_inputs)
        x = Dense(256, activation='relu')(x)
        outputs = Dense(self.state_dim, activation='linear')(x)

        # instantiate decoder model
        self.decoder = Model(feature_inputs, outputs, name='decoder')
        self.decoder.summary()
        plot_model(self.decoder, to_file='decoder.png', show_shapes=True)

        # autoencoder = encoder + decoder
        # instantiate autoencoder model
        self.autoencoder = Model(inputs, self.decoder(self.encoder(inputs)), name='autoencoder')
        self.autoencoder.summary()
        plot_model(self.autoencoder, to_file='autoencoder.png', show_shapes=True)

        # Mean Square Error (MSE) loss function, Adam optimizer
        self.autoencoder.compile(loss='mse', optimizer='adam')


    # training the autoencoder using randomly sampled
    # states from the environment
    def train_autoencoder(self, x_train, x_test):
        # train the autoencoder
        batch_size = 32
        self.autoencoder.fit(x_train,
                             x_train,
                             validation_data=(x_test, x_test),
                             epochs=10,
                             batch_size=batch_size)


    # 4 models are built but 3 models share the same parameters.
    # hence training one, trains the rest.
    # the 3 models that share the same parameters are action, logp,
    # and entropy models. entropy model is used by A2C only.
    # each model has the same MLP structure:
    # Input(2)-Encoder-Output(1).
    # the output activation depends on the nature of the output.
    def build_actor_critic(self):
        inputs = Input(shape=(self.state_dim, ), name='state')
        self.encoder.trainable = False
        x = self.encoder(inputs)
        mean = Dense(1,
                     activation='linear',
                     kernel_initializer='zero',
                     name='mean')(x)
        stddev = Dense(1,
                       kernel_initializer='zero',
                       name='stddev')(x)
        # use of softplusk avoids stddev = 0
        stddev = Activation('softplusk', name='softplus')(stddev)
        action = Lambda(self.action,
                        output_shape=(1,),
                        name='action')([mean, stddev])
        self.actor_model = Model(inputs, action, name='action')
        self.actor_model.summary()
        plot_model(self.actor_model, to_file='actor_model.png', show_shapes=True)

        logp = Lambda(self.logp,
                      output_shape=(1,),
                      name='logp')([mean, stddev, action])
        self.logp_model = Model(inputs, logp, name='logp')
        self.logp_model.summary()
        plot_model(self.logp_model, to_file='logp_model.png', show_shapes=True)

        entropy = Lambda(self.entropy,
                         output_shape=(1,),
                         name='entropy')([mean, stddev])
        self.entropy_model = Model(inputs, entropy, name='entropy')
        self.entropy_model.summary()
        plot_model(self.entropy_model, to_file='entropy_model.png', show_shapes=True)

        value = Dense(1,
                      activation='linear',
                      kernel_initializer='zero',
                      name='value')(x)
        self.value_model = Model(inputs, value, name='value')
        self.value_model.summary()
        plot_model(self.value_model, to_file='value_model.png', show_shapes=True)

        # beta of entropy used in A2C
        beta = 0.9 if self.args.a2c else 0.0

        # logp loss of policy network
        loss = self.logp_loss(self.get_entropy(self.state), beta=beta)
        optimizer = RMSprop(lr=1e-3)
        self.logp_model.compile(loss=loss, optimizer=optimizer)

        # loss function of A2C is mse, while the rest use their own
        # loss function called value loss
        loss = 'mse' if self.args.a2c else self.value_loss
        optimizer = Adam(lr=1e-3)
        self.value_model.compile(loss=loss, optimizer=optimizer)


    # logp loss, the 3rd and 4th variables (entropy and beta) are needed
    # by A2C so we have a different loss function structure
    def logp_loss(self, entropy, beta=0.0):
        def loss(y_true, y_pred):
            return -K.mean((y_pred * y_true) + (beta * entropy), axis=-1)

        return loss


    # typical loss function structure that accepts 2 arguments only
    # this will be used by value loss of all methods except A2C
    def value_loss(self, y_true, y_pred):
        return -K.mean(y_pred * y_true, axis=-1)


    # save the actor, critic and encoder weights
    # useful for restoring the trained models
    def save_weights(self, actor_weights, encoder_weights, value_weights=None):
        self.actor_model.save_weights(actor_weights)
        self.encoder.save_weights(encoder_weights)
        if value_weights is not None:
            self.value_model.save_weights(value_weights)


    # load the trained weights
    # useful if we are interested in using the network right away
    def load_weights(self, actor_weights, value_weights=None):
        self.actor_model.load_weights(actor_weights)
        if value_weights is not None:
            self.value_model.load_weights(value_weights)

    
    # load encoder trained weights
    # useful if we are interested in using the network right away
    def load_encoder_weights(self, encoder_weights):
        self.encoder.load_weights(encoder_weights)

    
    # call the policy network to sample an action
    def act(self, state):
        action = self.actor_model.predict(state)
        return action[0]


    # call the value network to predict the value of state
    def value(self, state):
        value = self.value_model.predict(state)
        return value[0]


    # return the entropy of the policy distribution
    def get_entropy(self, state):
        entropy = self.entropy_model.predict(state)
        return entropy[0]


    # train by episode (REINFORCE, REINFORCE with baseline
    # and A2C use this routine to prepare the dataset before
    # the step by step training)
    def train_by_episode(self, last_value=0):
        if self.args.actor_critic:
            print("Actor-Critic must be trained per step")
            return
        elif self.args.a2c:
            # implements A2C training from the last state
            # to the first state
            # discount factor
            gamma = 0.95
            r = last_value
            # the memory is visited in reverse as shown
            # in Algorithm 10.5.1
            for item in self.memory[::-1]:
                [step, state, next_state, reward, done] = item
                # compute the return
                r = reward + gamma*r
                item = [step, state, next_state, r, done]
                # train per step
                # a2c reward has been discounted
                self.train(item)

            return

        # only REINFORCE and REINFORCE with baseline
        # use the ff codes
        # convert the rewards to returns
        rewards = []
        gamma = 0.99
        for item in self.memory:
            [_, _, _, reward, _] = item
            rewards.append(reward)
        # rewards = np.array(self.memory)[:,3].tolist()

        # compute return per step
        # return is the sum of rewards from t til end of episode
        # return replaces reward in the list
        for i in range(len(rewards)):
            reward = rewards[i:]
            horizon = len(reward)
            discount =  [math.pow(gamma, t) for t in range(horizon)]
            return_ = np.dot(reward, discount)
            self.memory[i][3] = return_

        # train every step
        for item in self.memory:
            self.train(item, gamma=gamma)


    # main routine for training as used by all 4 policy gradient
    # methods
    def train(self, item, gamma=1.0):
        [step, state, next_state, reward, done] = item

        # must save state for entropy computation
        self.state = state

        discount_factor = gamma**step

        # reinforce-baseline: delta = return - value
        # actor-critic: delta = reward - value + discounted_next_value
        # a2c: delta = discounted_reward - value
        delta = reward - self.value(state)[0] 

        # only REINFORCE does not use a critic (value network)
        critic = False
        if self.args.baseline:
            critic = True
        elif self.args.actor_critic:
            # since this function is called by Actor-Critic
            # directly, evaluate the value function here
            critic = True
            if not done:
                next_value = self.value(next_state)[0]
                # add  the discounted next value
                delta += gamma*next_value
        elif self.args.a2c:
            critic = True
        else:
            delta = reward

        # apply the discount factor as shown in Algortihms
        # 10.2.1, 10.3.1 and 10.4.1
        discounted_delta = delta * discount_factor
        discounted_delta = np.reshape(discounted_delta, [-1, 1])
        verbose = 1 if done else 0

        # train the logp model (implies training of actor model
        # as well) since they share exactly the same set of
        # parameters
        self.logp_model.fit(np.array(state),
                            discounted_delta,
                            batch_size=1,
                            epochs=1,
                            verbose=verbose)

        # in A2C, the target value is the return (reward
        # replaced by return in the train_by_episode function)
        if self.args.a2c:
            discounted_delta = reward
            discounted_delta = np.reshape(discounted_delta, [-1, 1])

        # train the value network (critic)
        if critic:
            self.value_model.fit(np.array(state),
                                 discounted_delta,
                                 batch_size=1,
                                 epochs=1,
                                 verbose=verbose)


def setup_parser():
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
                        help="Advantage-Actor-Critic (A2C)")
    parser.add_argument("-r",
                        "--random",
                        action='store_true',
                        help="Random action policy")
    parser.add_argument("-w",
                        "--actor_weights",
                        help="Load pre-trained actor model weights")
    parser.add_argument("-y",
                        "--value_weights",
                        help="Load pre-trained value model weights")
    parser.add_argument("-e",
                        "--encoder_weights",
                        help="Load pre-trained encoder model weights")
    parser.add_argument("-t",
                        "--train",
                        help="Enable training",
                        action='store_true')
    args = parser.parse_args()
    return args


def setup_files(args):
    # housekeeping to keep the output logs in separate folders
    postfix = 'reinforce'
    has_value_model = False
    if args.baseline:
        postfix = "reinforce-baseline"
        has_value_model = True
    elif args.actor_critic:
        postfix = "actor-critic"
        has_value_model = True
    elif args.a2c:
        postfix = "a2c"
        has_value_model = True
    elif args.random:
        postfix = "random"

    # create the folder for log files
    try:
        os.mkdir(postfix)
    except FileExistsError:
        print(postfix, " folder exists")

    fileid = "%s-%d" % (postfix, int(time.time()))
    actor_weights = "actor_weights-%s.h5" % fileid
    actor_weights = os.path.join(postfix, actor_weights)
    encoder_weights = "encoder_weights-%s.h5" % fileid
    encoder_weights = os.path.join(postfix, encoder_weights)
    value_weights = None
    if has_value_model:
        value_weights = "value_weights-%s.h5" % fileid
        value_weights = os.path.join(postfix, value_weights)

    outdir = "/tmp/%s" % postfix

    misc = (postfix, fileid, outdir, has_value_model)
    weights = (actor_weights, encoder_weights, value_weights)

    return weights, misc



def setup_agent(env, args):
    # instantiate agent
    agent = PolicyAgent(env, args)
    # if weights are given, lets load them
    if args.encoder_weights:
        agent.load_encoder_weights(args.encoder_weights)
    else:
        x_train = [env.observation_space.sample() for x in range(200000)]
        x_train = np.array(x_train)
        x_test = [env.observation_space.sample() for x in range(20000)]
        x_test = np.array(x_test)
        agent.train_autoencoder(x_train, x_test)

    agent.build_actor_critic()
    train = True
    # if weights are given, lets load them
    if args.actor_weights:
        train = False
        if args.value_weights:
            agent.load_weights(args.actor_weights,
                               args.value_weights)
        else:
            agent.load_weights(args.actor_weights)

    return agent, train


def setup_writer(fileid, postfix):
    # we dump episode num, step, total reward, and 
    # number of episodes solved in a csv file for analysis
    csvfilename = "%s.csv" % fileid
    csvfilename = os.path.join(postfix, csvfilename)
    csvfile = open(csvfilename, 'w', 1)
    writer = csv.writer(csvfile,
                        delimiter=',',
                        quoting=csv.QUOTE_NONNUMERIC)
    writer.writerow(['Episode',
                     'Step',
                     'Total Reward',
                     'Number of Episodes Solved'])

    return csvfile, writer


if __name__ == '__main__':
    args = setup_parser()
    logger.setLevel(logger.ERROR)

    weights, misc = setup_files(args)
    actor_weights, encoder_weights, value_weights = weights
    postfix, fileid, outdir, has_value_model = misc

    env = gym.make(args.env_id)
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    
    # register softplusk activation. just in case the reader wants
    # to use this activation
    get_custom_objects().update({'softplusk':Activation(softplusk)})
   
    agent, train = setup_agent(env, args)

    if args.train or train:
        train = True
        csvfile, writer = setup_writer(fileid, postfix)

    # number of episodes we run the training
    episode_count = 1000
    state_dim = env.observation_space.shape[0]
    n_solved = 0 
    start_time = datetime.datetime.now()
    # sampling and fitting
    for episode in range(episode_count):
        state = env.reset()
        # state is car [position, speed]
        state = np.reshape(state, [1, state_dim])
        # reset all variables and memory before the start of
        # every episode
        step = 0
        total_reward = 0
        done = False
        agent.reset_memory()
        while not done:
            # [min, max] action = [-1.0, 1.0]
            # for baseline, random choice of action will not move
            # the car pass the flag pole
            if args.random:
                action = env.action_space.sample()
            else:
                action = agent.act(state)
            env.render()
            # after executing the action, get s', r, done
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_dim])
            # save the experience unit in memory for training
            # Actor-Critic does not need this but we keep it anyway.
            item = [step, state, next_state, reward, done]
            agent.remember(item)

            if args.actor_critic and train:
                # only actor-critic performs online training
                # train at every step as it happens
                agent.train(item, gamma=0.99)
            elif not args.random and done and train:
                # for REINFORCE, REINFORCE with baseline, and A2C
                # we wait for the completion of the episode before 
                # training the network(s)
                # last value as used by A2C
                v = 0 if reward > 0 else agent.value(next_state)[0]
                agent.train_by_episode(last_value=v)

            # accumulate reward
            total_reward += reward
            # next state is the new state
            state = next_state
            step += 1

        if reward > 0:
            n_solved += 1
        elapsed = datetime.datetime.now() - start_time
        fmt = "Episode=%d, Step=%d, Action=%f, Reward=%f"
        fmt = fmt + ", Total_Reward=%f, Elapsed=%s"
        msg = (episode, step, action[0], reward, total_reward, elapsed)
        print(fmt % msg)
        # log the data on the opened csv file for analysis
        if train:
            writer.writerow([episode, step, total_reward, n_solved])



    # after training, save the actor and value models weights
    if not args.random and train:
        if has_value_model:
            agent.save_weights(actor_weights,
                               encoder_weights,
                               value_weights)
        else:
            agent.save_weights(actor_weights,
                               encoder_weights)

    # close the env and write monitor result info to disk
    if train:
        csvfile.close()
    env.close() 
