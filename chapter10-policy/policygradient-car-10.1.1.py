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
from keras.optimizers import Adam
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

import tensorflow as tf

import numpy as np
import argparse
import gym
from gym import wrappers, logger
import sys
import csv
import time
import os


# some implementations use a modified softplus to ensure that
# the stddev is never zero
# added here just for curious readers
def softplusk(x):
    return K.softplus(x) + 1e-3


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
        state_size = env.observation_space.shape[0]
        self.state = np.reshape(self.state, [1, state_size])

        # 4 models
        # actor, logp and entropy share the same parameters
        # value is a separate network
        actor, logp, entropy, value = self.build_models(state_size)
        self.actor_model = actor
        self.logp_model = logp
        self.entropy_model = entropy
        self.value_model = value

        # beta of entropy used in A2C
        beta = 0.9 if self.args.a2c else 0.0

        # logp loss of policy network
        loss = self.logp_loss(self.get_entropy(self.state), beta=beta)

        # learning rate
        lr = 1e-3

        # adjust decay for future optimizations
        decay = 0.0 

        # apply logp loss
        self.logp_model.compile(loss=loss,
                                optimizer=Adam(lr=lr, decay=decay))

        # smaller value learning rate allows the policy to explore
        # bigger value results to too early optimization of policy
        # network missing the flag on the mountain top
        lr = 1e-5
        if args.actor_critic:
            lr = 1e-7

        # adjust decay for future optimizations
        decay = 0.0

        # loss function of A2C is mse, while the rest use their own
        # loss function called value loss
        loss = 'mse' if self.args.a2c else self.value_loss
        self.value_model.compile(loss=loss,
                                 optimizer=Adam(lr=lr, decay=decay))


    # clear the memory before the start of every episode
    def reset_memory(self):
        self.memory = []


    # remember every s,a,r,s' in every step of the episode
    def remember(self, item):
        self.memory.append(item)


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


    # given the mean, stddev, and action compute
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


    # 4 models are built but 3 models share the same parameters.
    # hence training one, trains the rest.
    # the 3 models that share the same parameters are action, logp,
    # and entropy models. entropy model is used by A2C only.
    # each model has the same MLP structure:
    # Input(2)-Dense(256)-Dense(256)-Output(1).
    # the output activation depends on the nature of the output.
    def build_models(self, n_inputs):
        inputs = Input(shape=(n_inputs, ), name='state')
        # all parameters are initially set to zero
        kernel_initializer = 'zeros'
        x = Dense(256,
                  activation='relu',
                  kernel_initializer=kernel_initializer)(inputs)
        x = Dense(256,
                  activation='tanh',
                  kernel_initializer=kernel_initializer)(x)

        mean = Dense(1,
                     activation='linear',
                     kernel_initializer=kernel_initializer,
                     name='mean')(x)
        stddev = Dense(1,
                       activation='softplus',
                       kernel_initializer=kernel_initializer,
                       name='stddev')(x)
        # in case the reader is curious on how to use the softplusk
        # comment out the softplus in stddev and uncomment this line
        # stddev = Activation('softplusk')(stddev)
        action = Lambda(self.action,
                        output_shape=(1,),
                        name='action')([mean, stddev])
        actor_model = Model(inputs, action, name='action')
        actor_model.summary()

        logp = Lambda(self.logp,
                      output_shape=(1,),
                      name='logp')([mean, stddev, action])
        logp_model = Model(inputs, logp, name='logp')
        logp_model.summary()

        entropy = Lambda(self.entropy,
                         output_shape=(1,),
                         name='entropy')([mean, stddev])
        entropy_model = Model(inputs, entropy, name='entropy')
        entropy_model.summary()

        x = Dense(256,
                  activation='relu',
                  kernel_initializer=kernel_initializer)(inputs)
        x = Dense(256,
                  activation='tanh',
                  kernel_initializer=kernel_initializer)(x)
        value = Dense(1,
                      activation='linear',
                      name='value',
                      kernel_initializer=kernel_initializer)(x)
        value_model = Model(inputs, value, name='value')
        value_model.summary()

        return actor_model, logp_model, entropy_model, value_model


    # save the actor and critic (if applicable) weights
    # useful for restoring the trained models
    def save_weights(self, actor_weights, value_weights=None):
        self.actor_model.save_weights(actor_weights)
        if value_weights is not None:
            self.value_model.save_weights(value_weights)


    # load the trained weights
    # useful if we are interested in using the network right away
    def load_weights(self, actor_weights, value_weights=None):
        self.actor_model.load_weights(actor_weights)
        if value_weights is not None:
            self.value_model.load_weights(value_weights)

    
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
            gamma = 0.99
            i = 1
            r = last_value
            # the memory is visited in reverse as shown
            # in Algorithm 10.5.1
            for item in self.memory[::-1]:
                [step, state, next_state, reward, done] = item
                # compute the return
                r = reward + gamma*r
                item = [step, state, next_state, r, done]
                # train per step
                self.train(item)
                i += 1

            return

        # only REINFORCE and REINFORCE with baseline
        # use the ff codes
        # convert the rewards to returns
        rewards = []
        for item in self.memory:
            [_, _, _, reward, _] = item
            rewards.append(reward)

        # compute return per step
        # return is the sum of rewards from t til end of episode
        # return replaces reward in the list
        for i in range(len(rewards)):
            self.memory[i][3] = np.sum(rewards[i:])

        # train every step
        for item in self.memory:
            self.train(item)


    # main routine for training as used by all 4 policy gradient
    # methods
    def train(self, item):
        [step, state, next_state, reward, done] = item

        # must save state for entropy computation
        self.state = state

        # discount factor
        gamma = 0.99

        # a2c reward has been discounted in the train_by_episode
        if self.args.a2c:
            gamma = 1.0

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
            if done:
                next_value = 0.0
            else:
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

        # in A2C, the target value is the return (which is
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
    args = parser.parse_args()

    logger.setLevel(logger.ERROR)
    env = gym.make(args.env_id)

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
    value_weights = None
    if has_value_model:
        value_weights = "value_weights-%s.h5" % fileid
        value_weights = os.path.join(postfix, value_weights)

    outdir = "/tmp/%s" % postfix

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

    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    
    # register softplusk activation. just in case the reader wants
    # to use this activation
    get_custom_objects().update({'softplusk':Activation(softplusk)})
    
    # instantiate agent
    agent = PolicyAgent(env, args)
    train = True
    # if weights are given, lets load them
    if args.actor_weights:
        train = False
        if args.value_weights:
            agent.load_weights(args.actor_weights,
                               args.value_weights)
        else:
            agent.load_weights(args.actor_weights)

    # number of episodes we run the training
    episode_count = 200
    state_size = env.observation_space.shape[0]
    n_solved = 0 

    # sampling and fitting
    for episode in range(episode_count):
        state = env.reset()
        # state is car [position, speed]
        state = np.reshape(state, [1, state_size])
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
            next_state = np.reshape(next_state, [1, state_size])
            # save the experience unit in memory for training
            # Actor-Critic does not need this but we keep it anyway.
            item = [step, state, next_state, reward, done]
            agent.remember(item)

            if args.actor_critic and train:
                # only actor-critic performs online training
                # train at every step as it happens
                agent.train(item)
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
        fmt = "Episode=%d, Step=%d. Action=%f, Reward=%f, Total_Reward=%f"
        print(fmt % (episode, step, action[0], reward, total_reward))
        # log the data on the opened csv file for analysis
        writer.writerow([episode, step, total_reward, n_solved])

    # after training, save the actor and value models weights
    if not args.random and train:
        if has_value_model:
            agent.save_weights(actor_weights, value_weights)
        else:
            agent.save_weights(actor_weights)

    # close the env and write monitor result info to disk
    csvfile.close()
    env.close() 
