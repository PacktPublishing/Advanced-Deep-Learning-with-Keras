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

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Lambda, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.utils import plot_model

import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np
import argparse
import gym
from gym import wrappers, logger
import csv
import time
import os
import datetime
import math


def softplusk(x):
    """Some implementations use a modified softplus 
        to ensure that the stddev is never zero
    Argument:
        x (tensor): activation input
    """
    return K.softplus(x) + 1e-10


class PolicyAgent:
    def __init__(self, env):
        """Implements the models and training of 
            Policy Gradient Methods
        Argument:
            env (Object): OpenAI gym environment
        """

        self.env = env
        # entropy loss weight
        self.beta = 0.0
        # value loss for all policy gradients except A2C
        self.loss = self.value_loss
        
        # s,a,r,s' are stored in memory
        self.memory = []

        # for computation of input size
        self.state = env.reset()
        self.state_dim = env.observation_space.shape[0]
        self.state = np.reshape(self.state, [1, self.state_dim])
        self.build_autoencoder()


    def reset_memory(self):
        """Clear the memory before the start 
            of every episode
        """
        self.memory = []


    def remember(self, item):
        """Remember every s,a,r,s' in every step of the episode
        """
        self.memory.append(item)


    def action(self, args):
        """Given mean and stddev, sample an action, clip 
            and return
            We assume Gaussian distribution of probability 
            of selecting an action given a state
        Argument:
            args (list) : mean, stddev list
        Return:
            action (tensor): policy action
        """
        mean, stddev = args
        dist = tfp.distributions.Normal(loc=mean, scale=stddev)
        action = dist.sample(1)
        action = K.clip(action,
                        self.env.action_space.low[0],
                        self.env.action_space.high[0])
        return action


    def logp(self, args):
        """Given mean, stddev, and action compute
            the log probability of the Gaussian distribution
        Argument:
            args (list) : mean, stddev action, list
        Return:
            logp (tensor): log of action
        """
        mean, stddev, action = args
        dist = tfp.distributions.Normal(loc=mean, scale=stddev)
        logp = dist.log_prob(action)
        return logp


    def entropy(self, args):
        """Given the mean and stddev compute 
            the Gaussian dist entropy
        Argument:
            args (list) : mean, stddev list
        Return:
            entropy (tensor): action entropy
        """
        mean, stddev = args
        dist = tfp.distributions.Normal(loc=mean, scale=stddev)
        entropy = dist.entropy()
        return entropy


    def build_autoencoder(self):
        """Autoencoder to convert states into features
        """
        # first build the encoder model
        inputs = Input(shape=(self.state_dim, ), name='state')
        feature_size = 32
        x = Dense(256, activation='relu')(inputs)
        x = Dense(128, activation='relu')(x)
        feature = Dense(feature_size, name='feature_vector')(x)

        # instantiate encoder model
        self.encoder = Model(inputs, feature, name='encoder')
        self.encoder.summary()
        plot_model(self.encoder,
                   to_file='encoder.png', 
                   show_shapes=True)

        # build the decoder model
        feature_inputs = Input(shape=(feature_size,), 
                               name='decoder_input')
        x = Dense(128, activation='relu')(feature_inputs)
        x = Dense(256, activation='relu')(x)
        outputs = Dense(self.state_dim, activation='linear')(x)

        # instantiate decoder model
        self.decoder = Model(feature_inputs, 
                             outputs, 
                             name='decoder')
        self.decoder.summary()
        plot_model(self.decoder, 
                   to_file='decoder.png', 
                   show_shapes=True)

        # autoencoder = encoder + decoder
        # instantiate autoencoder model
        self.autoencoder = Model(inputs, 
                                 self.decoder(self.encoder(inputs)),
                                 name='autoencoder')
        self.autoencoder.summary()
        plot_model(self.autoencoder, 
                   to_file='autoencoder.png', 
                   show_shapes=True)

        # Mean Square Error (MSE) loss function, Adam optimizer
        self.autoencoder.compile(loss='mse', optimizer='adam')


    def train_autoencoder(self, x_train, x_test):
        """Training the autoencoder using randomly sampled
            states from the environment
        Arguments:
            x_train (tensor): autoencoder train dataset
            x_test (tensor): autoencoder test dataset
        """
        # train the autoencoder
        batch_size = 32
        self.autoencoder.fit(x_train,
                             x_train,
                             validation_data=(x_test, x_test),
                             epochs=10,
                             batch_size=batch_size)


    def build_actor_critic(self):
        """4 models are built but 3 models share the
            same parameters. hence training one, trains the rest.
            The 3 models that share the same parameters 
                are action, logp, and entropy models. 
            Entropy model is used by A2C only.
            Each model has the same MLP structure:
            Input(2)-Encoder-Output(1).
            The output activation depends on the nature 
                of the output.
        """
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
        plot_model(self.actor_model, 
                   to_file='actor_model.png', 
                   show_shapes=True)

        logp = Lambda(self.logp,
                      output_shape=(1,),
                      name='logp')([mean, stddev, action])
        self.logp_model = Model(inputs, logp, name='logp')
        self.logp_model.summary()
        plot_model(self.logp_model, 
                   to_file='logp_model.png', 
                   show_shapes=True)

        entropy = Lambda(self.entropy,
                         output_shape=(1,),
                         name='entropy')([mean, stddev])
        self.entropy_model = Model(inputs, entropy, name='entropy')
        self.entropy_model.summary()
        plot_model(self.entropy_model, 
                   to_file='entropy_model.png', 
                   show_shapes=True)

        value = Dense(1,
                      activation='linear',
                      kernel_initializer='zero',
                      name='value')(x)
        self.value_model = Model(inputs, value, name='value')
        self.value_model.summary()
        plot_model(self.value_model, 
                   to_file='value_model.png', 
                   show_shapes=True)


        # logp loss of policy network
        loss = self.logp_loss(self.get_entropy(self.state), 
                              beta=self.beta)
        optimizer = RMSprop(lr=1e-3)
        self.logp_model.compile(loss=loss, optimizer=optimizer)

        optimizer = Adam(lr=1e-3)
        self.value_model.compile(loss=self.loss, optimizer=optimizer)


    def logp_loss(self, entropy, beta=0.0):
        """logp loss, the 3rd and 4th variables 
            (entropy and beta) are needed by A2C 
            so we have a different loss function structure
        Arguments:
            entropy (tensor): Entropy loss
            beta (float): Entropy loss weight
        Return:
            loss (tensor): computed loss
        """
        def loss(y_true, y_pred):
            return -K.mean((y_pred * y_true) \
                    + (beta * entropy), axis=-1)

        return loss


    def value_loss(self, y_true, y_pred):
        """Typical loss function structure that accepts 
            2 arguments only
           This will be used by value loss of all methods 
            except A2C
        Arguments:
            y_true (tensor): value ground truth
            y_pred (tensor): value prediction
        Return:
            loss (tensor): computed loss
        """
        loss = -K.mean(y_pred * y_true, axis=-1)
        return loss


    def save_weights(self, 
                     actor_weights, 
                     encoder_weights, 
                     value_weights=None):
        """Save the actor, critic and encoder weights
            useful for restoring the trained models
        Arguments:
            actor_weights (tensor): actor net parameters
            encoder_weights (tensor): encoder weights
            value_weights (tensor): value net parameters
        """
        self.actor_model.save_weights(actor_weights)
        self.encoder.save_weights(encoder_weights)
        if value_weights is not None:
            self.value_model.save_weights(value_weights)


    def load_weights(self,
                     actor_weights, 
                     value_weights=None):
        """Load the trained weights
           useful if we are interested in using 
                the network right away
        Arguments:
            actor_weights (string): filename containing actor net
                weights
            value_weights (string): filename containing value net
                weights
        """
        self.actor_model.load_weights(actor_weights)
        if value_weights is not None:
            self.value_model.load_weights(value_weights)

    
    def load_encoder_weights(self, encoder_weights):
        """Load encoder trained weights
           useful if we are interested in using 
            the network right away
        Arguments:
            encoder_weights (string): filename containing encoder net
                weights
        """
        self.encoder.load_weights(encoder_weights)

    
    def act(self, state):
        """Call the policy network to sample an action
        Argument:
            state (tensor): environment state
        Return:
            act (tensor): policy action
        """
        action = self.actor_model.predict(state)
        return action[0]


    def value(self, state):
        """Call the value network to predict the value of state
        Argument:
            state (tensor): environment state
        Return:
            value (tensor): state value
        """
        value = self.value_model.predict(state)
        return value[0]


    def get_entropy(self, state):
        """Return the entropy of the policy distribution
        Argument:
            state (tensor): environment state
        Return:
            entropy (tensor): entropy of policy
        """
        entropy = self.entropy_model.predict(state)
        return entropy[0]


class REINFORCEAgent(PolicyAgent):
    def __init__(self, env):
        """Implements the models and training of 
           REINFORCE policy gradient method
        Arguments:
            env (Object): OpenAI gym environment
        """
        super().__init__(env) 

    def train_by_episode(self):
        """Train by episode
           Prepare the dataset before the step by step training
        """
        # only REINFORCE and REINFORCE with baseline
        # use the ff code
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


    def train(self, item, gamma=1.0):
        """Main routine for training 
        Arguments:
            item (list) : one experience unit
            gamma (float) : discount factor [0,1]
        """
        [step, state, next_state, reward, done] = item

        # must save state for entropy computation
        self.state = state

        discount_factor = gamma**step
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


class REINFORCEBaselineAgent(REINFORCEAgent):
    def __init__(self, env):
        """Implements the models and training of 
           REINFORCE w/ baseline policy 
           gradient method
        Arguments:
            env (Object): OpenAI gym environment
        """
        super().__init__(env) 


    def train(self, item, gamma=1.0):
        """Main routine for training 
        Arguments:
            item (list) : one experience unit
            gamma (float) : discount factor [0,1]
        """
        [step, state, next_state, reward, done] = item

        # must save state for entropy computation
        self.state = state

        discount_factor = gamma**step

        # reinforce-baseline: delta = return - value
        delta = reward - self.value(state)[0] 

        # apply the discount factor as shown in Algorithms
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

        # train the value network (critic)
        self.value_model.fit(np.array(state),
                             discounted_delta,
                             batch_size=1,
                             epochs=1,
                             verbose=verbose)

class A2CAgent(PolicyAgent):
    def __init__(self, env):
        """Implements the models and training of 
           A2C policy gradient method
        Arguments:
            env (Object): OpenAI gym environment
        """
        super().__init__(env) 
        # beta of entropy used in A2C
        self.beta = 0.9
        # loss function of A2C value_model is mse
        self.loss = 'mse'


    def train_by_episode(self, last_value=0):
        """Train by episode 
           Prepare the dataset before the step by step training
        Arguments:
            last_value (float): previous prediction of value net
        """
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


    def train(self, item, gamma=1.0):
        """Main routine for training 
        Arguments:
            item (list) : one experience unit
            gamma (float) : discount factor [0,1]
        """
        [step, state, next_state, reward, done] = item

        # must save state for entropy computation
        self.state = state

        discount_factor = gamma**step

        # a2c: delta = discounted_reward - value
        delta = reward - self.value(state)[0] 

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
        discounted_delta = reward
        discounted_delta = np.reshape(discounted_delta, [-1, 1])

        # train the value network (critic)
        self.value_model.fit(np.array(state),
                             discounted_delta,
                             batch_size=1,
                             epochs=1,
                             verbose=verbose)


class ActorCriticAgent(PolicyAgent):
    def __init__(self, env):
        """Implements the models and training of 
           Actor Critic policy gradient method
        Arguments:
            env (Object): OpenAI gym environment
        """
        super().__init__(env) 


    def train(self, item, gamma=1.0):
        """Main routine for training
        Arguments:
            item (list) : one experience unit
            gamma (float) : discount factor [0,1]
        """
        [step, state, next_state, reward, done] = item

        # must save state for entropy computation
        self.state = state

        discount_factor = gamma**step

        # actor-critic: delta = reward - value 
        #       + discounted_next_value
        delta = reward - self.value(state)[0] 

        # since this function is called by Actor-Critic
        # directly, evaluate the value function here
        if not done:
            next_value = self.value(next_state)[0]
            # add  the discounted next value
            delta += gamma*next_value

        # apply the discount factor as shown in Algorithms
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
                        "--actor-critic",
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
                        "--actor-weights",
                        help="Load pre-trained actor model weights")
    parser.add_argument("-y",
                        "--value-weights",
                        help="Load pre-trained value model weights")
    parser.add_argument("-e",
                        "--encoder-weights",
                        help="Load pre-trained encoder model weights")
    parser.add_argument("-t",
                        "--train",
                        help="Enable training",
                        action='store_true')
    args = parser.parse_args()
    return args


def setup_files(args):
    """Housekeeping to keep the output logs in separate folders
    Arguments:
        args: user-defined arguments
    """
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
    """Agent initialization
    Arguments:
        env (Object): OpenAI environment
        args : user-defined arguments
    """
    # instantiate agent
    if args.baseline:
        agent = REINFORCEBaselineAgent(env)
    elif args.a2c:
        agent = A2CAgent(env)
    elif args.actor_critic:
        agent = ActorCriticAgent(env)
    else:
        agent = REINFORCEAgent(env)

    # if weights are given, lets load them
    if args.encoder_weights:
        agent.load_encoder_weights(args.encoder_weights)
    else:
        x_train = [env.observation_space.sample() \
                   for x in range(200000)]
        x_train = np.array(x_train)
        x_test = [env.observation_space.sample() \
                  for x in range(20000)]
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
    """Use to prepare file and writer for data logging
    Arguments:
        fileid (string): unique file identfier
        postfix (string): path
    """
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
        start_time = datetime.datetime.now()
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
                if args.a2c:
                    v = 0 if reward > 0 else agent.value(next_state)[0]
                    agent.train_by_episode(last_value=v)
                else:
                    agent.train_by_episode()

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
