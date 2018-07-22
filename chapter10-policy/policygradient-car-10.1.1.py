"""Trains a policy network to solve
MountainCarCountinuous-v0 problem


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


def softplusk(x):
    return K.softplus(x) + 1e-3


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
        beta = 0.5 if self.args.a2c else 0.0
        loss = self.logp_loss(self.get_entropy(self.state), beta=beta)
        lr = 1e-4
        decay = 0.0 # lr*1e-6

        self.logp_model.compile(loss=loss,
                                   optimizer=Adam(lr=lr, decay=decay))
        lr = 1e-5
        if args.actor_critic:
            lr = 1e-7
        decay = 0.0 # lr*1e-6

        loss = 'mse' if self.args.a2c else self.value_loss
        self.value_model.compile(loss=loss,
                                 optimizer=Adam(lr=lr, decay=decay))


    def reset_memory(self):
        self.memory = []


    def remember(self, item):
        self.memory.append(item)


    def logp_loss(self, entropy, beta=0.0):
        def loss(y_true, y_pred):
            return -K.mean((y_pred * y_true) + (beta * entropy), axis=-1)

        return loss


    def value_loss(self, y_true, y_pred):
        return -K.mean(y_pred * y_true, axis=-1)


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


    def save_weights(self, actor_weights, value_weights=None):
        self.actor_model.save_weights(actor_weights)
        if value_weights is not None:
            self.value_model.save_weights(value_weights)


    def load_weights(self, actor_weights, value_weights=None):
        self.actor_model.load_weights(actor_weights)
        if value_weights is not None:
            self.value_model.load_weights(value_weights)


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
        if self.args.actor_critic:
            print("Actor-Critic must be trained per step")
            return
        elif self.args.a2c:
            gamma = 0.99
            i = 1
            max_step = len(self.memory)
            r = last_value
            for item in self.memory[::-1]:
                [step, state, next_state, reward, done] = item
                step = max_step - i
                r = reward + gamma*r
                item = [step, state, next_state, r, done]
                self.train(item)
                i += 1

            return


        rewards = []
        for item in self.memory:
            [_, _, _, reward, _] = item
            rewards.append(reward)

        # compute return per step
        # return is the sum of rewards from t til end of episode
        # return replaces reward in the list
        for i in range(len(rewards)):
            self.memory[i][3] = np.sum(rewards[i:])

        for item in self.memory:
            self.train(item)


    def train(self, item):
        [step, state, next_state, reward, done] = item

        # must save state for entropy computation
        self.state = state
        gamma = 0.99

        # a2c reward has been discounted in the train_per_episode
        if self.args.a2c:
            gamma = 1.0

        discount_factor = gamma**step

        # reinforce-baseline, delta = return - value
        # actor-critic, delta = reward - value + discounted_next_value
        # a2c, delta = discounted_reward - value
        delta = reward - self.value(state)[0] 

        critic = False
        if self.args.baseline:
            critic = True
        elif self.args.actor_critic:
            critic = True
            if done:
                next_value = 0.0
            else:
                next_value = self.value(next_state)[0]
            delta += gamma*next_value
        elif self.args.a2c:
            critic = True
        else:
            delta = reward

        discounted_delta = delta * discount_factor
        discounted_delta = np.reshape(discounted_delta, [-1, 1])
        verbose = 1 if done else 0

        self.logp_model.fit(np.array(state),
                            discounted_delta,
                            batch_size=1,
                            epochs=1,
                            verbose=verbose)

        if self.args.a2c:
            discounted_delta = reward
            discounted_delta = np.reshape(discounted_delta, [-1, 1])

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
                        #default='Pendulum-v0',
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
    parser.add_argument("-w",
                        "--actor_weights",
                        help="Load pre-trained actor model weights")
    parser.add_argument("-y",
                        "--value_weights",
                        help="Load pre-trained value model weights")
    args = parser.parse_args()

    logger.setLevel(logger.ERROR)
    env = gym.make(args.env_id)

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

    get_custom_objects().update({'softplusk':Activation(softplusk)})
    
    # instantiate agent
    agent = PolicyAgent(env, args)
    if args.actor_weights:
        if args.value_weights:
            agent.load_weights(args.actor_weights,
                               args.value_weights)
        else:
            agent.load_weights(args.actor_weights)

    # should be solved in this number of episodes
    episode_count = 200
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
            item = [step, state, next_state, reward, done]
            agent.remember(item)

            if args.actor_critic:
                agent.train(item)
            elif not args.random and done:
                v = 0 if reward > 0 else agent.value(next_state)[0]
                agent.train_by_episode(last_value=v)

            total_reward += reward
            state = next_state
            step += 1

        if reward > 0:
            n_solved += 1
        fmt = "Episode=%d, Step=%d. Action=%f, Reward=%f, Total_Reward=%f"
        print(fmt % (episode, step, action[0], reward, total_reward))
        writer.writerow([episode, step, total_reward, n_solved])

    if not args.random:
        if has_value_model:
            agent.save_weights(actor_weights, value_weights)
        else:
            agent.save_weights(actor_weights)

    # close the env and write monitor result info to disk
    csvfile.close()
    env.close() 
