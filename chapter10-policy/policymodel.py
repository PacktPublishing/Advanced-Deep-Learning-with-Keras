"""Policy Model for
MountainCarCountinuous-v0 problem


"""

from keras.layers import Dense, Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

import tensorflow as tf

import numpy as np
import gym
from gym import wrappers, logger


class PolicyModel():
    def __init__(self, env, args):

        self.env = env
        self.args = args

        # discount rate
        self.state = env.reset()
        state_size = env.observation_space.shape[0]
        self.state = np.reshape(self.state, [1, state_size])

        n_inputs = env.observation_space.shape[0]
        self.actor_model = None
        self.logprob_model = None
        self.entropy_model = None
        self.value_model = None
        self.build_models(n_inputs)
        beta = 0.0 if args.a3c else 0.0
        loss = self.logprob_loss(self.get_entropy(self.state), beta=beta)
        self.logprob_model.compile(loss=loss, optimizer=Adam(lr=1e-2))
        optimizer = Adam(lr=5e-3)
        loss = 'mse' if args.a3c else self.value_loss
        self.value_model.compile(loss=loss, optimizer=optimizer)


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
        action = K.clip(action,
                        self.env.action_space.low[0],
                        self.env.action_space.high[0])
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
        x = Dense(16,
                  activation='relu',
                  kernel_initializer=kernel_initializer)(inputs)
        x = Dense(16,
                  activation='tanh',
                  kernel_initializer=kernel_initializer)(x)

        value = Dense(1,
                      activation='linear',
                      kernel_initializer=kernel_initializer,
                      name='value')(x)
        self.value_model = Model(inputs, value)
        self.value_model.summary()

        x = Dense(16,
                  activation='relu',
                  kernel_initializer=kernel_initializer)(inputs)
        x = Dense(16,
                  activation='tanh',
                  kernel_initializer=kernel_initializer)(x)
        mean = Dense(1,
                     activation='linear',
                     kernel_initializer=kernel_initializer,
                     name='action_mean')(x)
        stddev = Dense(1,
                       activation='softplus',
                       kernel_initializer=kernel_initializer,
                       name='action_stddev')(x)
        action = Lambda(self.action,
                        output_shape=(1,),
                        name='action')([mean, stddev])
        self.actor_model = Model(inputs, action)
        self.actor_model.summary()

        logprob = Lambda(self.logprob,
                         output_shape=(1,),
                         name='logprob')([mean, stddev, action])
        self.logprob_model = Model(inputs, logprob)
        self.logprob_model.summary()

        entropy = Lambda(self.action,
                         output_shape=(1,),
                         name='entropy')([mean, stddev])
        self.entropy_model = Model(inputs, entropy)
        self.entropy_model.summary()


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
