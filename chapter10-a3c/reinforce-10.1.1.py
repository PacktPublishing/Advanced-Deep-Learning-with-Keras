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

        # weights filename
        self.weights_file = 'reinforce_mountaincarcontinuous.h5'
        n_inputs = env.observation_space.shape[0]
        self.action_model, self.log_prob_model = self.build_model(n_inputs)
        # self.pi_model = self.build_model(n_inputs)
        self.log_prob_model.compile(loss=self.loss, optimizer=Adam(lr=1e-5))

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
        mean = Dense(1, activation='linear', name='action_mean')(x)
        stddev = Dense(1, activation='softplus', name='action_stddev')(x)
        action = Lambda(self.action_sample, output_shape=(1,), name='action')([mean, stddev])
        log_prob = Lambda(self.action_log_prob, output_shape=(1,), name='action_log_prob')([mean, stddev, action])
        action_model = Model(inputs, action)
        action_model.summary()
        log_prob_model = Model(inputs, log_prob)
        log_prob_model.summary()
        return action_model, log_prob_model

    def act(self, state):
        outputs = self.action_model.predict(state)
        # print(outputs.shape)
        action = outputs[0]
        return action

    def remember(self, item):
        self.memory.append(item)

    def train_by_step(self, step, state, reward):
        discount_factor = self.gamma**step
        reward *= discount_factor
        target = np.array([reward])
        target = np.reshape(target, [1, 1])
        if reward > 0 or step == 998:
            verbose = 1
        else:
            verbose = 0
        self.log_prob_model.fit(np.array(state),
                          target,
                          batch_size=1,
                          epochs=1,
                          verbose=verbose)

    def train_by_episode(self):
        state_batch, target_batch = [], []
        for item in self.memory:
            step, state, reward = item
            target_batch.append(reward)
            state_batch.append(state)
    
        batch_size = len(self.memory)
        state_batch = np.reshape(state_batch, [batch_size,2])
        print(state_batch.shape)
        self.log_prob_model.fit(state_batch,
                          np.array(target_batch),
                          batch_size=batch_size,
                          epochs=1,
                          verbose=1)


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
    # print(env._max_episode_steps)

    # instantiate agent
    agent = PolicyAgent(env)

    # should be solved in this number of episodes
    episode_count = 1000
    state_size = env.observation_space.shape[0]
    train_by_episode = False

    # sampling and fitting
    for episode in range(episode_count):
        state = env.reset()
        # state is car [position, speed]
        state = np.reshape(state, [1, state_size])
        if train_by_episode:
            agent.reset_memory()
        step = 0
        total_reward = 0
        done = False
        while not done:
            # [min, max] action = [-1.0, 1.0]
            action = agent.act(state)
            env.render()
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                reward = 0
            if train_by_episode:
                item = (step, state, reward)
                agent.remember(item)
            else:
                agent.train_by_step(step, state, reward)
            state = next_state
            state = np.reshape(state, [1, state_size])
            step += 1

        fmt = "Episode=%d, Step=%d. Action=%f, Reward=%f, Total_Reward=%f"
        print(fmt % (episode, step, action[0], reward, total_reward))
        if train_by_episode:
            agent.train_by_episode()


    # close the env and write monitor result info to disk
    env.close() 
