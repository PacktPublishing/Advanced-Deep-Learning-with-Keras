
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from collections import deque
import numpy as np
import random
import argparse
import logging
import sys
import gym
from gym import wrappers, logger


class DQNAgent(object):
    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space
        self.build_model()
        self.memory = deque()
        self.gamma = 0.9    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999
        
    def build_model(self):
        inputs = Input(shape=(self.state_space.shape[0], ), name='q_input')
        x = Dense(32, activation='relu')(inputs)
        x = Dense(32, activation='relu')(x)
        x = Dense(self.action_space.n, activation='linear')(x)
        self.model = Model(inputs, x)
        self.model.compile(loss='mse', optimizer=Adam())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # explore
            act = self.action_space.sample()
            return act

        # exploit
        act = self.model.predict(state)
        act = np.argmax(act[0])
        return act

    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    def replay(self, batch_size):
        mini_batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                q_value = np.amax(self.model.predict(next_state)[0])
                target += (self.gamma * q_value)
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        self.update_epsilon()

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id',
                        nargs='?',
                        default='CartPole-v0',
                        help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logging.DEBUG or logging.WARN if you
    # want to change the amount of output.
    logger.setLevel(logging.ERROR)

    env = gym.make(args.env_id)
    env._max_episode_steps = 4000

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = "/tmp/dqn-gym-%s" % args.env_id
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = DQNAgent(env.observation_space, env.action_space)

    episode_count = 3000
    state_size = env.observation_space.shape[0]
    batch_size = 32
    max_t = 0

    for i in range(episode_count):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        t = 0
        while True:
            # in CartPole, action=0 is left and action=1 is right
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -1
            # in CartPole:
            # state = [pos, vel, theta, angular speed]
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            t += 1
            if done:
                # max_t = t if t > max_t else max_t
                if t > max_t:
                    max_t = t
                    env.render()
                print("episode: {}/{}, score: {}, max score: {}, exploration: {:.2}"
                       .format(i, episode_count, t, max_t, agent.epsilon))
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.
        if len(agent.memory) >= batch_size:
            agent.replay(batch_size)
            

    # close the env and write monitor result info to disk
    env.close() 
    print("Max score: ", max_t)
