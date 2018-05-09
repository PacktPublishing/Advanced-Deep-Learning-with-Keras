"""Q-learning to solve FrozenLake-v0 problem


"""

from collections import deque
import numpy as np
import argparse
import os
import time
import gym
from gym import wrappers, logger

class QAgent(object):
    def __init__(self, observation_space, action_space):
        self.action_space = action_space
        col = action_space.n
        row = observation_space.n
        self.q_table = np.zeros([row, col])
        self.gamma = 0.9
        self.epsilon = 0.9
        self.epsilon_decay = 0.98
        self.epsilon_min = 0.1


    def act(self, state):
        # 0 - left, 1 - Down, 2 - Right, 3 - Up
        if np.random.rand() <= self.epsilon:
            # explore - do random action
            return self.action_space.sample()

        # exploit - choose action with max Q-value
        return np.argmax(self.q_table[state])


    def update_q_table(self, state, action, reward, next_state):
        # Q(s, a) = reward + gamma * max_a' Q(s', a')
        q_value = self.gamma * np.amax(self.q_table[next_state])
        q_value += reward
        self.q_table[state, action] = q_value


    def print_q_table(self):
        print(self.q_table)
        print("Epsilon : ", self.epsilon)


    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id',
                        nargs='?',
                        default='FrozenLake-v0',
                        help='Select the environment to run')
    args = parser.parse_args()

    logger.setLevel(logger.INFO)

    env = gym.make(args.env_id)
    env.render()

    outdir = "/tmp/dqn-%s" % args.env_id
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = QAgent(env.observation_space, env.action_space)

    episode_count = 3000
    wins = 0
    maxwins = 20
    scores = deque(maxlen=maxwins)

    t = 0
    for i in range(episode_count):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            os.system('clear')
            env.render()
            agent.update_q_table(state, action, reward, next_state)
            state = next_state
            if reward > 0:
                wins += 1
                agent.update_epsilon()
                scores.append(t)
                t = 0
                if wins > maxwins:
                    print(scores)
                    agent.print_q_table()
                    env.close()
                    exit(0)
        t += 1
        # time.sleep(1)

    print(scores)
    agent.print_q_table()
    # close the env and write monitor result info to disk
    env.close() 
