"""Q-Learning to solve FrozenLake-v0 problem


"""

from collections import deque
import numpy as np
import argparse
import os
import time
import gym
from gym import wrappers, logger

class QAgent():
    def __init__(self,
                 observation_space,
                 action_space,
                 demo=False,
                 slippery=False,
                 episodes=40000):
        
        self.action_space = action_space
        # number of columns is equal to number of actions
        col = action_space.n
        # number of rows is equal to number of states
        row = observation_space.n
        # build Q Table with row x col dims
        self.q_table = np.zeros([row, col])

        # discount factor
        self.gamma = 0.9

        # initially 90% exploration, 10% exploitation
        self.epsilon = 0.9
        # iteratively applying decay til 10% exploration/90% exploitation
        self.epsilon_min = 0.1
        self.epsilon_decay = self.epsilon_min / self.epsilon
        self.epsilon_decay = self.epsilon_decay ** (1. / float(episodes))

        # learning rate of Q-Learning
        self.learning_rate = 0.1
        
        # file where Q Table is saved on/restored fr
        if slippery:
            self.filename = 'q-frozenlake-slippery.npy'
        else:
            self.filename = 'q-frozenlake.npy'

        # demo or train mode 
        self.demo = demo
        # if demo mode, no exploration
        if demo:
            self.epsilon = 0

    # determine the next action
    # if random, choose from random action space
    # else use the Q Table
    def act(self, state, is_explore=False):
        # 0 - left, 1 - Down, 2 - Right, 3 - Up
        if is_explore or np.random.rand() < self.epsilon:
            # explore - do random action
            return self.action_space.sample()

        # exploit - choose action with max Q-value
        return np.argmax(self.q_table[state])


    # TD(0) learning (generalized Q-Learning) with learning rate
    def update_q_table(self, state, action, reward, next_state):
        # Q(s, a) += alpha * (reward + gamma * max_a' Q(s', a') - Q(s, a))
        q_value = self.gamma * np.amax(self.q_table[next_state])
        q_value += reward
        q_value -= self.q_table[state, action]
        q_value *= self.learning_rate
        q_value += self.q_table[state, action]
        self.q_table[state, action] = q_value


    # dump Q Table
    def print_q_table(self):
        print(self.q_table)
        print("Epsilon : ", self.epsilon)


    # save trained Q Table
    def save_q_table(self):
        np.save(self.filename, self.q_table)


    # load trained Q Table
    def load_q_table(self):
        self.q_table = np.load(self.filename)


    # adjust epsilon
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id',
                        nargs='?',
                        default='FrozenLake-v0',
                        help='Select the environment to run')
    help_ = "Demo learned Q Table"
    parser.add_argument("-d",
                        "--demo",
                        help=help_,
                        action='store_true')
    help_ = "Frozen lake is slippery"
    parser.add_argument("-s",
                        "--slippery",
                        help=help_,
                        action='store_true')
    help_ = "Exploration only. For baseline."
    parser.add_argument("-e",
                        "--explore",
                        help=help_,
                        action='store_true')
    help_ = "Sec of time delay in UI. Useful for viz in demo mode."
    parser.add_argument("-t",
                        "--delay",
                        help=help_,
                        type=int)
    args = parser.parse_args()

    logger.setLevel(logger.INFO)

    # instantiate a gym environment (FrozenLake-v0)
    env = gym.make(args.env_id)

    # debug dir
    outdir = "/tmp/q-learning-%s" % args.env_id
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    if not args.slippery:
        env.is_slippery = False

    if args.delay is not None:
        delay = args.delay 
    else: 
        delay = 0

    # number of times the Goal state is reached
    wins = 0
    # number of episodes to train
    episodes = 40000

    # instantiate a Q Learning agent
    agent = QAgent(env.observation_space,
                   env.action_space,
                   demo=args.demo,
                   slippery=args.slippery,
                   episodes=episodes)

    if args.demo:
        agent.load_q_table()

    # loop for the specified number of episode
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            # determine the agent's action given state
            action = agent.act(state, is_explore=args.explore)
            # get observable data
            next_state, reward, done, _ = env.step(action)
            # clear the screen before rendering the environment
            os.system('clear')
            # render the environment for human debugging
            env.render()
            # training of Q Table
            if done:
                # update exploration-exploitation ratio
                # reward > 0 only when Goal is reached
                # otherwise, it is a Hole
                if reward > 0:
                    wins += 1

            if not args.demo:
                agent.update_q_table(state, action, reward, next_state)
                agent.update_epsilon()

            state = next_state
            percent_wins = 100.0 * wins / (episode + 1)
            print("-------%0.2f%% Goals in %d Episodes---------"
                  % (percent_wins, episode))
            if done:
                time.sleep(5 * delay)
            else:
                time.sleep(delay)


    print("Episodes: ", episode)
    print("Goals/Holes: %d/%d" % (wins, episode - wins))

    agent.print_q_table()
    if not args.demo and not args.explore:
        agent.save_q_table()
    # close the env and write monitor result info to disk
    env.close() 
