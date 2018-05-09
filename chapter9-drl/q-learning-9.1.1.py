"""Q-learning to solve a simple world model

Simple world model is made of 6 grids (states)
---------------------------------
|         |          |          |
|  Start  |          |  Goal    |
|         |          |          |
---------------------------------
|         |          |          |
|         |          |  Hole    |
|         |          |          |
---------------------------------

"""

from collections import deque
import numpy as np
import argparse
import os
import time
from termcolor import colored

class QWorld(object):
    def __init__(self):
        # 4 actions
        # 0 - Left, 1 - Down, 2 - Right, 3 - Up
        self.col = 4

        # 6 states
        self.row = 6

        self.q_table = np.zeros([self.row, self.col])

        self.gamma = 0.9
        self.epsilon = 0.9
        self.epsilon_decay = 0.98
        self.epsilon_min = 0.1
        self.init_transition_table()
        self.init_reward_table()
        self.reset()


    def reset(self):
        self.state = 0
        return self.state


    def init_reward_table(self):
        """
        0 - Left, 1 - Down, 2 - Right, 3 - Up
        ----------------
        | 0 | 0 | 100  |
        ----------------
        | 0 | 0 | -100 |
        ----------------
        """
        self.reward_table = np.zeros([self.row, self.col])
        self.reward_table[1, 2] = 100.
        self.reward_table[4, 2] = -100.


    def init_transition_table(self):
        """
        0 - Left, 1 - Down, 2 - Right, 3 - Up
        -------------
        | 0 | 1 | 2 |
        -------------
        | 3 | 4 | 5 |
        -------------
        """
        self.transition_table = np.zeros([self.row, self.col], dtype=int)

        self.transition_table[0, 0] = 0
        self.transition_table[0, 1] = 3
        self.transition_table[0, 2] = 1
        self.transition_table[0, 3] = 0

        self.transition_table[1, 0] = 0
        self.transition_table[1, 1] = 4
        self.transition_table[1, 2] = 2
        self.transition_table[1, 3] = 1

        # self-absorbing state (terminal goal state)
        self.transition_table[2, 0] = 2
        self.transition_table[2, 1] = 2
        self.transition_table[2, 2] = 2
        self.transition_table[2, 3] = 2

        self.transition_table[3, 0] = 3
        self.transition_table[3, 1] = 3
        self.transition_table[3, 2] = 4
        self.transition_table[3, 3] = 0

        self.transition_table[4, 0] = 3
        self.transition_table[4, 1] = 4
        self.transition_table[4, 2] = 5
        self.transition_table[4, 3] = 1

        # self-absorbing state (terminal hole state)
        self.transition_table[5, 0] = 5
        self.transition_table[5, 1] = 5
        self.transition_table[5, 2] = 5
        self.transition_table[5, 3] = 5
        

    def step(self, action):
        next_state = self.transition_table[self.state, action]
        done = True if (next_state == 2 or next_state == 5) else False
        reward = self.reward_table[self.state, action]
        self.state = next_state
        return next_state, reward, done


    def act(self):
        # 0 - Left, 1 - Down, 2 - Right, 3 - Up
        if np.random.rand() <= self.epsilon:
            # explore - do random action
            return np.random.choice(4,1)[0]

        # exploit - choose action with max Q-value
        return np.argmax(self.q_table[self.state])


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

    def print_cell(self, row=0):
        print("")
        for i in range(13):
            j = i - 2
            if j in [0, 4, 8]: 
                if j == 8:
                    if self.state == 2 and row == 0:
                        marker = "\033[4mG\033[0m"
                    elif self.state == 5 and row == 1:
                        marker = "\033[4mH\033[0m"
                    else:
                        marker = 'G' if row == 0 else 'H'
                    color = self.state == 2 and row == 0
                    color = color or (self.state == 5 and row == 1)
                    color = 'red' if color else 'blue'
                    print(colored(marker, color), end='')
                elif self.state in [0, 1, 3, 4]:
                    if self.state == 0 and row == 0 and j == 0:
                        marker = '_'
                    elif self.state == 1 and row == 0 and j == 4:
                        marker = '_'
                    elif self.state == 3 and row == 1 and j == 0:
                        marker = '_'
                    elif self.state == 4 and row == 1 and j == 4:
                        marker = '_'
                    else:
                        marker = ' '
                    print(colored(marker, 'red'), end='')
                else:
                    print(' ', end='')
            elif i % 4 == 0:
                    print('|', end='')
            else:
                print(' ', end='')
        print("")


    def print_world(self, action):
        actions = { 0: "(Left)", 1: "(Down)", 2: "(Right)", 3: "(Up)" }
        print(actions[action])
        for _ in range(13):
            print('-', end='')
        self.print_cell()
        for _ in range(13):
            print('-', end='')
        self.print_cell(row=1)
        for _ in range(13):
            print('-', end='')
        print("")


if __name__ == '__main__':
    episode_count = 3000
    wins = 0
    maxwins = 20
    scores = deque(maxlen=maxwins)
    q_world = QWorld()

    t = 0
    for i in range(episode_count):
        state = q_world.reset()
        done = False
        while not done:
            action = q_world.act()
            next_state, reward, done = q_world.step(action)
            print(state, action, reward, next_state)
            os.system('clear')
            q_world.print_world(action)
            q_world.print_q_table()
            q_world.update_q_table(state, action, reward, next_state)
            state = next_state
            if reward > 0:
                wins += 1
                q_world.update_epsilon()
                scores.append(t)
                t = 0
                if wins > maxwins:
                    print(scores)
                    q_world.print_world(action)
                    q_world.print_q_table()
                    exit(0)
            time.sleep(1)
        t += 1

    print(scores)
    q_world.print_q_table()
