"""Trains an A3C agent to solve the
MountainCarCountinuous-v0 problem


"""

import gym
from gym import wrappers, logger
from policymodel import PolicyModel
import numpy as np
import argparse
import sys
import csv


class A3CAgent():
    def __init__(self, env, args):

        self.env = env
        self.args = args
        self.gamma = 0.99
        # experience buffer
        self.memory = []
        self.model = PolicyModel(env, args)

    def train(self, step, state, next_state, reward):
        self.model.state = state
        next_value = self.model.value(next_state)[0]
        delta = reward + self.gamma*next_value
        delta -= self.model.value(state)[0] 
        delta = np.reshape(delta, [1, 1])
        verbose = 1 if step == 0 else 0
        self.model.value_model.fit(np.array(state),
                                   delta,
                                   batch_size=1,
                                   epochs=1,
                                   verbose=verbose)
        self.model.logprob_model.fit(np.array(state),
                                     delta,
                                     batch_size=1,
                                     epochs=1,
                                     verbose=verbose)

    def reset_memory(self):
        self.memory = []
   

    def remember(self, item):
        self.memory.append(item)


    def train_by_episode(self, last_value=0):
        state_batch, delta_batch = [], []
        batch_size = len(self.memory)
        r = last_value
        
        # self.model.state = state
        for item in self.memory:
            step, state, next_state, reward, done = item
            r = reward + self.gamma*r 
            delta = r - self.model.value(state)[0] 
            delta_batch.append(delta)
            state_batch.append(state)
    
        state_batch = np.reshape(state_batch, [batch_size,2])
        delta_batch = np.reshape(delta_batch, [batch_size,1])
        self.model.value_model.fit(state_batch,
                          delta_batch,
                          batch_size=batch_size,
                          epochs=1,
                          verbose=1,
                          shuffle=False)

        self.model.logprob_model.fit(state_batch,
                          delta_batch,
                          batch_size=batch_size,
                          epochs=1,
                          verbose=1,
                          shuffle=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id',
                        nargs='?',
                        default='MountainCarContinuous-v0',
                        help='Select the environment to run')
    parser.add_argument("-a",
                        "--a3c",
                        default=True,
                        action='store_true',
                        help="A3C enabled by default")
    parser.add_argument("-r",
                        "--random",
                        action='store_true',
                        help="Random action policy")
    parser.add_argument("-w",
                        "--weights",
                        help="Load pre-trained actor model weights")
    args = parser.parse_args()

    logger.setLevel(logger.ERROR)
    env = gym.make(args.env_id)

    postfix = 'a3c'
    filename = "actor_weights-%s-%s.h5" % (postfix, args.env_id)
    outdir = "/tmp/%s-%s" % (postfix, args.env_id)
    csvfilename = "%s.csv" % postfix
    csvfile = open(csvfilename, 'w', 1)
    writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    writer.writerow(['Episode','Step','Total Reward','Number of Episodes Solved'])

    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    
    # instantiate agent
    agent = A3CAgent(env, args)
    train = True
    if args.weights:
        agent.model.load_actor_weights(args.weights)
        train = False

    # should be solved in this number of episodes
    episode_count = 100
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
        while not done:
            # [min, max] action = [-1.0, 1.0]
            if args.random:
                action = env.action_space.sample()
            else:
                action = agent.model.act(state)
            env.render()
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            item = (step, state, next_state, reward, done)
            agent.remember(item)
            total_reward += reward
            if not args.random and train and done:
                #agent.train(step, state, next_state, reward)
                last_value = 0 if reward > 0 else agent.model.value(state)
                agent.train_by_episode(last_value=last_value)
                agent.reset_memory()
            state = next_state
            step += 1

        if reward > 0:
            n_solved += 1
        fmt = "Episode=%d, Step=%d. Action=%f, Reward=%f, Total_Reward=%f"
        print(fmt % (episode, step, action[0], reward, total_reward))
        writer.writerow([episode, step, total_reward, n_solved])

    if not args.weights and not args.random:
        agent.model.save_actor_weights(filename)

    # close the env and write monitor result info to disk
    csvfile.close()
    env.close() 
