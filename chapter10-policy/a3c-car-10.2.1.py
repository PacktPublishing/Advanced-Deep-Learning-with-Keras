"""Trains an A3C agent to solve the
MountainCarCountinuous-v0 problem


"""

from keras import backend as K
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
        self.get_value_gradients = self.get_gradients(self.model.value_model)
        self.get_logp_gradients = self.get_gradients(self.model.logp_model)


    def reset_memory(self):
        self.memory = []
   

    def remember(self, item):
        self.memory.append(item)


    def train_by_step(self, item):
        step, state, next_state, reward, done = item
        r = self.model.value(next_state)[0] 
        r = reward + self.gamma*r 
        delta = r - self.model.value(state)[0] 
        r = np.reshape(r, [1, 1])
        delta = np.reshape(delta, [1, 1])
        verbose = 1 if done else 0
        self.model.value_model.fit(np.array(state), r, batch_size=1, verbose=verbose)
        # self.model.logp_model.fit(np.array(state), delta, batch_size=1, verbose=verbose)


    def train(self, last_value=0):
        state_batch, delta_batch, r_batch = [], [], []
        batch_size = len(self.memory)
        r = last_value
        
        value_loss = 0.0
        logp_loss = 0.0
        for item in self.memory[::-1]:
            step, action, state, next_state, reward, done = item
            r = reward + self.gamma*r 
            self.model.state = state
            delta = r - self.model.value(state)[0] 

            r = np.reshape(r, [-1, 1])
            value_loss += self.model.value_model.train_on_batch(state, r)

            delta = np.reshape(delta, [-1, 1])
            logp_loss += self.model.logp_model.train_on_batch(state, delta)

        value_loss /= len(self.memory)
        logp_loss /= len(self.memory)
        print("Value loss: %f, Logp loss: %f" % (value_loss, logp_loss))


    def get_gradients(self, model):
        weights = model.trainable_weights # weight tensors
        gradients = model.optimizer.get_gradients(model.total_loss, weights) # gradient tensors
        inputs = model.inputs + model.sample_weights + model.targets + [K.learning_phase()]
        return K.function(inputs=inputs, outputs=gradients)


    def apply_gradients(self, model, gradients, lr=1e-3):
        weights = model.get_weights() 
        # print(weights.shape)
        # print(gradients.shape)
        weights = np.add(weights, -lr*gradients)
        model.set_weights(weights)
        

    def train_by_gradient(self, last_value=0):
        state_batch, delta_batch, r_batch = [], [], []
        batch_size = len(self.memory)
        r = last_value
        
        value_gradients = None
        logp_gradients = None

        for item in self.memory[::-1]:
            step, action, state, next_state, reward, done = item
            r = reward + self.gamma*r 
            self.model.state = state
            delta = r - self.model.value(state)[0] 

            r = np.reshape(r, [-1, 1])
            inputs = [state, [1], r, 1]
            g = self.get_value_gradients(inputs)
            value_gradients = g if value_gradients is None else np.add(g, value_gradients)

            delta = np.reshape(delta, [-1, 1])
            inputs = [state, [1], delta, 1]
            g = self.get_logp_gradients(inputs)
            logp_gradients = g if logp_gradients is None else np.add(g, logp_gradients)

        self.apply_gradients(self.model.value_model, value_gradients, lr=1e-4)        
        self.apply_gradients(self.model.logp_model, logp_gradients, lr=1e-3)        



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
    episode_count = 1000
    state_size = env.observation_space.shape[0]
    n_solved = 0 

    # sampling and fitting
    for episode in range(episode_count):
        # state is car [position, speed]
        state = env.reset()
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
            item = (step, action, state, next_state, reward, done)
            agent.remember(item)
            total_reward += reward
            if not args.random and train and done:
                last_value = 0 if reward > 0 else agent.model.value(next_state)
                agent.train(last_value=last_value)
                # agent.train_by_gradient(last_value=last_value)
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
