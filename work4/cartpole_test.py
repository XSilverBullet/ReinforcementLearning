# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
from functools import reduce
from matplotlib import pyplot as plt

EPISODES = 1000
TRAJECTORY_LENGTH = 20000
BATCH_SIZE = 64
GAMMA = 0.95
LEARNING_RATE = 0.001
C = 100

class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.9, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = gamma    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu', kernel_initializer='random_uniform'))
        #model.add(Dense(32, activation='relu', kernel_initializer='random_uniform'))
        #model.add(Dense(64, activation='relu'))
        model.add(Dense(16, activation='relu', kernel_initializer='random_uniform'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size ):

        x_batch = []
        y_batch = []

        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:

            y_target = self.model.predict(state)
            if done:
                y_target[0][action] = reward
            else:
                a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                y_target[0][action] = reward + self.gamma * t[np.argmax(a)]

            x_batch.append(state[0])
            y_batch.append(y_target[0])
        hist = self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        mean_loss = hist.history['loss'][0]

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return mean_loss

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)



if __name__=='__main__':
    env = gym.make('CartPole-v0').unwrapped
    # env = gym.make('MountainCar-v0')
    # env = gym.make('Acrobot-v1')

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size, gamma=GAMMA, learning_rate=LEARNING_RATE)
    agent.load("./model/cartpole-ndqn.h5")

    last_time_steps = np.ndarray(0)  # store every trajectory of cartpole
    done = False

    steps = []
    for e in range(EPISODES):
        sum_steps = 0
        #record = VideoRecorder(env=env, path='cartpole.mp4')
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(TRAJECTORY_LENGTH):
            ##env.render()

            #action = agent.model.predict(state)[0]
            next_state, reward, done, _ = env.step(np.argmax(agent.model.predict(state)[0]))
            next_state = np.reshape(next_state, [1, state_size])
            state = next_state

            sum_steps += reward

            if done:
                #print("Episode finished after {} timesteps".format(t + 1))
                break
        print("e: {} steps: {}".format(e, sum_steps))
        last_time_steps = np.append(last_time_steps, [int(sum_steps + 1)])


        # record.close()

    trajectory = last_time_steps.tolist()
    trajectory.sort()

    print("Best Trajectory: {:0.2f}".format(trajectory[-1]))
    print("Meaning score: {:0.2f}".format(last_time_steps.mean()))
    print("Std: {:0.2f}".format(np.std(trajectory[:])))


