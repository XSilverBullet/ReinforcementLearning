# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from functools import reduce

EPISODES = 1500
TRAJECTORY_LENGTH = 20000
BATCH_SIZE = 32
GAMMA = 0.9
LEARNING_RATE = 0.001


class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.9, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = gamma  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        # Deep-Q learning Model
        model = Sequential()
        model.add(Dense(256, input_dim=self.state_size, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # returns action
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        # print(act_values)
        # print(np.argmax(act_values[0]))
        return np.argmax(act_values[0])

    # train target changes with origin
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    #env = gym.make('CartPole-v0')
    env = gym.make('MountainCar-v0')
    #env = gym.make('Acrobot-v1')

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size, gamma=GAMMA, learning_rate=LEARNING_RATE)
    # agent.load("./data/cartpole-dqn.h5")

    last_time_steps = np.ndarray(0)  # store every trajectory of cartpole
    done = False

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(TRAJECTORY_LENGTH):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            reward = reward if not done else -100
            next_state = np.reshape(next_state, [1, state_size])

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, epsilon: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                last_time_steps = np.append(last_time_steps, [int(time + 1)])
                break
        if len(agent.memory) > BATCH_SIZE:
            agent.replay(BATCH_SIZE)
        if e % 10 == 0:
            agent.save("./model/cartpole-dqn.h5")

    trajectory = last_time_steps.tolist()

    trajectory.sort()

    print("Best Trajectory: {:0.2f}".format(trajectory[-1]))
    print("Meaning score: {:0.2f}".format(last_time_steps.mean()))
    print("Std: {:0.2f}".format(np.std(trajectory[:])))

    print("Best 100 meaning score: {:0.2f}".format(reduce(lambda x, y: x + y, trajectory[-100:]) / len(trajectory[-100:])))
    print("Best 100 Std: {:0.2f}".format(np.std(trajectory[-100:])))
