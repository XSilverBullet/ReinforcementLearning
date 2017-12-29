import gym
import pandas
import numpy as np
import pandas
from functools import reduce
from work2 import utils, QLearn
from gym.monitoring import VideoRecorder
import math
from matplotlib import pyplot as plt


if __name__ == '__main__':
    env = gym.make('CartPole-v0').unwrapped

    #goal_average_steps = 19500
    max_number_of_steps = 20000
    last_time_steps = np.ndarray(0)
    n_bins = 5
    n_bins_angle = 5

    action_of_num = env.action_space.n
    number_of_features = env.observation_space.shape[0]  # print 4
    last_time_steps = np.ndarray(0)

    #for figure
    x = []
    reward_y = []
    # Number of states is huge so in order to simplify the situation
    # we discretize the space to: 10 ** number_of_features
    cart_position_bins = pandas.cut([-2.4, 2.4], bins=n_bins, retbins=True)[1][1:-1]
    cart_velocity_bins = pandas.cut([-0.419, 0.419], bins=n_bins, retbins=True)[1][1:-1]
    pole_angle_bins = pandas.cut([-0.5, 0.5], bins=n_bins_angle, retbins=True)[1][1:-1]
    angle_rate_bins = pandas.cut([-math.radians(50), math.radians(50)], bins=n_bins_angle, retbins=True)[1][1:-1]

    # The Q-learn algorithm
    qlearn = QLearn.QLearn(actions=range(env.action_space.n), alpha=0.05 , gamma=0.99, epsilon=0.2)
    #print(env.observation_space.high, env.observation_space.high, )

    print("start Traing QLearning...")
    for i_episode in range(20000):
        observation = env.reset()
        rewards_sum = 0

        cart_position, cart_velocity, pole_angle, angle_rate_of_change = observation
        state = utils.build_state([utils.to_bin(cart_position, cart_position_bins),
                                   utils.to_bin(cart_velocity, cart_velocity_bins),
                                   utils.to_bin(pole_angle, pole_angle_bins),
                                   utils.to_bin(angle_rate_of_change, angle_rate_bins)])

        for t in range(max_number_of_steps):
            #env.render()

            # Pick an action based on the current state
            action = qlearn.chooseAction(state)
            # Execute the action and get feedback
            observation, reward, done, info = env.step(action)
            # Digitize the observation to get a state
            cart_position, cart_velocity, pole_angle, angle_rate_of_change = observation
            nextState = utils.build_state([utils.to_bin(cart_position, cart_position_bins),
                                           utils.to_bin(cart_velocity, cart_velocity_bins),
                                           utils.to_bin(pole_angle, pole_angle_bins),
                                           utils.to_bin(angle_rate_of_change, angle_rate_bins)])
            if not (done):
                qlearn.learn(state, action, reward, nextState)
                state = nextState
                rewards_sum += reward
            else:
                # Q-learn stuff
                reward = -200
                rewards_sum += reward

                qlearn.learn(state, action, reward, nextState)
                last_time_steps = np.append(last_time_steps, [int(t + 1)])
                print(t + 1)
                break

        x.append(i_episode)
        reward_y.append(rewards_sum)

    print("Training finished...")

    print("start test...")

    test_reward = []

    for i_episode in range(200):
        observation = env.reset()

        rewards_sum = 0

        cart_position, cart_velocity, pole_angle, angle_rate_of_change = observation
        state = utils.build_state([utils.to_bin(cart_position, cart_position_bins),
                                   utils.to_bin(cart_velocity, cart_velocity_bins),
                                   utils.to_bin(pole_angle, pole_angle_bins),
                                   utils.to_bin(angle_rate_of_change, angle_rate_bins)])

        record = VideoRecorder(env=env, path="cartpolev0.mp4")
        for t in range(max_number_of_steps):
            #env.render()

            # Pick an action based on the current state
            action = 0 if qlearn.getQ(state,0)>qlearn.getQ(state,1) else 1
            # Execute the action and get feedback
            observation, reward, done, info = env.step(action)

            # Digitize the observation to get a state
            cart_position, cart_velocity, pole_angle, angle_rate_of_change = observation
            nextState = utils.build_state([utils.to_bin(cart_position, cart_position_bins),
                                           utils.to_bin(cart_velocity, cart_velocity_bins),
                                           utils.to_bin(pole_angle, pole_angle_bins),
                                           utils.to_bin(angle_rate_of_change, angle_rate_bins)])
            if not (done):
                #qlearn.learn(state, action, reward, nextState)
                state = nextState
                rewards_sum += reward
            else:
                # Q-learn stuff
                reward = -200
                rewards_sum += reward
                break

        print(rewards_sum)
        test_reward.append(rewards_sum)
        record.close()

    print("Test Meaning_value: {}".format(np.mean(test_reward)))
    print("Test std: {}".format(np.std(test_reward)))
    # print("length: ")
    print("Test finished...")



    x = np.asarray(x, dtype=int)
    reward_y = np.asarray(reward_y, dtype=float)

    plt.figure()
    plt.plot(x, reward_y)
    plt.savefig('cartpole.png')
    plt.show()

    trajectory = last_time_steps.tolist()

    trajectory.sort()

    print("All Trajecotries: ")
    print(trajectory)
    print("Best Trajectory: {:0.2f}".format(trajectory[-1]))
    print("Meaning score: {:0.2f}".format(last_time_steps.mean()))
    print("Std: {:0.2f}".format(np.std(trajectory[:])))

    print("Best 100 meaning score: {:0.2f}".format(
        reduce(lambda x, y: x + y, trajectory[-100:]) / len(trajectory[-100:])))
    print("Best 100 Std: {:0.2f}".format(np.std(trajectory[-100:])))
