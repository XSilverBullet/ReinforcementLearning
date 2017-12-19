import gym
import pandas
import numpy as np
import pandas
from functools import reduce
from work2 import utils, QLearn
if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    goal_average_steps = 19500
    max_number_of_steps = 20000
    last_time_steps = np.ndarray(0)
    n_bins = 8
    n_bins_angle = 10

    number_of_features = env.observation_space.shape[0] #print 4
    last_time_steps = np.ndarray(0)

    # Number of states is huge so in order to simplify the situation
    # we discretize the space to: 10 ** number_of_features
    cart_position_bins = pandas.cut([-2.4, 2.4], bins=n_bins, retbins=True)[1][1:-1]
    print(pandas.cut([-2.4, 2.4], bins=n_bins, retbins=True)[1][:])
    pole_angle_bins = pandas.cut([-2, 2], bins=n_bins_angle, retbins=True)[1][1:-1]
    #print(pole_angle_bins)
    cart_velocity_bins = pandas.cut([-1, 1], bins=n_bins, retbins=True)[1][1:-1]
    #print(cart_velocity_bins)
    angle_rate_bins = pandas.cut([-3.5, 3.5], bins=n_bins_angle, retbins=True)[1][1:-1]
    print(angle_rate_bins)

    # The Q-learn algorithm
    qlearn = QLearn.QLearn(actions=range(env.action_space.n), alpha=0.5, gamma=0.90, epsilon=0.1)
    print(env.observation_space.high, env.observation_space.high,)

    for i_episode in range(300):
        observation = env.reset()

        cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation
        state = utils.build_state([utils.to_bin(cart_position, cart_position_bins),
                             utils.to_bin(pole_angle, pole_angle_bins),
                             utils.to_bin(cart_velocity, cart_velocity_bins),
                             utils.to_bin(angle_rate_of_change, angle_rate_bins)])
        # print(to_bin(cart_position, cart_position_bins))
        # print(to_bin(pole_angle, pole_angle_bins))
        # print("state:")
        # print(state)
        #print(qlearn.q)
        for t in range(max_number_of_steps):
            #env.render()

            # Pick an action based on the current state
            action = qlearn.chooseAction(state)
            # Execute the action and get feedback
            observation, reward, done, info = env.step(action)

            # Digitize the observation to get a state
            cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation
            nextState = utils.build_state([utils.to_bin(cart_position, cart_position_bins),
                                       utils.to_bin(pole_angle, pole_angle_bins),
                                       utils.to_bin(cart_velocity, cart_velocity_bins),
                                       utils.to_bin(angle_rate_of_change, angle_rate_bins)])

            if not(done):
                qlearn.learn(state, action, reward, nextState)
                state = nextState
            else:
                # Q-learn stuff
                reward = -200
                qlearn.learn(state, action, reward, nextState)
                last_time_steps = np.append(last_time_steps, [int(t + 1)])
                print(t+1)
                break
    # print("length: ")
    # print(len(qlearn.q))
    trajectory = last_time_steps.tolist()

    trajectory.sort()

    #print(trajectory)
    print("Best Trajectory: {:0.2f}".format(trajectory[-1]))
    print("Meaning score: {:0.2f}".format(last_time_steps.mean()))
    print("Std: {:0.2f}".format(np.std(trajectory[:])))

    print("Best 100 meaning score: {:0.2f}".format(
        reduce(lambda x, y: x + y, trajectory[-100:]) / len(trajectory[-100:])))
    print("Best 100 Std: {:0.2f}".format(np.std(trajectory[-100:])))