import gym
import pandas
import numpy as np
import pandas
from functools import reduce
from work2 import utils, QLearn
if __name__ == '__main__':
    env = gym.make('Acrobot-v1')

    max_number_of_steps = 2000
    last_time_steps = np.ndarray(0)
    n_bins = 8
    n_bins_angle = 10

    number_of_features = env.observation_space.shape[0] #print 4
    last_time_steps = np.ndarray(0)

    # Number of states is huge so in order to simplify the situation
    # we discretize the space to: 10 ** number_of_features
    c1_bins = pandas.cut([-2.4, 2.4], bins=n_bins, retbins=True)[1][1:-1]
    #print(cart_position_bins)
    p1_bins = pandas.cut([-2, 2], bins=n_bins_angle, retbins=True)[1][1:-1]
    #print(pole_angle_bins)
    c2_bins = pandas.cut([-1, 1], bins=n_bins, retbins=True)[1][1:-1]
    #print(cart_velocity_bins)
    p2_bins = pandas.cut([-3.5, 3.5], bins=n_bins_angle, retbins=True)[1][1:-1]

    c3_bins = pandas.cut([-1, 1], bins=n_bins, retbins=True)[1][1:-1]
    #print(cart_velocity_bins)

    p3_bins = pandas.cut([-3.5, 3.5], bins=n_bins_angle, retbins=True)[1][1:-1]
    #print(angle_rate_bins)

    # The Q-learn algorithm
    qlearn = QLearn.QLearn(actions=range(env.action_space.n), alpha=0.5, gamma=0.90, epsilon=0.1)

    for i_episode in range(300):
        observation = env.reset()

        c1, p1, c2, p2, c3, p3 = observation
        state = utils.build_state([utils.to_bin(c1, c1_bins),
                                   utils.to_bin(p1, p1_bins),
                                   utils.to_bin(c2, c2_bins),
                                   utils.to_bin(p2, p2_bins),
                                   utils.to_bin(c3, c3_bins),
                                   utils.to_bin(p3, p3_bins)])
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
            c1, p1, c2, p2, c3, p3 = observation
            nextState = utils.build_state([utils.to_bin(c1, c1_bins),
                                       utils.to_bin(p1, p1_bins),
                                       utils.to_bin(c2, c2_bins),
                                       utils.to_bin(p2, p2_bins),
                                       utils.to_bin(c3, c3_bins),
                                       utils.to_bin(p3, p3_bins)])

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

    trajectory = last_time_steps.tolist()

    trajectory.sort()

    print("Best Trajectory: {:0.2f}".format(trajectory[-1]))
    print("Meaning score: {:0.2f}".format(last_time_steps.mean()))
    print("Std: {:0.2f}".format(np.std(trajectory[:])))

    print("Best 100 meaning score: {:0.2f}".format(reduce(lambda x, y: x + y, trajectory[-100:]) / len(trajectory[-100:])))
    print("Best 100 Std: {:0.2f}".format(np.std(trajectory[-100:])))