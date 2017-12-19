import gym

env = gym.make('CartPole-v0')
env = gym.make('MountainCar-v0')
#env = gym.make('Acrobot-v1')

for i_episode in range(20):
    observation = env.reset()
    #print(observation)
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action=action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

#print(env.action_space.high)
# print(env.observation_space.low)
# print(env.observation_space.high)
# from gym import spaces
# space = spaces.Discrete(8)
# x = space.sample()
# assert space.contains(x)
# assert space.n == 8
#
# from gym import envs
# print(envs.registry.all())
#print(env.observation_space.shape[0])