import gym
#from gym.utils import play

env = gym.make('Breakout-v0')

# Uncomment line below to play the game as a human
#play.play(env, zoom=3)



# Let the environment play, by taking random actions at each timestep
for i in range(3):
    env.reset()
    while True:
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print(f"Iteration {i} with reward: {reward}")
            break

