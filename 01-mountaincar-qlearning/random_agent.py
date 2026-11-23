import gymnasium as gym

env = gym.make("MountainCar-v0")

for episode in range(1, 11):
    observation, info = env.reset()
    total_reward = 0.0
    episode_over = False

    while not episode_over:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        episode_over = terminated or truncated

    print(f"Episode {episode}: total reward = {total_reward}")


env.close()
