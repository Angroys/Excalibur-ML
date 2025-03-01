import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
def run(episodes, render = False, is_training=True):
    # Initialise the environment
    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human" if render else None)

    # Reset the environment to generate the first observation
    
    learning_rate = 0.1
    discount_factor = 0.9

    epsilon = 1
    epsilon_decay_rate = 0.0001
    rng = np.random.default_rng()
    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        with open('frozen_lake.pkl', 'rb') as f:
            q = pickle.load(f)
    rewards_per_episodes = np.zeros(episodes)
    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        while(not terminated and not truncated):
            if rng.random() < epsilon and is_training:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state, :])
            new_state, reward, terminated, truncated, _ = env.step(action)
            if is_training:
                q[state, action] = q[state,action] +learning_rate * (reward +  discount_factor * np.max(q[new_state, :]) - q[state, action])

            state = new_state

            rewards_per_episodes [i]+= reward
        if is_training:
            epsilon = max(0, epsilon - epsilon_decay_rate)
            if epsilon ==0:
                learning_rate = 0.00001
    
    env.close()
    print(q)
    if is_training:
        sum_rewards = np.zeros(episodes)
        for t in range(episodes):
            sum_rewards[t] = np.sum(rewards_per_episodes[max(0, t-100):(t+1)])
        plt.plot(sum_rewards)
        plt.savefig('frozen_lage.png')
        
        with open("frozen_lake.pkl", "wb") as f:
            pickle.dump(q, f)

if __name__ == "__main__":
    run(1500, False, False)