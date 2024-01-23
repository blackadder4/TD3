import gym
import numpy as np
import matplotlib.pyplot as plt
from Agent import Agent  # Replace with your actual import
from UTILS.Noise import OUActionNoise
from UTILS.manage_memory import manage_memory
from UTILS.plot_learning_curve import plot_learning_curve

if __name__ == '__main__':
    # env = gym.make('BipedalWalker-v3')
    manage_memory()
    env = gym.make('LunarLanderContinuous-v2')
    agent = Agent(alpha=0.001, beta=0.002, gamma=0.99, n_actions=env.action_space.shape[0],
                  fc1_dims=400, fc2_dims=300, max_size=1000000, tau=0.005, batch_size=64
                  ,checkpoint='Models/', env=env,noise_mu=0.0, noise_theta=0.15, noise_dt=1e-2)
    n_games = 1000
    filename = 'plots/' + 'lunar_lander_' + str(n_games) + '_games.png'

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False
    if load_checkpoint:
        agent.load_models()

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.store_transition(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            score += reward
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
        print('episode {} score {:.1f} avg score {:.1f}'.
              format(i, score, avg_score))
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, filename)
