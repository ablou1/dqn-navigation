from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from dqn_agent import DqnAgent, DoubleDqnAgent
from dueling_dqn_agent import DuelingDqnAgent, DuelingDoubleDqnAgent
import torch
import random

# Load the Banana environment
env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

action_size = brain.vector_action_space_size    # number of actions

# examine the state space
state = env_info.vector_observations[0]
state_size = len(state)

# Create the agent to train with the parameters to use
agent = DuelingDqnAgent(state_size=state_size, action_size=action_size, seed=0, batch_size=32, hidden_layer_size=32)


def dqn(agent, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, save_checkpoint=False):
    """Deep Q-Learning.

    Params
    ======
        agent (Agent): agent to train
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    best_score = 13.0                  # Only save the agent if he gets a result better than 13.0
    # Number of episodes needed to solve the environment (mean score of 13 on the 100 last episodes)
    episode_solved = n_episodes
    scores = []                        # list containing scores from each episode
    scores_mean = []                   # List containing mean value of score_window
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset()[brain_name]
        state = env_info.vector_observations[0]            # get the current state
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        scores_mean.append(np.mean(scores_window))
        eps = max(eps_end, eps_decay*eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= best_score:
            episode_solved = min(episode_solved, i_episode-100)
            best_score = np.mean(scores_window)
            if save_checkpoint:
                checkpoint = {'state_size': agent.state_size,
                              'action_size': agent.action_size,
                              'hidden_layer_size': agent.hidden_layer_size,
                              'state_dict': agent.qnetwork_local.state_dict()
                              }
                torch.save(checkpoint, f'{agent.name}_checkpoint.pth')
    if episode_solved < n_episodes:
        print(f'\n{agent.name} - best average score : {best_score} - Environment solved after {episode_solved} episodes')
    return scores, scores_mean


scores, _ = dqn(agent, eps_decay=0.98, save_checkpoint=True)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()


env.close()
