from unityagents import UnityEnvironment
import numpy as np
from dqn_agent import DqnAgent
from agent import Agent
import torch

AGENT_CLASS = DqnAgent

# Load the Banana environment
env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    assert checkpoint['state_size'] == state_size
    assert checkpoint['action_size'] == action_size
    agent = AGENT_CLASS(state_size, action_size, 0,
                        hidden_layer_size=checkpoint['hidden_layer_size'], mode=Agent.PLAYING)
    agent.qnetwork_local.load_state_dict(checkpoint['state_dict'])
    return agent


agent = load_checkpoint(f'{AGENT_CLASS.__name__}_checkpoint.pth')
env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
state = env_info.vector_observations[0]            # get the current state
score = 0                                          # initialize the score
while True:
    action = agent.act(state)                      # select an action
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]                  # see if episode has finished
    score += reward                                # update the score
    state = next_state                             # roll over the state to next time step
    if done:                                       # exit loop if episode finished
        break

print("Score: {}".format(score))

env.close()
