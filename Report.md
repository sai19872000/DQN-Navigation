# Banana Collection agent

## Goal
The goal is to train a agent using DQNs to collect all yellow banana's while ignoring blue ones

## Simulation environment

Unity had this enviroment under ML-agents and I had to use the udacity's workspace because of limited constraints.

## Agent Reward 

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction.

Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

- 0 - move forward.
- 1 - move backward.
- 2 - turn left.
- 3 - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.


# Agent training using DQNs

I used DQNs for training the agent. Here is a brief description of the DQN

Deep Q-Networks combines RLs Q learning algorithms (SARSA max) with Deep neural networks to approximate the corrsponding Q table. 
The code is taken from the lunar lander exercise and modified to fit the needs in this project. 
I have used 2 additional improvements to this method as done the exercise:

\item Experiance replay
\item Fixed Qs


# Code walk through

The main file Navigation.ipynb utilizes 2 important files 

- 1) model.py: Contains the neural network in pytorch. I used the MLP with 2 hildden layers each with 128 hidden units with relu activation. These networks use Adam optimizer and a learning rate of 5e^-4
- 2) dqn_agent.py: This contains the training of the agent. Firstly the code initializes the replay buffer then initializes 2 (target and local) instances of NN described by model.py. Then the agent takes a step and stores it in the replay buffer. I set the size to 4 so when the target network updates every 4 steps with the local network. I used the the epsilon-greed policy to make a selection of action

- here are the key parameters used:
```  
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size 
GAMMA = 0.995           # discount factor 
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
```

# Results:

The results meet the basic criterion as shown in the notebook. The agent is able to recieve an average reward of +13 in less than 300 episodes.


# Potential improvements:

- Major improvement would be to use the raw pixels to train the agent using CNNs as shown in the lessons.
- Using Double DQNs
- Prioritized experince replay
- Dueling DQNs


