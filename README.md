# Mastering Pong using DQN 
## (CS 238: Decision Making Under Uncertainty - Capstone Project)

---

## Project Overview
This project implements a **Deep Q-Network (DQN)** to train an agent to play the classic Atari game **Pong** using reinforcement learning. The goal of the agent is to maximize the score by learning optimal paddle movements based on the game's state. 

This project highlights key reinforcement learning concepts, including:
- Neural network-based Q-value approximation.
- Experience replay for stabilizing training.
- Reward shaping to accelerate learning.

---

## Features
- **DQN Architecture**: Uses a convolutional neural network to approximate Q-values.
- **Reward Shaping**: Encourages faster learning by providing rewards for:
  - Tracking the ball (proximity reward).
  - Hitting the ball with the paddle (paddle hit reward).
  - Scoring points (score reward).
- **Experience Replay**: Stores past experiences in a replay buffer to improve sample efficiency and break correlation between samples.
- **Epsilon-Greedy Policy**: Balances exploration and exploitation during training.
- **Metrics Logging**: Tracks and saves performance metrics (e.g., rewards, epsilon) during training.

---

## Technical Details
- **Frameworks**: 
  - Python
  - PyTorch for neural network implementation
  - OpenAI Gym for the Pong environment
- **Key Algorithms**:
  - Deep Q-Learning
  - Epsilon-Greedy Action Selection
- **Architecture**:
  - Convolutional layers for processing game frames.
  - Fully connected layers for Q-value prediction.
