
# AI Doom Player Using Reinforcement Learning

## Project Description
This project aims to leverage reinforcement learning algorithms to train an AI to play Doom effectively. Utilizing environments provided by VizDoom and Gymnasium, the AI is trained under different conditions, offering insights into the capabilities and versatility of reinforcement learning techniques.

## Algorithms
In the `Algorithms` folder, various reinforcement learning algorithms are implemented in Python files. These algorithms are the backbone of our AI's learning process. The folder includes a `PPO.py` file as first algorithm, which implements the Proximal Policy Optimization algorithm.

## Test Environments
- **test_discrete:** This subfolder contains the setup for testing the AI with the `Cartpole-v1` environment from Gymnasium. It serves as a discrete action space for testing the effectiveness of algorithms in simpler scenarios.
- **test_continuous:** This part of the repository deals with a more complex environment, the `Inverted Pendulum` from Gymnasium's Mujoco collection. It provides a continuous action space for testing and refining our AI models.

## BASIC Environment
The `BASIC` folder is dedicated to training the AI in the VizDoom BASIC environment. This setting replicates the scenarios in Doom, allowing the AI to learn and adapt to the game's dynamics, offering a practical application of reinforcement learning in a complex, real-time decision-making process.

## Setup Instructions
This is a work in progress (will get updates frequently), USE AT YOUR OWN PERIL.

## Contribution Guidelines
Contributions are welcome! If you're interested in improving the AI Doom Player or expanding its capabilities, please:
- Fork the repository.
- Create a new branch for your feature.
- Commit your changes.
- Open a pull request with a detailed description of your contribution.
