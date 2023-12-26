# Test Discrete Subfolder

## Overview
The `test_discrete` subfolder is focused on testing and refining reinforcement learning algorithms in continuous action space environments. This subfolder contains scripts and configurations for testing in the `InvertedPendulum-v4` environment from Gymnasium, serving as a preliminary testing ground for the algorithms developed in the `Algorithms` subfolder.

## Contents
- **current_models.py:** A script containing the models that interfaces with the chosen reinforcement learning algorithm to test its performance in the InvertedPendulum-v4 environment.
- **PPO_settings.txt:** Configuration settings for the Proximal Policy Optimization (PPO) algorithm when used in the InvertedPendulum-v4 environment.

## Adding New Tests
To add new tests or modify existing ones:
1. Develop or update the test scripts.
2. If a new algorithm is being tested, ensure it's added to the `Algorithms` subfolder.
3. Update or add any necessary configuration files.
4. Document your changes in this README.

## Contribution
Contributions to enhance the test environments or to add new ones are encouraged. Please adhere to the main project's contribution guidelines.
