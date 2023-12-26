#!/usr/bin/env python3
import numpy as np
import pandas as pd
import math
import os
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import gymnasium as gym
import copy
from vizdoom import gymnasium_wrapper # This import will register all the environments
    
def advantage_GAE(observations, actions, rewards, model_Vn, final, discount, lam, device='cpu'):      
    """
    Calculate the Generalized Advantage Estimation (GAE) for given observations, actions, and rewards.

    Parameters:
    observations (np.array): Observations from the environment.
    actions (np.array): Actions taken in the environment.
    rewards (np.array): Rewards received from the environment.
    model_Vn (torch.nn.Module): A neural network model that estimates the value function.
    final (bool): A flag indicating if the final state is terminal.
    discount (float): Discount factor for future rewards.
    lam (float): Lambda parameter for GAE.
    device (str): The device ('cpu' or 'cuda') used for tensor computations.

    Returns:
    tuple: A tuple containing discounted rewards, processed observations,
           processed actions, and calculated advantages.
    """
    
    factor = lam*discount
    observations = torch.tensor(observations, dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.float32, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        
    # Evaluate the value function for each observation
    with torch.no_grad():
        raw_Vn = torch.squeeze(model_Vn(observations))
    
    # Prepare value function estimates for current and next states
    Vn = raw_Vn[:-1]
    Vn1 = raw_Vn[1:]
    if final:
        Vn1[-1] = 0.0 # If final state is terminal, set next state's value to 0
    
    # Compute delta = reward + discount * V(next state) - V(current state)
    delta = rewards + discount*Vn1 - Vn
    
    # Initialize tensors for advantages and discounted rewards
    new_advantages = torch.zeros(len(delta), dtype=torch.float32, device=device) 
    new_advantages[-1] = delta[-1]
    discount_rewards = torch.zeros(len(rewards), dtype=torch.float32, device=device)
    discount_rewards[-1] = rewards[-1] + discount * Vn1[-1]
    
    # Calculate advantages and discounted rewards backwards
    for num in range(len(new_advantages)-2, -1, -1):
        new_advantages[num] = delta[num] + factor*new_advantages[num + 1]
        discount_rewards[num] = rewards[num] + discount * discount_rewards[num+1]
    
    new_observations = observations[:-1]
    new_actions = actions
    
    return discount_rewards, new_observations, new_actions, new_advantages
 

def loss_fn(model, model_old, observation_tensor, action_tensor, weight_tensor, beta, epsilon):
    """
    Calculate the loss for a policy network with entropy regularization.

    Parameters:
    model (torch.nn.Module): The policy model that outputs action probabilities.
    model_old (torch.nn.Module): The old policy model that outputs action probabilities (the model with which the data was generated).
    observation_tensor (torch.Tensor): Tensor of observations.
    action_tensor (torch.Tensor): Tensor of actions taken.
    weight_tensor (torch.Tensor): Tensor of weights (advantages or returns).
    beta (float): Coefficient for entropy regularization.
    epsilon (float): Clipping parameter to clip the policy loss in ppo.

    Returns:
    torch.Tensor: The calculated loss value.
    """
    # Compute the log probability of the actions taken
    logp = torch.squeeze(model.log_prob(observation_tensor, action_tensor))
    with torch.no_grad():
        logp_old = torch.squeeze(model_old.log_prob(observation_tensor, action_tensor))

    # Calculate the ppo loss
    r = torch.exp(logp-logp_old)
    r_clip = torch.clamp(r, min=(1.0-epsilon), max=(1.0+epsilon))
    policy_loss = -(torch.minimum(r*weight_tensor, r_clip*weight_tensor).mean())

    # Calculate the entropy regularization term
    entropy_regularization = -beta * (torch.exp(logp) * logp).mean()

    # The total loss is the sum of policy loss and entropy regularization
    return policy_loss + entropy_regularization 

def sync_agents(model_policy, model_Vn, envs, max_length, lam, discount, last_observations, lengths_final, rewards_final, temp_lengths, temp_rewards, games, flattened_observation_space, flattened_action_space, device):
    """
    use multiple agents in a reinforcement learning environment and computes rewards and advantages.

    Parameters:
    model_policy (torch.nn.Module): The policy model for choosing actions.
    model_Vn (torch.nn.Module): The value network model for evaluating state values.
    envs (gym.vector.VectorEnv): A vector of environments to run the agents in.
    max_length (int): The maximum number of steps to simulate for each agent per cycle.
    lam (float): Lambda parameter used in GAE calculation.
    discount (float): Discount factor for future rewards.
    last_observations (np.array): Last observed states from the environment.
    lengths_final (np.array): Array to record the final lengths of each episode.
    rewards_final (np.array): Array to record the final rewards of each episode.
    temp_lengths (np.array): Temporary array to hold lengths for ongoing episodes.
    temp_rewards (np.array): Temporary array to hold rewards for ongoing episodes.
    games (np.array): Array to record the number of games played per agent per cycle. 
    flattened_observation_space (int): Size of the flattened observation space.
    flattened_action_space (int): Size of the flattened action space.
    device (str): Device for PyTorch tensors ('cpu' or 'cuda').

    Returns:
    tuple: Tuple containing tensors for discounted rewards, new observations, new actions, 
           new advantages, final lengths and rewards, and temporary lengths and rewards.
    """
    
    # Initialize numpy arrays to store observations, actions, and rewards
    observation_list = np.full((max_length+1, len(rewards_final), flattened_observation_space), np.nan)
    action_list = np.full((max_length, len(rewards_final), flattened_action_space), np.nan)
    reward_list = np.full((max_length, len(rewards_final)), np.nan)
    
    # Lists to store rewards, observations, actions, and advantages for training
    discount_rewards_list = []
    new_observations_list = []
    new_actions_list = []
    new_advantages_list = []
    
    step = 0
    
    while step < max_length:
        # Sample actions
        with torch.no_grad():
            actions = model_policy.sample(torch.tensor(last_observations, dtype=torch.float32, device=device)).cpu().numpy()
        
        # Step through the environment with the chosen actions and record the observations, rewards and actions taken
        observation_list[step,:,:] = last_observations.copy()
        observations, rewards, termination, truncation, infos = envs.step(model_policy.unflatten_action(actions))      
        final = (termination | truncation)
        action_list[step,:,:] = actions
        reward_list[step,:] = rewards
        last_observations = model_policy.flatten_observation(observations)
        final_idx = np.nonzero(final)[0]
        
        # Process agents that have reached a final state
        for agent_idx in final_idx:
            num_idx = (~np.isnan(reward_list[:,agent_idx]))    
            observation_list[step+1,:,:] = last_observations.copy()
            observation_idx = (~np.isnan(observation_list[:,agent_idx,0]))
            
            # Calculate advantages and rewards
            if sum(num_idx) > 1:
                discount_rewards, new_observations, new_actions, new_advantages = advantage_GAE(observation_list[observation_idx,agent_idx,:], action_list[num_idx,agent_idx,:], reward_list[num_idx,agent_idx], model_Vn, True, discount, lam, device)
                discount_rewards_list.append(discount_rewards)
                new_observations_list.append(new_observations)
                new_actions_list.append(new_actions)
                new_advantages_list.append(new_advantages)
            
            # Update the final lengths and rewards, and reset the temporary counters
            lengths_final[agent_idx] = sum(num_idx) + temp_lengths[agent_idx]
            rewards_final[agent_idx] = sum(reward_list[num_idx,agent_idx]) + temp_rewards[agent_idx]            
            temp_lengths[agent_idx] = 0
            temp_rewards[agent_idx] = 0
            games[agent_idx] += 1
            observation_list[:,agent_idx,:] = np.nan
            action_list[:,agent_idx] = np.nan
            reward_list[:,agent_idx] = np.nan
            
        step +=1
    
    not_final_idx = np.nonzero(~final)[0]
    # Process agents that have not yet reached a final state
    for agent_idx in not_final_idx:
        num_idx = (~np.isnan(reward_list[:,agent_idx]))        
        observation_list[max_length,:,:] = last_observations.copy()
        observation_idx = (~np.isnan(observation_list[:,agent_idx,0]))
        
        # Calculate advantages and rewards
        if sum(num_idx) > 1:
            discount_rewards, new_observations, new_actions, new_advantages = advantage_GAE(observation_list[observation_idx,agent_idx,:], action_list[num_idx,agent_idx,:], reward_list[num_idx,agent_idx], model_Vn, True, discount, lam, device)
            discount_rewards_list.append(discount_rewards)
            new_observations_list.append(new_observations)
            new_actions_list.append(new_actions)
            new_advantages_list.append(new_advantages)
        
        # Update the temporary lengths and rewards
        temp_lengths[agent_idx] += sum(num_idx)
        temp_rewards[agent_idx] += sum(reward_list[num_idx,agent_idx])
    
    # Concatenate and return the collected data
    return torch.cat(discount_rewards_list, dim=0), torch.cat(new_observations_list, dim=0), torch.cat(new_actions_list, dim=0), torch.cat(new_advantages_list, dim=0), lengths_final, rewards_final, temp_lengths, temp_rewards, games, last_observations
        

def train(model_policy, model_Vn, save_name, env_name, lam, discount, beta, epsilon,
          n_cycles, n_epochs, batch_size, n_agents=1, max_length=64, asynchronous=False,
          report_updates=None, gradient_clip_policy=None, gradient_clip_Vn=None, median_stop_threshold=None,
          median_stop_patience=None, length=False, save_cycles=None, 
          policy_lr=0.001, policy_beta1=0.9, policy_beta2=0.999, policy_eps=1e-08, 
          Vn_lr=0.001, Vn_beta1=0.9, Vn_beta2=0.999, Vn_eps=1e-08, device='cpu', inference_device='cpu'):
    """
    Trains policy and value network models using Proximal Policy Optimization (PPO).

    Parameters:
    model_policy (torch.nn.Module): Policy model to train.
    model_Vn (torch.nn.Module): Value network model to train.
    save_name (str): Base name for saving model states and the outputs.
    env_name (str): Name of the Gymnasium environment to use.
    lam (float): Lambda parameter for GAE.
    discount (float): Discount factor for rewards.
    beta (float): Coefficient for entropy regularization in PPO.
    epsilon (float): Clipping parameter for PPO.
    n_cycles (int): Number of training cycles.
    n_epochs (int): Number of training epochs per cycle.
    batch_size (int): Size of the training batch.
    n_agents (int): Number of agents in the environment.
    max_length (int): Maximum length of an episode.
    asynchronous (bool): Flag for asynchronous environment execution.
    report_updates (int): Frequency of reporting training progress.
    gradient_clip_policy (float): Gradient clipping value for policy model.
    gradient_clip_Vn (float): Gradient clipping value for value network.
    median_stop_threshold (float): Threshold for early stopping based on median reward.
    median_stop_patience (int): Number of cycles to wait for early stopping (the threshold nneds to be beaten for the length of the patience period).
    length (bool): Flag to use length instead of reward for reporting and stopping.
    save_cycles (int): Frequency of saving model states and statistics.
    policy_lr (float): Learning rate for the policy optimizer.
    policy_beta1 (float): Beta1 for the policy optimizer.
    policy_beta2 (float): Beta2 for the policy optimizer.
    policy_eps (float): Epsilon for the policy optimizer.
    Vn_lr (float): Learning rate for the value network optimizer.
    Vn_beta1 (float): Beta1 for the value network optimizer.
    Vn_beta2 (float): Beta2 for the value network optimizer.
    Vn_eps (float): Epsilon for the value network optimizer.
    device (str): Device for training ('cpu' or 'cuda').
    inference_device (str): Device for inference ('cpu' or 'cuda').

    Returns:
    tuple: Arrays of summed rewards and lengths for each cycle.
    """
    
    # Initialize arrays to store rewards, lengths, and number of games played
    reward_sum = np.zeros((n_cycles, n_agents))
    reward_len = np.zeros((n_cycles, n_agents))
    games_played = np.zeros((n_cycles, n_agents))
    
    # Temporary storage for ongoing episode data
    rewards = np.zeros((n_agents))
    lengths = np.zeros((n_agents))
    games = np.zeros((n_agents))
    temp_rewards = np.zeros((n_agents))
    temp_lengths = np.zeros((n_agents))
    
    # Setting up optimizers for the policy and value networks
    optimizer_policy = torch.optim.Adam(model_policy.parameters(), lr=policy_lr, betas=(policy_beta1, policy_beta2), eps=policy_eps)
    optimizer_Vn = torch.optim.Adam(model_Vn.parameters(), lr=Vn_lr, betas=(Vn_beta1, Vn_beta2), eps=Vn_eps)
    
    # Transfer models to the inference device
    model_policy, model_Vn = model_policy.to(inference_device), model_Vn.to(inference_device)
    
    # Calculate the dimensions for the flatened observation and action arrays
    test_env = gym.make(env_name)
    test_obs,_ = test_env.reset()    
    flattened_observation = model_policy.flatten_observation(test_obs)
    flattened_observation_space = flattened_observation.shape[-1] 
    test_action = model_policy.sample(torch.tensor(flattened_observation, dtype=torch.float32, device=inference_device)).cpu().numpy()
    flattened_action_space = test_action.shape[-1]
    
    # Create the environments for training
    envs = gym.vector.make(env_name, num_envs=n_agents, asynchronous=asynchronous)
    last_observations, _ = envs.reset()
    last_observations = model_policy.flatten_observation(last_observations)
    
    for cycle in range(n_cycles):  
        # Synchronize agents and gather training data
        discount_rewards, observations, actions, advantages, lengths, rewards, temp_lengths, temp_rewards, games, last_observations = sync_agents(model_policy, model_Vn, envs, max_length, lam, discount, last_observations, lengths, rewards, temp_lengths, temp_rewards, games, flattened_observation_space, flattened_action_space, inference_device)
        
        # Update rewards, lengths, and games played
        reward_sum[cycle,:] = rewards
        reward_len[cycle,:] = lengths        
        games_played[cycle,:] = games.copy()
        
        # Transfer models to the training device if different from the inference device
        if inference_device != device:
            model_policy, model_Vn = model_policy.to(device), model_Vn.to(device)
        
        # Determine the number of batches
        batch_num = 1
        if len(observations) > batch_size:
            batch_num = int(len(observations)/batch_size)        
        model_old = copy.deepcopy(model_policy)
        
        for epoch in range(n_epochs):
            # Shuffle the observations and process each batch
            idx = np.random.choice(len(observations), len(observations), replace=False).astype(int)
            for i in range(batch_num):
                current_idx = idx[i*batch_size:(i+1)*batch_size]
                
                # Zero gradients
                optimizer_policy.zero_grad()    
                # Compute and backpropagate the policy loss                
                loss_policy = loss_fn(model_policy, model_old, observations[current_idx].to(device), actions[current_idx].to(device), advantages[current_idx].to(device), beta, epsilon)    
                loss_policy.backward()
                if gradient_clip_policy is not None:
                    nn.utils.clip_grad_norm_(model_policy.parameters(), gradient_clip_policy)
                optimizer_policy.step()
                
                
                # Zero gradients for both models
                optimizer_Vn.zero_grad()
                # Compute and backpropagate the value network loss
                Vn = torch.squeeze(model_Vn(observations[current_idx].to(device)))
                loss_Vn = nn.MSELoss()(Vn, discount_rewards[current_idx].to(device))
                loss_Vn.backward()
                if gradient_clip_Vn is not None:
                    nn.utils.clip_grad_norm_(model_Vn.parameters(), gradient_clip_Vn)
                optimizer_Vn.step()
        
        # Transfer models back to the inference device if necessary
        if inference_device != device:
            model_policy, model_Vn = model_policy.to(inference_device), model_Vn.to(inference_device)
        
        # Reporting
        if report_updates is not None and cycle>0 and cycle%report_updates==0:
            if length:
                median_reward = np.quantile(reward_len[cycle,:], 0.5)
                q1_reward = np.quantile(reward_len[cycle,:], 0.25)
                q3_reward = np.quantile(reward_len[cycle,:], 0.75)
                print("cycle: " + str(cycle) + ' length: ' + str(median_reward) + " - q1:"+str(q1_reward)+ " - q3:"+str(q3_reward))
            else:
                median_reward = np.quantile(reward_sum[cycle,:], 0.5)
                q1_reward = np.quantile(reward_sum[cycle,:], 0.25)
                q3_reward = np.quantile(reward_sum[cycle,:], 0.75)
                print("cycle: " + str(cycle) + ' reward: ' + str(median_reward) + " - q1:"+str(q1_reward)+ " - q3:"+str(q3_reward))
        
        # Early stopping
        if median_stop_threshold is not None and median_stop_patience is not None and cycle >= median_stop_patience:
            if length:
                if sum(np.quantile(reward_len[cycle+1-median_stop_patience:cycle+1,:], 0.5, axis=-1) >= median_stop_threshold) >= median_stop_patience:
                    break
            else:
                if sum(np.quantile(reward_sum[cycle+1-median_stop_patience:cycle+1,:], 0.5, axis=-1) >= median_stop_threshold) >= median_stop_patience:
                    break
        
        # Saving model and results            
        if save_cycles is not None and cycle%save_cycles==0:
            torch.save(model_policy.state_dict(), save_name + "_policy.pt")
            torch.save(model_Vn.state_dict(), save_name + "_Vn.pt")
            np.savetxt(save_name + "_episoderewards.csv", reward_sum[:cycle+1,:], delimiter=',')
            np.savetxt(save_name + "_episodelength.csv", reward_len[:cycle+1,:], delimiter=',')
            np.savetxt(save_name + "_gamesplayed.csv", games_played[:cycle+1,:], delimiter=',')
    return reward_sum[:cycle+1,:], reward_len[:cycle+1,:]

if __name__ == "__main__":
    # Importing necessary models and argparse for command-line argument parsing
    from current_models import current_policy, current_Vn
    import argparse
    
    # Setting up an argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=str, default='PPO_settings.txt')
    parser.add_argument('--exp_name', type=str, default='InvertedPendulum')
    args = parser.parse_args()
    
    settings = {}
    # Reading and updating settings from the provided CSV file
    if os.path.exists(args.settings):        
        settings_table = pd.read_csv(args.settings, sep=',', header=0)
        settings = {k:v for k, v in zip(settings_table.iloc[:,0], settings_table.iloc[:,1])} 
    
    # Processing the settings to ensure correct data types
    for key, value in settings.items():
        # Handling empty or 'None' string values
        if isinstance(value, str):
            if value.lower() in ['', 'none', 'na', 'nan']:
                settings[key] = None
        elif pd.isna(value):
            settings[key] = None
        
        # Converting settings to appropriate data types
        if not key in ["env_name", "device", "inference_device"] and value is not None:
            if key in ["n_cycles", "report_updates", "median_stop_patience", "n_agents", "n_epochs", "batch_size", "max_length", "save_cycles"]:
                settings[key] = int(value)
            elif key in ["length", "asynchronous"]:
                settings[key] = (value.lower()=="true")
            else:
                settings[key] = float(value)
    
    # Loading the models for training    
    model_policy = current_policy
    model_Vn = current_Vn
    
    # Training the models with the specified settings
    reward_sum, reward_len = train(model_policy, model_Vn, args.exp_name, **settings)
    
    # Saving the training results and model states
    np.savetxt(args.exp_name + "_episoderewards.csv", reward_sum, delimiter=',')
    np.savetxt(args.exp_name + "_episodelength.csv", reward_len, delimiter=',')    
    torch.save(model_policy.state_dict(), args.exp_name + "_policy.pt")
    torch.save(model_Vn.state_dict(), args.exp_name + "_Vn.pt")