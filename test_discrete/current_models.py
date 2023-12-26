import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class BaseDiscreteModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs):   
        raise AssertionError('Base forward model is not functional')
        return inputs
    
    def categorical(self, inputs):
        logits = self.forward(inputs)
        return torch.distributions.categorical.Categorical(logits=logits)
    
    def sample(self, observation):
        return self.categorical(observation).sample()
    
    def log_prob(self, observations, actions):
        return self.categorical(observations).log_prob(actions) 
    
    def flatten_observation(self, observation):
        return observation
    
    def flatten_action(self, action):
        return action
    
    def unflatten_observation(self, observation):
        return observation
    
    def unflatten_action(self, action):
        return action


class BaseContinuousModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs):   
        raise AssertionError('Base forward model is not functional')
        return inputs
    
    def normal(self, inputs):
        mean, scale = self.forward(inputs)
        #print(outputs)
        return torch.distributions.normal.Normal(loc=mean, scale=scale)
    
    def sample(self, observation):
        return self.normal(observation).sample()
    
    def log_prob(self, observations, actions):
        return self.normal(observations).log_prob(actions)
    
    def flatten_observation(self, observation):
        return observation
    
    def flatten_action(self, action):
        return action
    
    def unflatten_observation(self, observation):
        return observation
    
    def unflatten_action(self, action):
        return action
    

class PolicyModel(BaseDiscreteModel):
    def __init__(self, neurons, activation):
        super().__init__()
        self.neurons = neurons
        self.activation = activation
        
        self.stack = nn.ModuleList([nn.Linear(in_features=neurons[i], out_features=neurons[i+1]) for i in range(len(neurons)-1)])
        
    def forward(self, inputs):
        x = inputs
        
        for num, layer in enumerate(self.stack):
            x = layer(x)
            if num < (len(self.stack)-1):
                x = self.activation(x)
        
        return x

    def flatten_action(self, action):
        return torch.unsqueeze(action, dim=-1)
    
    def unflatten_action(self, action):
        return np.squeeze(action)
    
    def categorical(self, inputs):
        logits = self.forward(inputs)
        return torch.distributions.categorical.Categorical(logits=logits)
    
    def sample(self, observation):
        return self.flatten_action(self.categorical(observation).sample())
    
    def log_prob(self, observations, actions):
        return self.categorical(observations).log_prob(torch.squeeze(actions))
        
current_policy = PolicyModel(neurons=[4, 64, 64, 2], activation=nn.functional.relu)
current_Vn = PolicyModel(neurons=[4, 64, 64, 1], activation=nn.functional.relu)
#https://deepboltzer.codes/policy-types-in-reinforcement-learning