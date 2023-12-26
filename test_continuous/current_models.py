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
    

class VnModel(nn.Module):
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
    

class PolicyModel(BaseContinuousModel):#BaseDiscreteModel):
    def __init__(self, neurons, activation, scale=1):
        super().__init__()
        self.neurons = neurons
        self.activation = activation
        self.scale = scale
        
        self.stack = nn.ModuleList([nn.Linear(in_features=neurons[i], out_features=neurons[i+1]) for i in range(len(neurons)-1)])
        
    def forward(self, inputs):
        x = inputs
        
        for num, layer in enumerate(self.stack):
            x = layer(x)
            if num < (len(self.stack)-1):
                x = self.activation(x)
        
        return x, self.scale


class PolicyModelStd(BaseContinuousModel):#BaseDiscreteModel):
    def __init__(self, neurons, activation, mean_neurons, scale_neurons):
        super().__init__()
        self.neurons = neurons
        self.activation = activation
        self.mean_neurons = [neurons[-1]] + mean_neurons
        self.scale_neurons = [neurons[-1]] + scale_neurons
        
        self.stack = nn.ModuleList([nn.Linear(in_features=neurons[i], out_features=neurons[i+1]) for i in range(len(neurons)-1)])
        self.mean_stack = nn.ModuleList([nn.Linear(in_features=mean_neurons[i], out_features=mean_neurons[i+1]) for i in range(len(mean_neurons)-1)])
        self.scale_stack = nn.ModuleList([nn.Linear(in_features=scale_neurons[i], out_features=scale_neurons[i+1]) for i in range(len(mean_neurons)-1)])
        
    def forward(self, inputs):
        x = inputs
        
        for num, layer in enumerate(self.stack):
            x = layer(x)
            if num < (len(self.stack)-1):
                x = self.activation(x)
        
        mean = torch.clone(x)
        for num, layer in enumerate(self.mean_stack):
            mean = layer(mean)
            if num < (len(self.mean_stack)-1):
                mean = self.activation(mean)
        
        scale = torch.clone(x)
        for num, layer in enumerate(self.scale_stack):
            scale = layer(scale)
            if num < (len(self.scale_stack)-1):
                scale = self.activation(scale)
                
        return mean, torch.exp(scale)+1e-6

#current_policy = PolicyModel(neurons=[4, 64, 64, 1], activation=nn.functional.relu)
current_policy = PolicyModelStd(neurons=[4, 64], scale_neurons=[64, 1], mean_neurons=[64, 1], activation=nn.functional.relu)
current_Vn = VnModel(neurons=[4, 64, 64, 1], activation=nn.functional.relu)
    
#current_policy = LearningModel(neurons=[4, 64, 64, 2], activation=nn.functional.relu)
#current_Vn = VnModel(neurons=[4, 64, 64, 1], activation=nn.functional.relu)
#https://deepboltzer.codes/policy-types-in-reinforcement-learning