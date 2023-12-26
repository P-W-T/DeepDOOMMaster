import numpy as np
import pandas as pd
import math
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
    

class PolicyModel(BaseDiscreteModel):
    def __init__(self, C, H, W, end):
        super().__init__()        
        
        self.C = C
        self.H = H
        self.W = W
        self.end = end
        
        self.neurons = math.floor(H/(2**5))*math.floor(W/(2**5))*128 + 1
        
        self.conv1 = nn.Conv2d(in_channels=C, out_channels=8, kernel_size=7, padding='same')
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=7, padding='same')
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, padding='same')
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, padding='same')
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7, padding='same')       
        
        self.ff1 = nn.Linear(in_features=self.neurons, out_features=128)
        self.ff2 = nn.Linear(in_features=128, out_features=64)
        self.ff3 = nn.Linear(in_features=64, out_features=end)
       
    
    def forward(self, observations): 
        screen, variables = self.unflatten_observation(observations)
        screen = torch.permute(screen, (0, 3, 1, 2))
        x = self.conv1(screen)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x,2)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x,2)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x,2)
        x = self.conv4(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x,2)
        x = self.conv5(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x,2)
        x = torch.cat((x.reshape((observations.shape[0], -1)), variables), dim=1)
        x = self.ff1(x)
        x = nn.functional.relu(x)
        x = self.ff2(x)
        x = nn.functional.relu(x)
        
        output = self.ff3(x)
        
        return output
    
    def unflatten_observation(self, observations): #works on tensors
        variables = observations[:,-1:]
        screen = observations[:,:-1].view(observations.shape[0], self.H, self.W, self.C)
        return screen, variables
        
    def flatten_observation(self, observations): #works on arrays (standard output)
        batch_dim = observations['gamevariables'].shape[0]
        variables = observations['gamevariables'].reshape(batch_dim, -1)
        screen = observations['screen'].reshape(batch_dim, -1)
        
        return np.concatenate((screen, variables), axis=-1)
    
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

current_policy = PolicyModel(C=3, H=240, W=320, end=4)
current_Vn = PolicyModel(C=3, H=240, W=320, end=1)
#https://deepboltzer.codes/policy-types-in-reinforcement-learning