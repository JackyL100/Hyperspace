import torch
import torch.nn as nn
import numpy as np

class DDPG_ACTOR(nn.Module):
    def __init__(self,
                 input,
                 output,
                 hidden_layers=(32,64),
                 activation_fn=nn.Tanh):
        super().__init__()
        #self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.device = torch.device("cpu")
        self.activation_fn = activation_fn
        self.model = nn.Sequential(
            nn.Linear(in_features=input, out_features=hidden_layers[0]),
            self.activation_fn(),
            nn.Linear(in_features=hidden_layers[0], out_features=hidden_layers[1]),
            self.activation_fn(),
            nn.Linear(in_features=hidden_layers[1], out_features=output),
            self.activation_fn()
        )
        self.to(self.device)
    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state,device=self.device,dtype=torch.float32)
            state.unsqueeze(0)
        return self.model(state)

class DDPG_CRITIC(nn.Module):
    def __init__(self,
                 input,
                 output,
                 hidden_layers=(32,64),
                 activation_fn=nn.ReLU):
        super().__init__()
        #self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.device = torch.device("cpu")
        self.activation_fn = activation_fn
        self.model = nn.Sequential(
            nn.Linear(in_features=input, out_features=hidden_layers[0]),
            self.activation_fn(),
            nn.Linear(in_features=hidden_layers[0], out_features=hidden_layers[1]),
            self.activation_fn(),
            nn.Linear(in_features=hidden_layers[1], out_features=output)
        )
        self.to(self.device)
    def forward(self, state, action):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=self.device, dtype=torch.float32)
            state.unsqueeze(0)
        if not isinstance(action, torch.Tensor): 
            action = torch.tensor(action, device=self.device, dtype=torch.float32)
            action.unsqueeze(0)
        state = torch.cat((state, action), dim=1)
        return self.model(state)