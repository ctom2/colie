import torch
import torch.nn as nn
import numpy as np


class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        if not self.is_last: 
            self.init_weights()
    
    def init_weights(self):
        b = 1 / self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        return nn.Sigmoid()(x) if self.is_last else torch.sin(self.w0 * x)


class INF(nn.Module):
    def __init__(self, patch_dim, num_layers, hidden_dim, add_layer, weight_decay=None):
        super().__init__()
        '''
        `add_layer` should be in range of  [1, num_layers-2]
        '''

        patch_layers = [SirenLayer(patch_dim, hidden_dim, is_first=True)]
        spatial_layers = [SirenLayer(2, hidden_dim, is_first=True)]
        output_layers = []
        
        for _ in range(1, add_layer - 2):
            patch_layers.append(SirenLayer(hidden_dim, hidden_dim))
            spatial_layers.append(SirenLayer(hidden_dim, hidden_dim))
        patch_layers.append(SirenLayer(hidden_dim, hidden_dim//2))
        spatial_layers.append(SirenLayer(hidden_dim, hidden_dim//2))
        
        for _ in range(add_layer, num_layers - 1):
            output_layers.append(SirenLayer(hidden_dim, hidden_dim))
        output_layers.append(SirenLayer(hidden_dim, 1, is_last=True))

        self.patch_net = nn.Sequential(*patch_layers)
        self.spatial_net = nn.Sequential(*spatial_layers)
        self.output_net = nn.Sequential(*output_layers)
        
        if not weight_decay:
            weight_decay = [0.1, 0.0001, 0.001]
            
        self.params = []
        self.params += [{'params':self.spatial_net.parameters(),'weight_decay':weight_decay[0]}]
        self.params += [{'params':self.patch_net.parameters(),'weight_decay':weight_decay[1]}]
        self.params += [{'params':self.output_net.parameters(),'weight_decay':weight_decay[2]}]

    def forward(self, patch, spatial):
        return self.output_net(torch.cat((self.patch_net(patch), self.spatial_net(spatial)), -1))