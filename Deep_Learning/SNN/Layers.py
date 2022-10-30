import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class LIF(nn.Module):
    def __init__(self, in_features, out_features, tau=1, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LIF, self).__init__()
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.weight.data.uniform_(0, 1)
        self.in_features = in_features
        self.out_features = out_features
        self.tau = tau
        self.v = torch.zeros(in_features, **factory_kwargs)     # Initial membrane potential -70mv ~ 0
        self.threshold = 20        # Threshold value -50mv ~ 20
        self.spikes = torch.zeros(in_features, **factory_kwargs)


    def forward(self, x, t):
        # Weighted sum of x for each neuron
        sum = torch.sum(x, 1)

        # Leak
        self.v *= math.exp(-0.01 / self.tau)       # Analytical solution to the differential equation dv/dt = -1/tau * (v - v_0)
        # Integrate
        self.v += sum * 10      # maximum 10mv per spike
        # Fire
        for i in range(0, self.v.shape[0]):     # Check if any neuron has fired
            if self.v[i] >= self.threshold:
                self.spikes[i] = 1
                self.v[i] = 0
        return self.spikes * self.weight


    def neuron_states(self):
        return self.spikes


    def weight_reset(self):
        with torch.no_grad():
            self.weight.zero_()


    def weight_to_one(self):
        with torch.no_grad():
            for i in range(0, self.weight.shape[0]):
                for j in range(0, self.weight.shape[1]):
                    self.weight[i][j] = 1



class TemporalEncoder(nn.Module):
    def __init__(self, in_features, out_features, tau=1, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TemporalEncoder, self).__init__()
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.weight.data.uniform_(0, 1)
        self.in_features = in_features
        self.out_features = out_features
        self.tau = tau
        self.spikes = torch.zeros(in_features, **factory_kwargs)


    def forward(self, x, t):
        # Initialize the output
        self.spikes = torch.zeros(self.out_features, x.shape[1])

        # Normalize input and set it from 0 to 255
        x = (x / torch.max(x)) * 255

        # Discretize input using tau
        x = torch.floor(x / self.tau)

        for i in range(x.shape[1]):
            if torch.tensor(t, dtype=torch.int32) == torch.tensor(x, dtype=torch.int32)[0][i]:
                self.spikes[0][i] = 1

        return self.weight * self.spikes


    def neuron_states(self):
        return self.spikes


    def weight_reset(self):
        with torch.no_grad():
            self.weight.zero_()


    def weight_to_one(self):
        with torch.no_grad():
            for i in range(0, self.weight.shape[0]):
                for j in range(0, self.weight.shape[1]):
                    self.weight[i][j] = 1