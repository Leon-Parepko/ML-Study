import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


# Simple class to save the spikes and their pre/post fire time intervals.
class SpikeHistory:
    def __init__(self, pre=None, cur=None, post=None):
        self.spike_history = {"pre": pre,
                              "cur": cur,
                              "post": post}

    def __getitem__(self, key):
        return self.spike_history[key]

    def __setitem__(self, key, value: torch.Tensor):
        self.spike_history[key] = value



class LIF(nn.Module):
    def __init__(self, in_features, out_features, tau=1, device=None, dtype=None, final=False):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LIF, self).__init__()
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.weight.data.uniform_(0, 1)
        self.in_features = in_features
        self.out_features = out_features
        self.tau = tau
        self.v = torch.zeros(in_features, **factory_kwargs)  # Initial membrane potential -70mv ~ 0
        self.threshold = 20  # Threshold value -50mv ~ 20

        #  Initialize post spike history if it is not final layer
        post = None
        if not final:
            post = torch.zeros(out_features, **factory_kwargs)
        self.spike_history = SpikeHistory(post=post)


    def forward(self, x, t):
        # init spikes as zeros
        spikes = torch.zeros(self.in_features)

        # init pre spike history as zeros if it is none
        if self.spike_history["pre"] is None:
            self.spike_history["pre"] = torch.zeros(x.shape[1])

        # Update pre history if any pre spikes
        for i in range(x.shape[1]):
            if x[0][i] > 0:
                self.spike_history["pre"][i] = 1

            elif self.spike_history["pre"][i] >= 1:
                self.spike_history["pre"][i] += 1

        # Weighted normalized sum of x for each neuron
        sum = torch.sum(x, 1)

        # Leak
        self.v *= math.exp(
            -0.01 / self.tau)  # Analytical solution to the differential equation dv/dt = -1/tau * (v - v_0)
        # Integrate
        self.v += sum * 10  # maximum 10mv per spike
        # Fire
        for i in range(0, self.v.shape[0]):
            if self.v[i] >= self.threshold:  # Check if any neuron has fired
                spikes[i] = 1
                self.v[i] = 0

        # Update spike history
        self.spike_history["cur"] = spikes

        # Normalize output
        return (spikes * self.weight) / self.in_features


    def history_update(self, post_spikes=None):
        if post_spikes is not None:
            for i, post_spike in enumerate(post_spikes):
                # If post spike, reset time for post spike in history
                if post_spike > 0:
                    self.spike_history["post"][i] = 1

                # Else increase time for post spike in history
                elif self.spike_history["post"][i] >= 1:
                    self.spike_history["post"][i] += 1
        return self.spike_history["cur"]



    def get_membrane_potential(self):
        return self.v

    def get_spike_history(self):
        return self.spike_history

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
        self.spike_history = SpikeHistory(post=torch.zeros(out_features, **factory_kwargs))

    def forward(self, x, t):
        # Initialize the output
        spikes = torch.zeros(self.in_features)

        # Normalize input and set it from 0 to 255
        x = (x / torch.max(x)) * 255

        # Discretize input using tau
        x = torch.floor(x / self.tau)

        for i in range(x.shape[1]):
            if torch.tensor(t, dtype=torch.int32) == torch.tensor(x, dtype=torch.int32)[0][i]:
                spikes[i] = 1

        # Update spike history
        self.spike_history["cur"] = spikes
        return self.weight * spikes


    def history_update(self, post_spikes):
        for i, post_spike in enumerate(post_spikes):
            # If post spike, reset time for post spike in history
            if post_spike > 0:
                self.spike_history["post"][i] = 1

            # Else increase time for post spike in history
            elif self.spike_history["post"][i] >= 1:
                self.spike_history["post"][i] += 1


    def neuron_states(self):
        return self.spikes

    def get_spike_history(self):
        return self.spike_history

    def weight_reset(self):
        with torch.no_grad():
            self.weight.zero_()

    def weight_to_one(self):
        with torch.no_grad():
            for i in range(0, self.weight.shape[0]):
                for j in range(0, self.weight.shape[1]):
                    self.weight[i][j] = 1


class RateEncoder(nn.Module):
    def __init__(self, in_features, out_features, tau=1, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(RateEncoder, self).__init__()
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.weight.data.uniform_(0, 1)
        self.in_features = in_features
        self.out_features = out_features
        self.tau = tau
        self.spike_history = SpikeHistory(post=torch.zeros(out_features, **factory_kwargs))

    def forward(self, x, t):
        # Initialize the output
        spikes = torch.zeros(self.in_features)

        # Normalize input and set it from 0 to 255
        x = (x / torch.max(x)) * 255

        # Discretize input using tau
        x = torch.floor(x / self.tau)

        for i in range(x.shape[1]):
            if torch.tensor(t, dtype=torch.int32) == torch.tensor(x, dtype=torch.int32)[0][i]:
                spikes[i] = 1

        # Update spike history
        self.spike_history["cur"] = spikes
        return self.weight * spikes


    def history_update(self, post_spikes):
        for i, post_spike in enumerate(post_spikes):
            # If post spike, reset time for post spike in history
            if post_spike > 0:
                self.spike_history["post"][i] = 1

            # Else increase time for post spike in history
            elif self.spike_history["post"][i] >= 1:
                self.spike_history["post"][i] += 1


    def neuron_states(self):
        return self.spikes

    def get_spike_history(self):
        return self.spike_history

    def weight_reset(self):
        with torch.no_grad():
            self.weight.zero_()

    def weight_to_one(self):
        with torch.no_grad():
            for i in range(0, self.weight.shape[0]):
                for j in range(0, self.weight.shape[1]):
                    self.weight[i][j] = 1
