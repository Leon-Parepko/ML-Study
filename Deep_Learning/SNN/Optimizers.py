import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.optim import Optimizer


class STDP(Optimizer):

    def __init__(self, params, spikes, device=None, dtype=None):
        super(STDP, self).__init__(params, {})
        self.spikes = spikes

    def step(self, closure=None):


        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                weights = p.data
                spikes = self.spikes[i]()
            #   TODO: Check and calculate stdp for each weight using pre and post (i+1) spikes

