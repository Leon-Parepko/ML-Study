import torch
import torch.nn as nn
import SNN.Layers as Layers
from SNN.Optimizers import STDP


class MNIST_SNN(nn.Module):

    def __init__(self, input_size, output_size):
        super(MNIST_SNN, self).__init__()
        self.encoder = Layers.TemporalEncoder(input_size, 15, tau=1),

        self.encoder = self.encoder[0]
        self.LIF_1 = Layers.LIF(15, 10, tau=1)
        self.LIF_2 = Layers.LIF(10, output_size, tau=1)

    def forward(self, X, t):
        out = self.encoder.forward(X, t)
        out = self.LIF_1.forward(out, t)
        out = self.LIF_2.forward(out, t)
        return out

    # Returns tuple of fire/not-fire for each layer of neurons
    def neuron_states(self):
        return self.encoder.neuron_states, self.LIF_1.neuron_states, self.LIF_2.neuron_states







test_individ = torch.rand(5,5)
history = []

mnist_snn = MNIST_SNN(25, 10)
optimizer = STDP(mnist_snn.parameters(), mnist_snn.neuron_states())

with torch.no_grad():
    for j in range(0, 254):
            Y_pred = mnist_snn.forward(test_individ.view(-1, 5*5), j)
            history.append(Y_pred.clone())
            optimizer.step()
            pass


            # print(Y_pred)



    #         # Stack the spikes
    #         if j == 0:
    #             spikes_stack = Y_pred
    #         else:
    #             spikes_stack = torch.cat((spikes_stack, Y_pred), 0)
    #
    #
    # print(spikes_stack.shape)