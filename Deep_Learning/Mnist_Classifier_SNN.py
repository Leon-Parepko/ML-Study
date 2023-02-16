import torch
import torch.nn as nn
import SNN.Layers as Layers
from SNN.Optimizers import STDP


class MNIST_SNN(nn.Module):

    # in -> 15 -> 10 -> out
    def __init__(self, input_size, output_size):
        super(MNIST_SNN, self).__init__()
        self.encoder = Layers.RateEncoder(input_size, 5, tau=1),

        self.encoder = self.encoder[0]
        self.LIF_1 = Layers.LIF(5, 5, tau=1)
        self.LIF_2 = Layers.LIF(5, output_size, tau=1, final=True)


        # self.encoder.weight_to_one()
        # self.LIF_1.weight_to_one()
        # self.LIF_2.weight_to_one()

    def forward(self, X, t):
        out = self.encoder.forward(X, t)
        out = self.LIF_1.forward(out, t)
        out = self.LIF_2.forward(out, t)

        s = self.LIF_2.history_update(post_spikes=None)
        s = self.LIF_1.history_update(post_spikes=s)
        s = self.encoder.history_update(post_spikes=s)

        return out

    def get_potentials(self):
        L1 = self.LIF_1.get_membrane_potential()
        L2 = self.LIF_2.get_membrane_potential()
        return L1, L2

    def spike_history(self):
        return self.encoder.get_spike_history, self.LIF_1.get_spike_history, self.LIF_2.get_spike_history




# ---------------
#
#
# test_individ = torch.rand(5,5)
# history = []
#
# mnist_snn = MNIST_SNN(25, 10)
# optimizer = STDP(mnist_snn.parameters(), mnist_snn.spike_history(), lr=0.001)
#
# with torch.no_grad():
#     for j in range(0, 254):
#             Y_pred = mnist_snn.forward(test_individ.view(-1, 5*5), j)
#             history.append(Y_pred.clone())
#             optimizer.step()
# ----------------


            # print(Y_pred)



    #         # Stack the spikes
    #         if j == 0:
    #             spikes_stack = Y_pred
    #         else:
    #             spikes_stack = torch.cat((spikes_stack, Y_pred), 0)
    #
    #
    # print(spikes_stack.shape)