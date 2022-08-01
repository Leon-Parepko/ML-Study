
import torch.nn as nn


class MNIST_NN(nn.Module):
    def __init__(self, input_size, output_size):
        super(MNIST_NN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, output_size),
        )

    def forward(self, X):
        return self.model(X)