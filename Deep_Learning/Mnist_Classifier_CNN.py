
import torch.nn as nn


class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv_model = nn.Sequential(
            nn.Conv2d(1, 10, 3, stride=1),
            nn.ReLU(),

            nn.Conv2d(10, 30, 3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(30, 50, 3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # nn.Conv2d(60, 100, 1, stride=1),
            # nn.ReLU(),
            # nn.MaxPool2d(2),
            )

        self.lin_model = nn.Sequential(
            nn.Linear(50*5*5, 250),
            nn.ReLU(),

            nn.Linear(250, 10),
            )

    def forward(self, X):
        out = self.conv_model(X)
        out = out.view(-1, 50*5*5)
        out = self.lin_model(out)
        return out