import numpy as np
import pandas as pd
from tqdm import tqdm


def init_weights(X):
    random_weights = np.random.rand(np.shape(X)[1] + 1)
    return random_weights

def mse():
    pass

def add_ones_col(X):
    return np.c_[np.ones(np.shape(X)[0]), X]

class LinearRegression:

    def __init__(self, loss="MSE"):
        # if loss == "MSE":
        self.loss = loss
        self.X_Data = []
        self.Y_Data = []
        self.weights = []


    def fit(self, X_Data, Y_Data):
        self.X_Data = X_Data
        self.Y_Data = Y_Data
        self.weights = init_weights(self.X_Data)


    def predict(self, X_input):
        out = np.array([])
        X_input = add_ones_col(X_input)
        if np.ndim(X_input) > 1:
            for individ in X_input:
                result = self.weights @ individ
                out = np.append(out, result)
        else:
            out = self.weights @ X_input

        return out


    def train(self, learning_rate=0.1):

        X = add_ones_col(self.X_Data)
        features_num = np.shape(X)[1]
        for i in tqdm(range(0, np.shape(X)[0])):
            d_loss = (1 / features_num) * np.transpose(X[i, :]) * (self.predict(self.X_Data[i, :]) - self.Y_Data[i])
            self.weights = self.weights - learning_rate * d_loss


if __name__ == '__main__':

    X = np.random.rand(21613, 1)
    Y = np.random.rand(21613, 1)

    L_R = LinearRegression()
    L_R.fit(X, Y)
    L_R.train()