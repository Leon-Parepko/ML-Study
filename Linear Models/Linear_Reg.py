import numpy as np
from tqdm import tqdm


def init_weights(X):
    random_weights = np.random.randn(np.shape(X)[1] + 1)
    return random_weights


def add_ones_col(X):
    return np.c_[np.ones(np.shape(X)[0]), X]

class LinearRegression:

    def __init__(self, loss="MSE"):
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
        if np.ndim(X_input) > 1:
            X_input = add_ones_col(X_input)
            for individ in X_input:
                result = individ @ self.weights
                out = np.append(out, result)
        else:
            X_input = np.append(1, X_input)
            out = X_input @ self.weights

        return out


    def train(self, lr=0.1, iter=100):
        X = add_ones_col(self.X_Data)
        features_num = np.shape(X)[1]
        for _ in tqdm(range(0, iter)):
            for i in range(0, np.shape(X)[0]):
                d_loss = (2 / features_num) * (self.predict(self.X_Data[i, :]) - self.Y_Data[i]) * X[i, :]
                self.weights = self.weights - lr * d_loss


if __name__ == '__main__':

    X = np.random.rand(2000, 50)
    Y = np.random.rand(2000, 1)

    L_R = LinearRegression()
    L_R.fit(X, Y)
    L_R.train(lr=0.01, iter=100)
    print(L_R.weights)