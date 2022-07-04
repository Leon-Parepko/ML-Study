import numpy as np
from tqdm import tqdm


''' 
    Initialize random weights vector
'''
def init_weights(X):
    random_weights = np.random.randn(np.shape(X)[1] + 1)
    return random_weights


'''
    Function to add colum of ones to the matrix
        (bias term)
'''
def add_ones_col(X):
    return np.c_[np.ones(np.shape(X)[0]), X]




class LinearRegression:
    def __init__(self):
        self.X_Data = []
        self.Y_Data = []
        self.weights = []


    def fit(self, X_Data, Y_Data):
        self.X_Data = X_Data
        self.Y_Data = Y_Data
        self.weights = init_weights(self.X_Data)    # Random weights


    def predict(self, X_input):
        out = np.array([])

        if np.ndim(X_input) > 1:
            X_input = add_ones_col(X_input)     # Add ones column if it is matrix
            for individ in X_input:
                result = individ @ self.weights     # Make prediction (X * W)
                out = np.append(out, result)
        else:
            X_input = np.append(1, X_input)     # Add single one if it is vector
            out = X_input @ self.weights

        return out


    def train(self, lr=0.1, iter=100, L1=False, L2=False):
        X = add_ones_col(self.X_Data)           # Add ones column if it is matrix
        features_num = np.shape(X)[1]

        for _ in tqdm(range(0, iter)):          # Start of gradient decent
            for i in range(0, np.shape(X)[0]):  # Iteratively get mean gradient of a sample

                d_lasso = 0
                d_ridge = 0

                if L1 != False:
                  d_lasso = L1 * (self.weights / np.absolute(self.weights))     # Lasso regularisation derivative

                if L2 != False:
                  d_ridge = 2 * L2 * self.weights       # Ridge regularisation derivative
                  d_ridge[0] = 0                        # Did not consider bias term

                # Calculate loss derivative with it's regularization
                d_loss = (2 / features_num) \
                         * (self.predict(self.X_Data[i, :]) - self.Y_Data[i])\
                         * X[i, :] + d_lasso + d_ridge

                self.weights = self.weights - lr * d_loss



if __name__ == '__main__':

    # Initialize test sample
    X = np.random.rand(2000, 50)
    Y = np.random.rand(2000, 1)

    # Initialize model
    L_R = LinearRegression()
    L_R.fit(X, Y)
    L_R.train(lr=0.01, iter=100, L1=0.2, L2=0.2)
    print(L_R.weights)