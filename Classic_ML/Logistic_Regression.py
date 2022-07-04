import numpy as np
from tqdm import tqdm
from sklearn import preprocessing


''' 
    Initialize random weights vector
'''
def init_weights(X, classes_num):
    random_weights = np.random.randn(np.shape(X)[1] + 1, classes_num)
    return random_weights


'''
    Function to add colum of ones to the matrix
        (bias term)
'''
def add_ones_col(X):
    return np.c_[np.ones(np.shape(X)[0]), X]


def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=0)



class LogisticRegression:
    def __init__(self):
        self.X_Data = []
        self.Y_Data = []
        self.weights = []


    def fit(self, X_Data, Y_Data, classes_num):
        self.X_Data = X_Data
        self.Y_Data = Y_Data
        self.weights = init_weights(self.X_Data, classes_num)    # Random weights


    def predict(self, X_input):
        if np.ndim(X_input) > 1:
            X_input = add_ones_col(X_input)     # Add ones column if it is matrix
        else:
            X_input = np.append(1, X_input)     # Add single one if it is vector

        out = softmax(- X_input @ self.weights)
        return out


    def train(self, lr=0.1, iter=100, L1=False, L2=False):

        X = add_ones_col(self.X_Data)           # Add ones column if it is matrix
        features_num = np.shape(X)[1]

        for _ in tqdm(range(0, iter)):          # Start of gradient decent
            d_lasso = 0
            d_ridge = 0

            if L1 != False:
              d_lasso = L1 * (self.weights / np.absolute(self.weights))     # Lasso regularisation derivative

            if L2 != False:
              d_ridge = 2 * L2 * self.weights       # Ridge regularisation derivative
              d_ridge[0] = 0


            prediction = self.predict(self.X_Data)
            d_loss = 1 / features_num * (X.T @ (prediction - self.Y_Data)) + L1 +L2

            self.weights = self.weights - lr * d_loss

            # for i in range(0, np.shape(X)[0]):  # Iteratively get mean gradient of a sample
            #
            #     d_lasso = 0
            #     d_ridge = 0
            #
            #     if L1 != False:
            #       d_lasso = L1 * (self.weights / np.absolute(self.weights))     # Lasso regularisation derivative
            #
            #     if L2 != False:
            #       d_ridge = 2 * L2 * self.weights       # Ridge regularisation derivative
            #       d_ridge[0] = 0                        # Did not consider bias term
            #
            #     prediction = self.predict(self.X_Data[i, :])
            #     # Calculate loss derivative with it's regularization (gradient)






if __name__ == '__main__':
    def test(x):
        zeros = np.zeros(10)
        zeros[x] = x
        return zeros


    # Initialize test sample
    X = np.random.rand(2000, 50)
    Y = np.random.randint(0, 10, (2000, 1))
    Y = np.array(list(map(lambda x: test(x), Y)))

    Y = preprocessing.minmax_scale(Y, axis=0)

    # Initialize model
    L_R = LogisticRegression()
    L_R.fit(X, Y, 10)
    L_R.train(lr=0.01, iter=100, L1=0.2, L2=0.2)

    print(L_R.weights)