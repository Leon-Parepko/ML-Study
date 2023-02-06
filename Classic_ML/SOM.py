import numpy as np

''' 
    Initialize random weights vector
'''
def init_weights(features_num, map_size):
    random_weights = np.random.randn(features_num, map_size[0], map_size[1])
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


def get_indexes_in_radius(array, center, radius):
    a = np.indices(array.shape).reshape(2, -1).T
    return a[np.abs(a-np.array(center)).sum(1) <= int(radius)]


class SOM:
    def __init__(self, features_num, map_size):
        self.map_size = map_size
        self.features_num = features_num
        self.weights = []
        self.Y = np.zeros(self.map_size)
        self.weights = init_weights(features_num, self.map_size)    # Random weights
        self.win_matrix = np.zeros(self.map_size)                   # Matrix of distances between weights and input vector
        self.label_map = np.zeros(self.map_size)                    # Matrix of labels



    def train(self, X, lr=0.1, iter=100, radius=1):
        X_for_W_update = np.tile(X, (self.map_size[0], self.map_size[1], 1)).T        # Stack X into tensor to update weights

        for _ in range(0, iter):                                     # Start of algorithm
            win_tensor = self.weights - X_for_W_update
            self.win_matrix = np.linalg.norm(win_tensor, axis=0)        # Calculate length of vectors (W - X)

            winner = np.argmin(self.win_matrix)      # Get minimum value index
            winner_coord = (winner // self.map_size[0], winner % self.map_size[1])

            for i, neuron_coord in enumerate(get_indexes_in_radius(self.Y, winner_coord, radius)):       # Update weights of all the neurons in radius
                distance_to_winner = np.abs(winner_coord[0] - neuron_coord[0]) + np.abs(winner_coord[1] - neuron_coord[1])   # Topological distance (not Euclidean)
                distance_to_winner = np.power(0.1, distance_to_winner)

                W_vec = self.weights[:, neuron_coord[0], neuron_coord[1]]       # Get weights vector of neuron
                self.weights[:, neuron_coord[0], neuron_coord[1]] += distance_to_winner * lr * (X - W_vec)           # (1 - lr * distance_to_winner) * W_vec + lr * distance_to_winner * X    # Update weights of a neuron




if __name__ == '__main__':
    pass