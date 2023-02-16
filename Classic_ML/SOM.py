import numpy as np



def init_weights(features_num, map_size):
    """
        Initialize random weights vector
    """
    random_weights = np.random.randn(features_num, map_size[0], map_size[1])
    return random_weights


def add_ones_col(X):
    """
        Function to add colum of ones to the matrix
        (bias term)
    """
    return np.c_[np.ones(np.shape(X)[0]), X]


def get_indexes_in_radius(array, center, radius):
    """ """
    a = np.indices(array.shape).reshape(2, -1).T
    return a[np.abs(a-np.array(center)).sum(1) <= int(radius)]


def get_win_neuron(weights, X, map_size):
    """ """
    X_for_W_update = np.tile(X, (map_size[0], map_size[1], 1)).T        # Stack X into tensor to update weights

    win_tensor = weights - X_for_W_update
    win_matrix = np.linalg.norm(win_tensor, axis=0)                     # Calculate length of vectors (W - X)

    winner = np.argmin(win_matrix)                                      # Get minimum value index
    winner_coord = (winner // map_size[0], winner % map_size[1])
    return winner_coord



class SOM:
    def __init__(self, features_num, map_size):
        self.map_size = map_size
        self.features_num = features_num
        self.weights = []
        self.map = np.zeros(self.map_size)
        self.weights = init_weights(features_num, self.map_size)        # Random weights
        self.win_matrix = np.zeros(self.map_size)                       # Matrix of distances between weights and input vector
        self.weights_history = []
        self.map_history = []


    def predict(self, X):
        """ """
        winner_coord = get_win_neuron(self.weights, X, self.map_size)
        return self.map[winner_coord[0]][winner_coord[1]]


    def create_map(self, X_all, Y_all):
        """ """
        label_map = np.full(self.map_size, 0, dtype=object)             # Create empty map

        # Fill label_map with empty lists
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                label_map[i][j] = []

        for i in range(X_all.shape[0]):
            X = X_all[i]
            Y = Y_all[i]
            winner_coord = get_win_neuron(self.weights, X, self.map_size)
            label_map[winner_coord[0]][winner_coord[1]].append(Y)

        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                arr = label_map[i][j]

                if len(arr) == 0:
                    self.map[i][j] = -1
                else:
                    self.map[i][j] = max(arr, key=arr.count)


    def train(self, data, iter, lr=0.1, radius=1, weight_history=False, map_history=False, Y = None):
        """ """

        self.weights_history = []
        self.map_history = []
        for i in range(iter):
            X = data[np.random.randint(0, data.shape[0])]                                                                       # Get random X vector from data

            winner_coord = get_win_neuron(self.weights, X, self.map_size)

            for neuron_coord in get_indexes_in_radius(self.map, winner_coord, radius):                                          # Update weights of all the neurons in radius
                distance_to_winner = np.abs(winner_coord[0] - neuron_coord[0]) + np.abs(winner_coord[1] - neuron_coord[1])      # Topological distance (Manhattan distance)
                distance_to_winner = np.power(0.1, distance_to_winner)

                W_vec = self.weights[:, neuron_coord[0], neuron_coord[1]]                                                       # Get weights vector of neuron
                self.weights[:, neuron_coord[0], neuron_coord[1]] += distance_to_winner * lr * (X - W_vec)           # (1 - lr * distance_to_winner) * W_vec + lr * distance_to_winner * X    # Update weights of a neuron

            if weight_history and i % 300 == 0:
                self.weights_history.append(np.copy(self.weights))

            if map_history and i % 300 == 0:
                self.create_map(data, Y)
                self.map_history.append(np.copy(self.map))