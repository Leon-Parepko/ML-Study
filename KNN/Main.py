import numpy as np
from math import sqrt
from tqdm import tqdm



class Knn:

    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors
        self.loss = 0
        self.X_Data = []
        self.Y_Data = []


    def fit (self, X_Data, Y_Data):
            self.X_Data = X_Data
            self.Y_Data = Y_Data


    def predict (self, X_Test):

        out_arr = np.array([])
        for X_Test_elem in tqdm(X_Test):

            distance_arr = np.array([])
            for X_elem in self.X_Data:
                p1 = X_elem
                p2 = X_Test_elem
                sub_coords = np.square(p1 - p2)
                distance = sqrt(np.sum(sub_coords))
                distance_arr = np.append(distance_arr, distance)

            distance_arr_ind = distance_arr.argsort()
            Y_sorted = self.Y_Data[distance_arr_ind]
            Y_sorted = Y_sorted[:self.n_neighbors]
            out = np.sum(Y_sorted) / Y_sorted.shape[0]
            out_arr = np.append(out_arr, out)

        return out_arr



X = np.random.rand(3, 4)
Y = np.random.rand(3)

test = np.random.rand(3, 4)

model = Knn(n_neighbors=1)
model.fit(X, Y)
model.predict(test)


