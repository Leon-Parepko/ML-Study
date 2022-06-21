import numpy as np
from math import sqrt



class knn ():

    def __init__(self, n_neighbors = 1):
        self.n_neighbors = n_neighbors
        self.loss = 0
        self.X_Data = []
        self.Y_Data = []



    def fit (self, X_Data, Y_Data):
            self.X_Data = X_Data
            self.Y_Data = Y_Data


    def predict (self, X_Test):
        for in :
            p1 =
            p2 =
            sum = 0
            for

                sum = (coord_p2 - Coord_p1) ^ 2
            distance = sqrt(sum)
