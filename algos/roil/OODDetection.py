

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest

class OODdetector(object):
    def __init__(self, model_type, seed=19260817):
        
        self.model_type = model_type
        if model_type == "IF":
            self.model = IsolationForest(random_state=seed)
        elif model_type == "KNN":
            self.model  = NearestNeighbors(n_neighbors=1)

            self.det_value = 0.1

            print("KNN OOD detector: ", "l2-distance > ", self.det_value)
        else:
            raise Exception("Not Implemented")
    def fit(self, X):

        self.model = self.model.fit(X)
    
    def predict(self, X, return_distance=False):

        # ind: 1, ood: -1

        if self.model_type == "IF":
            pred = self.model.predict(X)
        elif self.model_type == "KNN":
            dist, _ = self.model.kneighbors(X, n_neighbors=1, return_distance=True)

            dist = dist.mean(axis=1)
            # print(dist)

            pred = np.ones(dist.shape[0])
            pred[dist > self.det_value] = -1

            if return_distance:
                return pred, dist

        return pred





        
