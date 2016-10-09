# knn.py
# written by Eric Bridgeford
#
# a class for the k-nearest neighbor algorithm
#

from cs475_types import Predictor
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.spatial import distance
from scipy.stats import mode


class KNN(Predictor):
    __metaclass__ = ABCMeta

    def __init__(self, knn):
        self.knn = knn
        self.nfeatures = 0 # default to 0 feature unless our data says otherwise
        pass

    def load_example(self, instance):
        example = np.zeros(max(instance.max_feature()))
        for idx in instance:
            # since we have 1 indexed features and 0-indexed arrays
            example[idx - 1] = instance.get(idx)
        return example[:self.nfeatures]
    
    def train(self, instances):
        self.instances = instances
        for instance in self.instances:
            self.nfeatures = max(instance.max_feature(), self.nfeatures)
        pass

    def distance(self, ex1, ex2):
        return distance.euclidean(ex1, ex2)

    @abstractmethod
    def predict(self, instance): pass

class Standard_knn(KNN):
    __metaclass__ = ABCMeta

    def __init__(self, knn):
        self.knn = knn
        self.nfeatures = nfeatures

    def predict(self, instance):
        dist = np.zeros(len(instances)) # one distance for each 
        pred_ex = self.load_example(instance)
        for idx, example in enumerate(self.instances):
            cmp_ex = self.load_example(example)
            dist[idx] = self.distance(pred_ex, cmp_ex)
        knn_idx = np.argsort(dist)[:self.knn] # get the knn
        knn_class = np.array([int(instance.get_label()) for instance in self.instances[idx] for idx in knn_idx])
        return str(mode(knn_class)[0][0])
