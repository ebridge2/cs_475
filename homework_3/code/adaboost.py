# adaboost.py
# written by Eric Bridgeford
#
# a class for the adaboost algorithm
#

from cs475_types import Predictor
from abc import ABCMeta, abstractmethod
import numpy as np


class Adaboost(Predictor):
    __metaclass__ = ABCMeta

    def __init__(self, iterations):
        super(Adaboost, self).__init__()
        self.iter_boost = iterations
        self.nfeatures = 0
        pass

    def learn_features(self, instances):
        for instance in instances:
            self.nfeatures = max(self.nfeatures, instance.max_feature())
        pass

    def train(self, instances):
        self.learn_features(instances)
        self.hypothesis_set = []
        for feature_id in range(0, self.nfeatures):
            values = np.array([instance.get(feature_id + 1, default=0) for instance in instances])
            sorted_vals = np.sort(np.unique(values))
            self.hypothesis_set.append([np.mean([sorted_vals[i], sorted_vals[i + 1]]) for i in range(0, len(sorted_vals) - 1)])

    def predict(self, instance):
        pass

