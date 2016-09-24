# weighted_perceptron.py
# Eric Bridgeford
#
# An ipmlementation of a weighted perceptron.

from cs475_types import Predictor
import numpy as np

def class Weighted_Perceptron(Predictor):
    def __init__(self, nfeatures, rate, iterations):
        super(Weighted_Perceptron, self).__init__(nfeatures, rate, iterations)
        pass

    @classmethod
    def train(self, instances):
        for iteration in self.iterations:
            for instance in instances:

        pass

    @classmethod
    def predict(self, instance):
        pass
    
