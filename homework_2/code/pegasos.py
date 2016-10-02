# pegasos.py
#
# written by Eric Bridgeford
#
# a class for the pegasos algorithm

from cs475_types import Predictor
from abc import ABCMeta, abstractmethod
import numpy as np


class Pegasos(Predictor):
    __metaclass__ = ABCMeta

    def __init__(self, lambd, iterations):
        super(Pegasos, self).__init__(lambd, iterations)
        pass

    def dot_weight(self, example, weight):
        # load the example and compute the sign of the sum of
        # the dot product
        return np.dot(weight, example)

    def train(self, instances):
        t = 1
        for iteration in range(0, self.iterations):
            for instance in instances:
                example = self.load_example(instance)
                self.check_dims(example)
                y = self.l2s_dict[instance.get_label()]
                indic = float((y*self.dot_weight(example, self.w)) < 1)
                self.w = (1 - 1/float(t))*self.w + 1/float((self.rate * t))*indic*y*example
                t = t + 1
        pass

    def predict(self, instance):
        example = self.load_example(instance)
        self.check_dims(example)
        return self.s2l_dict[self.predict_sgn(example, self.w)]

    def predict_sgn(self, example, weight):
        value = self.dot_weight(example, weight)
        if (value >= 0):
            return 1
        else:
            return -1
