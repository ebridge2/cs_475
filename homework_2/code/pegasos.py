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
        # lambda is basically a rate, and we don't use rate here,
        # so initialize the same
        super(Pegasos, self).__init__(lambd, iterations)
        pass

    def dot_weight(self, example, weight):
        # load the example and compute the dot product
        return np.dot(weight, example)

    def train(self, instances):
        # initialize t
        t = 1
        for iteration in range(0, self.iterations):
            for instance in instances:
                example = self.load_example(instance)
                self.check_dims(example)
                # get the adjusted label (actual is binarized)
                y = self.l2s_dict[instance.get_label()]
                # if we are less than one, then this has a val of 1.0 (float)
                indic = float((y*self.dot_weight(example, self.w)) < 1)
                # given this formula
                self.w = (1 - 1/float(t))*self.w + 1/float((self.rate * t))*indic*y*example
                t = t + 1
        pass

    def predict(self, instance):
        example = self.load_example(instance)
        self.check_dims(example)
        return self.s2l_dict[self.predict_sgn(example, self.w)]

    def predict_sgn(self, example, weight):
        # returns 1 for the value being greater than zero, -1 for less
        value = self.dot_weight(example, weight)
        if (value >= 0):
            return 1
        else:
            return -1
