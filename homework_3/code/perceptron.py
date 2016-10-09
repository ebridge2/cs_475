# perceptron.py
#
# written by Eric Bridgeford
#
# a class for abstracting a perceptron to
# either a standard perceptron or a
# averaged perceptron model.

from cs475_types import Predictor_Weight
from abc import ABCMeta, abstractmethod
import numpy as np


class Perceptron_Base(Predictor_Weight):
    __metaclass__ = ABCMeta

    def __init__(self, rate, iterations):
        super(Perceptron_Base, self).__init__(rate, iterations)
        pass

    @abstractmethod
    def train(self, instances): pass

    def predict(self, instance):
        example = self.load_example(instance)
        self.check_dims(example)
        return self.s2l_dict[self.predict_sgn(example, self.w)]

    def dot_weight(self, example, weight):
        # load the example and compute the sign of the sum of
        # the dot product
        return np.sign(np.dot(weight, example))

    def predict_sgn(self, example, weight):
        value = self.dot_weight(example, weight)
        if (value >= 0):
            return 1
        else:
            return -1

class Perceptron(Perceptron_Base):

    def __init__(self, rate, iterations):
        super(Perceptron, self).__init__(rate, iterations)
        pass

    def train(self, instances):
        for iteration in range(0, self.iterations):
            for instance in instances:
                example = self.load_example(instance)
                self.check_dims(example)
                yhat = self.predict_sgn(example, self.w)
                y = self.l2s_dict[instance.get_label()]
                if (yhat != y):
                    self.w = self.w + self.rate * y * example
        pass


class Weighted_Perceptron(Perceptron_Base):

    def __init__(self, rate, iterations):
        super(Weighted_Perceptron, self).__init__(rate, iterations)
        pass

    def train(self, instances):
        w_round = self.w.copy()
        for iteration in range(0, self.iterations):
            for instance in instances:
                example = self.load_example(instance)
                w_round = self.check_dims(example, weight=w_round)
                yhat = self.predict_sgn(example, w_round)
                y = self.l2s_dict[instance.get_label()]
                if (yhat != y):
                    w_round = w_round + self.rate * y * example
                self.w += w_round
        pass


class Margin_Perceptron(Perceptron_Base):

    def __init__(self, rate, iterations):
        super(Margin_Perceptron, self).__init__(rate, iterations)
        pass

    def train(self, instances):
        for iteration in range(0, self.iterations):
            for instance in instances:
                example = self.load_example(instance)
                self.check_dims(example)
                value = self.dot_weight(example, self.w)
                y = self.l2s_dict[instance.get_label()]
                if ((y*value) < 1):
                    self.w = self.w + self.rate * y * example
        pass
