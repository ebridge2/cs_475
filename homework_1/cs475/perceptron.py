# perceptron.py
#
# written by Eric Bridgeford
#
# a class for abstracting a perceptron to
# either a standard perceptron or a
# averaged perceptron model.

from cs475_types import Predictor
from abc import ABCMeta, abstractmethod
import numpy as np


class Perceptron_Base(Predictor):
    __metaclass__ = ABCMeta

    def __init__(self, rate, iterations):
        super(Perceptron_Base, self).__init__()
        self.nfeatures = 1
        self.rate = rate
        self.iterations = iterations
        self.w = np.zeros(self.nfeatures)
        # define label to signed
        self.l2s_dict = {'0': -1, '1': 1}
        # define the signed to label
        self.s2l_dict = {-1: '0', 1: '1', 0: '1'}
        pass

    @abstractmethod
    def train(self, instances): pass

    def predict(self, instance):
        example = self.load_example(instance)
        self.check_dims(example)
        return self.s2l_dict[self.predict_sgn(example, self.w)]

    def predict_sgn(self, example, weight):
        # load the example and compute the sign of the sum of
        # the dot product
        value = np.sign(np.dot(weight, example))
        if (value >= 0):
            return 1
        else:
            return -1

    def load_example(self, instance):
        example = np.zeros(max(instance.max_feature(), self.nfeatures))
        for idx in instance:
            # since we have 1 indexed features and 0-indexed arrays
            example[idx - 1] = instance.get(idx)
        return example

    def check_dims(self, example, weight=None):
        inst_feat = example.shape[0]
        if (inst_feat > self.nfeatures):
            self.w = np.pad(self.w, (0, inst_feat - self.nfeatures),
                'constant', constant_values=0)
            self.nfeatures = inst_feat
            if (weight is not None):
                weight = np.pad(weight, (0, inst_feat - weight.shape[0]),
                            'constant', constant_values=0)
        if (weight is not None):
            return weight
        else:
            return self.w


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
