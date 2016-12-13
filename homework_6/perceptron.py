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

class MC_Perceptron():

    def __init__(self, iterations):
        self.nfeatures = 0
        self.iterations = iterations
        # labels to class number
        self.l2cdict = {}
        # class number to labels
        self.c2ldict = {}
        self.nclasses = 0
        pass

    def load_example(self, instance): 
        example = np.zeros((self.nfeatures)) 
        for idx in instance: 
            # since we have 1 indexed features but zero indexed arrays 
            example[idx - 1] = instance.get(idx) 
        return example[:self.nfeatures]
 
    def get_feature(self, instance, feature): 
        return instance.get(feature + 1, default=0) 

    # figure out number of classes we need to track, and number of features 
    # then, initialize the weight matrix
    def initialize_attrs(self, instances):
        # go thru once just to get maximum feature information,
        # and class information at same time
        for instance in instances: 
            self.nfeatures = max(instance.max_feature(), self.nfeatures) 
            l = instance.get_label()
            # if we haven't seen this class yet, increment our class counter
            # accordingly
            if l not in self.l2cdict.keys():
                self.l2cdict[l] = int(l) - 1
                self.c2ldict[int(l) - 1] = l
                self.nclasses += 1
        self.ninstances = len(instances)
        # initialize weight matrix as nclasses x nfeatures
        # where each row is just the weight vector for each class
        self.weight = np.zeros((self.nclasses, self.nfeatures))
        pass

    def train(self, instances):
        self.initialize_attrs(instances)
        weight = self.weight
        for i in range(0, self.iterations):
            for instance in instances:
                x = self.load_example(instance)
                # numpy 2 da rescue with dat dere matrix multiplication
                predictions = weight.dot(x)
                yhat = np.argmax(predictions)
                y = self.l2cdict[instance.get_label()]
                # when we are wrong, make an update
                if yhat != y:
                    # subtract x from the predicted
                    weight[yhat, :] -= x
                    # add x to the truth
                    weight[y, :] += x
        self.weight = weight
        pass

    def predict(self, instance):
        x = self.load_example(instance)
        # TGFN (Thank God for Numpy)
        predictions = self.weight.dot(x)
        yhat  = np.argmax(predictions)
        return self.c2ldict[yhat]
