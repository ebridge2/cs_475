# percepteron.py
# Eric Bridgeford
#
# An ipmlementation of a basic percepteron.

from cs475_types import Predictor

def class Perceptron(Predictor):
    def __init__(self, nfeatures, rate, iterations):
        super(Perceptron, self).__init__(nfeatures, rate, iterations)
        pass

    @classmethod
    def train(self, instances):
        for iteration in self.iterations:
            for instance in instances: # iterate over the training examples
                yhat = self.predict_sgn(instance) # predict the sign
                y = self.l2s_dict[instance.label]
                # if we guess wrong, update based on learning rate
                if (yhat not y):
                    self.w = self.w + self.rate * y * example
        pass

    @classmethod
    def predict(self, instance):
        # use the key for mapping signs to labels
        return self.s2l_dict(predict_label(instance))

    def predict_sgn(self, instance):
        # load the example and compute the sign of the product
        # of the transpose of w with itself
        example = self.load_example(instance)
        return np.sign((np.dot(self.w.transpose(), example)))

    def load_example(self, instance):
        # load a vector of zeros and fill in nonzero elements
        # from the feature vector dict (sparse)
        example = np.zeros((nfeatures, 1))
        for idx in instance:
            example[idx] = instance.get(idx)
        return example
