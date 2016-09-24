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
                example = np.zeros((nfeatures, 1))
                for idx in instance:
                    example[idx] = instance.get(idx)
                yhat = self.predict_sgn(instance)
                y = self.instance_label_key[instance.label]
                if (yhat not y):
                    self.w = self.w + self.rate * y * example
        pass

    @classmethod
    def predict(self, instance):
        return self.p_label_key(predict_label(instance))

    def predict_sgn(self, instance):
        example = self.load_example(instance)
        return np.sign((np.dot(self.w.transpose(), example)))

    def load_example(self, instance):
        example = np.zeros((nfeatures, 1))
        for idx in instance:
            example[idx] = instance.get(idx)
        return example
