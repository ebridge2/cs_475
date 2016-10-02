from abc import ABCMeta, abstractmethod
import numpy as np

# abstract base class for defining labels
class Label:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __str__(self): pass

       
class ClassificationLabel(Label):
    def __init__(self, label):
        self.label = label
        pass
        
    def __str__(self):
        return str(self.label)

class FeatureVector:
    def __init__(self):
      	self.feature_dict = {}
        
    def add(self, index, value):
        self.feature_dict[index] = value
        pass
        
    def get(self, index):
        return self.feature_dict[index]

    def __iter__(self):
        return iter(self.feature_dict)

    def max_feature(self):
        return max(self.feature_dict.keys(), key=int)

class Instance:
    def __init__(self, feature_vector, label):
        self._feature_vector = feature_vector
        self._label = label

    def __iter__(self):
        return iter(self._feature_vector)

    def get_label(self):
        return str(self._label)

    def get(self, idx):
        return self._feature_vector.get(idx)

    def max_feature(self):
        return self._feature_vector.max_feature()

# abstract base class for defining predictors
class Predictor:
    __metaclass__ = ABCMeta

    def __init__(self, rate, iterations):
        # define label to signed
        self.l2s_dict = {'0': -1, '1': 1}
        # define the signed to label
        self.s2l_dict = {-1: '0', 1: '1', 0: '1'}
        self.nfeatures = 1
        self.rate = rate
        self.iterations = iterations
        self.w = np.zeros(self.nfeatures)
        pass

    @abstractmethod
    def train(self, instances): pass

    @abstractmethod
    def predict(self, instance): pass 

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
