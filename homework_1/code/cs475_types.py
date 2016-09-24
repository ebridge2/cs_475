from abc import ABCMeta, abstractmethod

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
        return(str(self.label))

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

# abstract base class for defining predictors
class Predictor:
    __metaclass__ = ABCMeta

    def __init__(self, nfeatures, rate, iterations):
        self.rate = rate
        self.iterations = iterations
        self.nfeatures = nfeatures
        self.w = np.zeros((nfeatures, 1))
        self.instance_label_key = {0: -1, 1: 1}
        self.p_label_key = {-1: 0, 1: 1}

    @abstractmethod
    def train(self, instances): pass

    @abstractmethod
    def predict(self, instance): pass       
