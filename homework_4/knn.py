# knn.py
# written by Eric Bridgeford
#
# a class for the k-nearest neighbor algorithm
#

from cs475_types import Predictor
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.spatial import distance
from scipy.stats import mode


class KNN(Predictor):
    # an abstract class to save some space
    __metaclass__ = ABCMeta

    def __init__(self, knn):
        super(KNN, self).__init__()
        self.knn = knn
        self.nfeatures = 0 # default to 0 feature unless our data says otherwise
        pass

    def load_example(self, instance):
        example = np.zeros(max(self.nfeatures, instance.max_feature()))
        for idx in instance:
            # since we have 1 indexed features and 0-indexed arrays
            example[idx - 1] = instance.get(idx)
        return example[:self.nfeatures]
    
    def train(self, instances):
        self.instances = instances
        for instance in self.instances:
            self.nfeatures = max(instance.max_feature(), self.nfeatures)
        pass

    def distance(self, ex1, ex2):
        # we're allowed to use scipy, so might as well
        return distance.euclidean(ex1, ex2)

    @abstractmethod
    def predict(self, instance): pass

    def maxval(self, dict_class):
        list_min_class = []
        maxval = max(dict_class.items(), key=lambda x:x[1])[1]
        for cl_label in dict_class:
            if (dict_class[cl_label] == maxval):
                list_min_class.append(int(cl_label))

        return str(min(list_min_class))


class Standard_knn(KNN):
    __metaclass__ = ABCMeta

    def __init__(self, knn):
       super(Standard_knn, self).__init__(knn) 

    def predict(self, instance):
        dist = [] # one distance for each 
        dtype = [('idx', int), ('label', int), ('dist', float)]
        pred_ex = self.load_example(instance)
        # compare for each example
        for idx, example in enumerate(self.instances):
            cmp_ex = self.load_example(example)
            dist.append(tuple((idx, int(example.get_label()), self.distance(pred_ex, cmp_ex))))
        knn_idx = np.sort(np.array(dist, dtype=dtype), order=('dist', 'label'))['idx'][:self.knn] # get the knn
        knn_class = np.array([int(inst.get_label()) for inst in [self.instances[idx] for idx in knn_idx]])
        unique_class = { unique : 0 for unique in set(knn_class) }
        for idx, val in enumerate(knn_idx):
            unique_class[knn_class[idx]] = unique_class[knn_class[idx]] + 1
        return self.maxval(unique_class)

class Distance_knn(KNN):
    __metaclass__ = ABCMeta

    def __init__(self, knn):
        super(Distance_knn, self).__init__(knn)

    def predict(self, instance):
        dist = [] # one distance for each 
        dtype = [('idx', int), ('label', int), ('dist', float)]
        pred_ex = self.load_example(instance)
        for idx, example in enumerate(self.instances):
            cmp_ex = self.load_example(example)
            dist.append(tuple((idx, int(example.get_label()), self.distance(pred_ex, cmp_ex))))
        knn_idx = np.sort(np.array(dist, dtype=dtype), order=('dist', 'label'))[:self.knn] # get the knn
        knn_class = np.array([int(inst.get_label()) for inst in [self.instances[idx] for idx in knn_idx['idx']]])
        unique_class = { unique : 0 for unique in set(knn_class) }
        dist_class = knn_idx['dist']
        for idx, val in enumerate(knn_idx['idx']):
            unique_class[knn_class[idx]] = unique_class[knn_class[idx]] + 1/(1 + dist_class[idx]**2)
        return self.maxval(unique_class) # get the key with smallest value
