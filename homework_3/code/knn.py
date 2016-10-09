# knn.py
# written by Eric Bridgeford
#
# a class for the k-nearest neighbor algorithm
#

from cs475_types import Predictor
from abc import ABCMeta, abstractmethod
import numpy as np

class KNN(Predictor):
    __metaclass__ = ABCMeta

    def __init__(self, knn):
        self.knn = knn
        pass
