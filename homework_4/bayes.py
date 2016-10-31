# bayes.py
# written by Eric Bridgeford
#
# a class for the naive-bayes algorithm.
#

from cs475_types import Predictor
from abc import ABCMeta, abstractmethod
import numpy as np

class Naive_Bayes(Predictor):
    def __init__(self, nclust, iterations):
        super(Naive_Bayes, self).__init__() 
        pass
