# means.py
# written by Eric Bridgeford
#
# a class for the lambda-means algorithm.
#

from cs475_types import Predictor
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.spatial import distance
from scipy.stats import mode


class Lambda_Means(Predictor):
    def __init__(self, lambd, iterations):
        super(Lambda_Means, self).__init__()
        self.lambd = lambd
        self.iterations = iterations
        self.K = 1
        self.nfeatures = 0
        pass

    # load the example into an array
    def load_example(self, instance):
        example = np.zeros((self.nfeatures))
        for idx in instance:
            # since we have 1 indexed features but zero indexed arrays
            example[idx-1] = instance.get(idx)
        return example[:self.nfeatures]

    def get_feature(self, instance, feature):
        return instance.get(feature + 1, default=0)

    def calculate_max_feature(self):
        for instance in self.instances:
            self.nfeatures = max(instance.max_feature(), self.nfeatures)
        self.ninstances = len(self.instances)
        pass

    # get the mean of a particular feature based on a responsibility
    # vector.
    def get_feature_mean(self, resp, feat_id):
        this_feature = np.ones((self.ninstances))
        instances = self.instances
        for idx in range(0, self.ninstances):
            this_feature[idx] = self.get_feature(instances[idx], feat_id)
        return np.mean(this_feature)

    def initialize(self):
        # will be the mean of all of these
        resp = np.ones((self.ninstances, self.K))         
        means = np.zeros((self.nfeatures, 1))
        for feature in range(0, self.nfeatures):
            means[feature, 0] = self.get_feature_mean(resp, feature)
        self.resp = resp
        self.means = means
        pass

    # a function to compute the distance between a single datum and
    # all of the clusters we have
    def compute_distance(self, x, means):
        K = self.K
        distances = np.zeros((K))
        for k in range(0, K):
            distances[k] = distance.euclidean(x, means[:,k])**2
        return distances

    def get_responsibilities(self, lambd):
        r = np.zeros((self.ninstances, self.K))
        # load the means locally for speed
        means = self.means
        for idx in range(0, self.ninstances):
            print "resp" + str(idx)
            x = self.load_example(self.instances[idx])
            distance = self.compute_distance(x, means)
            distance = distance[distance <= lambd]
            if distance.shape[0] != 0:
                # compute the minimum distance, and use
                # only the minimum idx in case of ties
                clus_id = np.min(np.argmin(distance))
                # set the responsibility to 1, will be 0 otherwise
                r[idx, clus_id] = 1
            # make a new cluster if none of the current clusters
            # satisfy the lambda requirement
            else:
                means = np.column_stack((means, x))
                # add a responsibility vector that defaults to zeros
                # and set the value of this instance to 1
                new_rvec = np.zeros((self.ninstances, 1))
                new_rvec[idx] = 1
                r = np.column_stack((r, new_rvec))
                self.K = self.K + 1
        self.means = means
        self.resp = r
        pass

    def update_means(self):
        # use these locally for speed bump
        K = self.K
        r = self.resp
        means = np.zeros((self.nfeatures, K))
        instances = self.instances
        for k in range(0, K):
            print "means" + str(k)
            r_vec = r[:,k]
            # get the instance ids we want so we don't have to
            # check all of them
            r_ids = np.where(r_vec != 0)[0]
            for idx in r_ids:
                means[:,k] += self.load_example(instances[idx])
            if (r_ids.shape[0] != 0):
                means[:,k] = means[:,k]/r_ids.shape[0]
            else:
                means[:,k] = means[:,k]
        # update with our maximized means
        self.means = means
        pass

    def train(self, instances):
        self.instances = instances
        self.calculate_max_feature()
        # initialize the parameters of the model (resps and means)
        self.initialize()
        for i in range(0, self.iterations):
            print i
            # get the respnsibilities, E step
            self.get_responsibilities(self.lambd)
            # maximize the means, M step
            self.update_means()
        pass

    # make predictions baed on the shortest distance cluster to
    # a particular example
    def predict(self, instance):
        x = self.load_example(instance)
        distance = self.compute_distance(x, self.means)
        # return minimum cluster id that has the minimum distance
        return np.min(np.argmin(distance))
