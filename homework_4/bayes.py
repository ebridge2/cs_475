# bayes.py
# written by Eric Bridgeford
#
# a class for the naive-bayes algorithm.
#

from cs475_types import Predictor
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.spatial import distance

class Naive_Bayes(Predictor):
    def __init__(self, nclust, iterations):
        super(Naive_Bayes, self).__init__()
        self.iterations = iterations
        self.K = nclust
        self.nfeatures = 0
        pass

    # load the example into an array
    def load_example(self, instance):
        example = np.zeros((self.nfeatures))
        for idx in instance:
            # since we have 1 indexed features but zero indexed arrays
            example[idx - 1] = instance.get(idx)
        return example[:self.nfeatures]

    def get_feature(self, instance, feature):
        return instance.get(feature + 1, default=0)

    def calculate_max_feature(self):
        nfeatures = self.nfeatures
        for instance in self.instances:
            nfeatures = max(instance.max_feature(), nfeatures)
        self.nfeatures = nfeatures
        self.ninstances = len(self.instances)
        pass

    def initialize(self):
        instances = self.instances
        # will be the mean of all of these
        p = np.zeros((self.ninstances, self.K))
        # construct array of initial guess assignments
        # note that we roll one place to account for zero indexing
        assignments = np.mod(range(0, self.ninstances), self.K)
        # let this be zero
        x = np.zeros((self.ninstances, self.nfeatures))
        for idx in range(0, self.ninstances):
            x[idx, :] = self.load_example(instances[idx])
        for xid in range(0, self.ninstances):
            p[xid, assignments[xid]] = 1

        S = np.zeros((self.nfeatures))
        for j in range(0, self.nfeatures):
            S[j] = .01*np.var(x[:, j])*self.ninstances/(self.ninstances-1)

        S[S == 0] = 1
        self.S = S
        self.post = p
        self.update_params()
        pass

    def squared_distance(self, x, y):
        return distance.euclidean(x, y)**2

    # a function to return the analytical solution to the log
    # of the probability.
    def log_prob(self, x, mean, var):
        return (-.5*np.log(2*var**2*np.pi) - (x - mean)**2/float(2*var**2))
        #return 1/float(np.sqrt(2*var**2*np.pi))*np.exp(-(x - mean)**2/float(2*var**2))

    def get_posteriors(self):
        p = np.zeros((self.ninstances, self.K))
        # load the means locally for speed
        means = self.means
        var = self.var
        phi = self.phi
        instances = self.instances
        for idx in range(0, self.ninstances):
            x = self.load_example(instances[idx])
            k_clusts = np.zeros((self.K))
            for k in range(0, self.K):
                for j in range(0, self.nfeatures):
                    k_clusts[k] += self.log_prob(x[j], means[j, k], var[j, k])
            k_clusts = np.add(k_clusts, np.log(phi))
            p[idx, int(np.argmax(k_clusts))] = 1
        self.post = p
        pass

    def update_params(self):
        # use these locally for speed bump
        K = self.K
        p = self.post
        S = self.S
        phi = np.zeros((K))
        means = np.zeros((self.nfeatures, K))
        var = np.zeros((self.nfeatures, K))
        instances = self.instances
        for k in range(0, K):
            p_vec = p[:,k]
            # get the instance ids we want so we don't have to
            # check all of them
            p_ids = np.where(p_vec != 0)[0]
            ex_post = np.zeros((self.nfeatures, int(np.max([1, p_ids.shape[0]]))))
            for idx in range(0, len(p_ids)):
                ex_post[:, idx] = self.load_example(instances[p_ids[idx]])
            phi[k] = (p_ids.shape[0] + 1)/float(self.ninstances + K)

            # mean of the particular set of examples
            means[:, k] = ex_post.mean(axis=1)
            # variance of the set of examples
            # if we have 1 or 0 things in this cluster
            if (p_ids.shape[0] <= 1):
                var[:, k] = S
            # else, use max(S[j], unbiased variance[j])
            else:
                unbiased = p_ids.shape[0] - 1 
                for j in range(0, self.nfeatures):
                    for i in range(0, p_ids.shape[0]):
                        var[j,k] += self.squared_distance(ex_post[j, i], means[j, k])

                var[:, k] = var[:, k] / float(unbiased)
                var[:, k] = np.max(np.column_stack((var[:, k], S)), axis=1)
        # update with our maximized means
        self.phi = phi
        self.means = means
        self.var = var
        pass

    def train(self, instances):
        self.instances = instances
        self.calculate_max_feature()
        # initialize the parameters of the model (resps and means)
        self.initialize()
        for i in range(0, self.iterations):
            # get the respnsibilities, E step
            self.get_posteriors()
            # maximize the means, M step
            self.update_params()
        pass

    # make predictions baed on the shortest distance cluster to
    # a particular example
    def predict(self, instance):
        x = self.load_example(instance)
        k_clusts = np.zeros((self.K))
        means = self.means
        var = self.var
        nfeatures = self.nfeatures
        for k in range(0, self.K):
            for j in range(0, nfeatures):
                k_clusts[k] += self.log_prob(x[j], means[j, k], var[j, k])
        # return minimum cluster id that has the minimum distance
        return np.min(np.argmax(k_clusts))
