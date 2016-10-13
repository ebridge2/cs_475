# adaboost.py
# written by Eric Bridgeford
#
# a class for the adaboost algorithm
#

from cs475_types import Predictor
from abc import ABCMeta, abstractmethod
import numpy as np


class Adaboost(Predictor):
    __metaclass__ = ABCMeta

    # a datatype for how we will sort our hypothesis classes.
    sort_dtype = [('label', float), ('value', float)]

    def __init__(self, iterations):
        super(Adaboost, self).__init__()
        self.iter_boost = iterations
        self.s2l_key = {'0': -1, '1' : 1}
        self.l2s_key = {1 : '1', 0 : '1', -1: '0'}
        self.nfeatures = 0
        pass

    def learn_features(self):
        self.ninstances = len(self.instances)
        for instance in self.instances:
            self.nfeatures = np.max([self.nfeatures, instance.max_feature()])
        self.unique_labels = np.unique(self.s2l_key.values())
        pass

    def get_label(self, instance):
        return self.s2l_key[instance.get_label()]

    def get_feat(self, instance, feature):
        return instance.get(feature + 1, default=0)

    def get_feature_set(self, feature):
        values = np.array([tuple((self.get_label(instance),
                          self.get_feat(instance, int(feature)))) for instance in self.instances],
                          dtype=self.sort_dtype)
        return values

    def make_hypothesis_class(self, instances):
        self.instances = instances
        self.learn_features()
        self.hyp_dtype =  dtype=[('feature', int), ('cutoff', float),
                                 ('greater', int), ('less', int), ('error', float)]
 
        # overallocate so we don't have to copy ever
        self.hyp_set = np.array([tuple((0, np.nan, 0, 0, np.nan))
                                for i in range(0, self.nfeatures*len(self.instances))],
                                dtype = self.hyp_dtype)
        count = 0 # default to zero and replace as needed to hypo set
        for feature_id in range(0, self.nfeatures):
            vals = self.get_feature_set(feature_id)
            # store the labels with the values so we can simplify hypothesis class significantly
            this_feature = np.zeros(0)
            unique_vals = np.unique(vals['value'])
            for idx in range(0, unique_vals.shape[0] - 1):
                this_val = unique_vals[idx]
                next_val = unique_vals[idx + 1]
                cutoff = np.mean([this_val, next_val])
                # put our potential hypotheses in a 0-indexed list
                (gvals, gv_ct) = np.unique(vals[vals['value'] > cutoff]['label'],
                                           return_counts=True)
                (lvals, lv_ct) = np.unique(vals[vals['value'] <= cutoff]['label'],
                                           return_counts=True)
                greater = gvals[np.argmax(gv_ct)]
                less = lvals[np.argmax(lv_ct)]
                self.hyp_set[count] = tuple((feature_id, cutoff, greater, less, 0))
                count = count + 1
        # return only as needed
        self.hyp_set = self.hyp_set[0:count]
        pass

    def calculate_hyp(self, hypothesis, value):
        if (value > hypothesis['cutoff']):
            return hypothesis['greater']
        else:
            return hypothesis['less']
 
    def train(self, instances):
        self.make_hypothesis_class(instances)
        self.hyp = np.array([tuple((0, 0, 0, 0, 0))
                                    for i in range(0, self.iter_boost)],
                            dtype=self.hyp_dtype)
        self.alpha = np.array([1 for i in range(0, self.iter_boost)], dtype=float) # initialize our weight vecs
        D = np.zeros(self.ninstances)
        D.fill(1/float(self.ninstances))
        t = 0
        tol = 1 # default to worst case unless data tells us otherwise
        while (t < self.iter_boost and tol > .000001):
            # one best hypothesis per feature will be compared
            for idx in range(0, self.hyp_set.shape[0]):
                new_err = 0
                hypo = self.hyp_set[idx]
                feature = int(hypo['feature'])
                examples = self.get_feature_set(feature)
                for i in range(0, examples.shape[0]):
                    example = examples[i]
                    yhat = self.calculate_hyp(hypo, example['value'])
                    new_err += D[i]*float(int(yhat) != example['label'])
                self.hyp_set[idx]['error'] = new_err
            sorted_feat_hyps = np.sort(self.hyp_set, order=('error'))
            print sorted_feat_hyps
            self.hyp[t] = sorted_feat_hyps[0] # lowest error
            tol = self.hyp[t]['error'] # tolerance is our error
            if (tol != 0): # in case we have a divide by zero
                best_feature = self.hyp[t]['feature']
                feature_vals = self.get_feature_set(best_feature)
                self.alpha[t] = .5*np.log((1-tol)/tol)
                power_ar = np.array([-self.alpha[t] *\
                                     float(self.calculate_hyp(self.hyp[t], example['value']) * example['label'])
                                     for example in feature_vals])
                exp_term = np.exp(power_ar)
                Z = np.dot(D, exp_term).sum()
                print Z
                D = 1/float(Z) * np.multiply(D, exp_term)
            t = t + 1
        self.hyp = self.hyp[0:t]
        print self.hyp
        self.alpha = self.alpha[0:t]
        print self.alpha
        pass

    def predict(self, instance):
        dtype_predict = [('value', float), ('label', int)]
        possible_labels = np.array([tuple((0, label)) for label in self.unique_labels],
                                   dtype=dtype_predict)
        hypo_per_t = np.array([self.calculate_hyp(self.hyp[t], self.get_feat(instance, self.hyp[t]['feature']))
                              for t in range(0, self.hyp.shape[0])])
        for i in range(0, possible_labels.shape[0]):
            votes = np.array(hypo_per_t == possible_labels[i]['label']).astype(int) # number of matches
            print "votes" + str(votes)
            possible_labels[i]['value'] = np.dot(self.alpha, votes)
        yhat = np.sort(possible_labels, order=('value', 'label'))[possible_labels.shape[0]- 1]
        return self.l2s_key[yhat['label']]
