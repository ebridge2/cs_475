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
    sort_dtype = [('label', int), ('value', float)]

    def __init__(self, iterations):
        super(Adaboost, self).__init__()
        self.iter_boost = iterations
        self.s2l_key = {'0': -1, '1' : 1}
        self.l2s_key = {1 : '1', 0 : '1', -1: '0'}
        self.nfeatures = 0
        pass

    def learn_features(self, instances):
        self.instances = instances
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
                          self.get_feat(instance, feature))) for instance in self.instances],
                          dtype=self.sort_dtype)
        return np.sort(values, order=['value', 'label'])
 


    def make_hypothesis_class(self, instances):
        self.learn_features(instances)
        self.hypothesis_set = []
        for feature_id in range(0, self.nfeatures):
            sorted_vals = self.get_feature_set(feature_id)
            # store the labels with the values so we can simplify hypothesis class significantly
            this_feature = np.zeros(0)
            for idx in range(0, sorted_vals.shape[0] - 1):
                # if we have a new value, and if the label has changed
                # note that we greatly simplify our hypothesis class. For example, if we have the following
                # dataset: + + + + + + - - - - + - - - -
                # we would never place a potential hypothesis btwn two + + or two - -, since there
                # would in the weak classifier case always be a hypohthesis on the end that is 
                # better
                this_val = sorted_vals[idx]
                # some clever logic tricks. There are a lot of possible situations to consider, and without
                # building a datastructure that comprises unevenly linked lists btwn attributes (we need to
                # consider situations where we have same label on the same value and skip making a hypothesis here,
                # whether we have a same label as the next value to appear and then skip making a hypothesis here,
                # and finally consider when we are the last of a string of same labels on a particular value
                # the next label consists of a bunch of the same label, and then somewhere in there is a 
                # different label on the next value to appear.
                condition_ra_same_value = np.where(np.logical_and(sorted_vals['label'] == this_val['label'],
                                                                  sorted_vals['value'] == this_val['value']))[0]
                if (condition_ra_same_value.shape[0] != 0):
                    # the next different labels to appear with value greater than our current value
                    condition_idxs = np.where(np.logical_and(sorted_vals['label'] != this_val['label'],
                                                             sorted_vals['value'] >= this_val['value']))[0]
                    # if we have none of the other labels after this one, we are finished
                    if (condition_idxs.shape[0] != 0):
                        # the next same labels to appear with value greater than our current value
                        condition_ra_next_same_label = np.where(np.logical_and(sorted_vals['label'] == this_val['label'],
                                                                               sorted_vals['value'] > this_val['value']))[0]
                        next_val_id = np.min(condition_idxs)
                        if (condition_ra_next_same_label.shape[0] != 0):
                            next_val_same_label = np.min(condition_ra_next_same_label)
                            if (sorted_vals[next_val_id]['value'] > sorted_vals[next_val_same_label]['value']):
                                next_val_id = None
                    else:
                        next_val_id = None
                    if (next_val_id is not None):
                        next_val = sorted_vals[next_val_id]
                        # constrain hypothesis set greatly
                        # first need to check whether we are at the end of our value chains
                        # this prevents us from having a bunch of labels for one value and not
                        # considering it any change from the next point
                        # if there is a reason to add the feature, pick it
                        # uses max-margin (never going to split when the labels are the same on both
                        # sides of the cutoff)
                        this_feature = np.union1d(this_feature, [float(np.mean([this_val['value'], next_val['value']]))])
                # put our potential hypotheses in a 0-indexed list
            self.hypothesis_set.append(this_feature)

    def calculate_hypothesis(self, feat, c, sorted_vals):
        possible_values = np.zeros(self.unique_labels.shape[0])
        if (feat >= c):
            for label_id in range(0, self.unique_labels.shape[0]):
                possible_values[label_id] = np.where(np.logical_and(sorted_vals['value'] >= c,
                                                    sorted_vals['label'] == self.unique_labels[label_id]
                                                    ))[0].shape[0]
        else:
            for label_id in range(0, self.unique_labels.shape[0]):
                possible_values[label_id] = np.where(np.logical_and(sorted_vals['value'] < c,
                                                    sorted_vals['label'] == self.unique_labels[label_id]
                                                    ))[0].shape[0]
        return self.unique_labels[np.argmax(possible_values)]
 
    def train(self, instances):
        self.make_hypothesis_class(instances)
        dtype_hyp = [('feature', int), ('cutoff', float), ('error', float)]
        self.hyp = np.array([tuple((0, 0, 0)) for i in range(0, self.iter_boost)],
                            dtype=dtype_hyp)
        D = np.zeros(self.ninstances)
        D.fill(1/float(self.ninstances))
        t = 0
        tol = 1 # default to worst case unless data tells us otherwise
        self.alpha = np.ones(self.iter_boost) # initialize our weight vecs
        while (t < self.iter_boost and tol > .000001):
            # one best hypothesis per feature will be compared
            hyp_c = np.array([tuple((0, 0, 0)) for i in range(0, self.nfeatures)],
                             dtype=dtype_hyp)
            for j in range(0, len(self.hypothesis_set)):
                feat_err = np.array([tuple((j, i, 0)) for i in self.hypothesis_set[j]],
                                       dtype = dtype_hyp)
                sorted_vals = self.get_feature_set(j)
                for i in range(0, len(self.instances)):
                    example = self.instances[i]
                    feat = self.get_feat(example, j)
                    y = self.get_label(example)
                    for c in range(0, len(self.hypothesis_set[j])):
                        cutoff = feat_err[c]['cutoff']
                        yhat = self.calculate_hypothesis(feat, cutoff, sorted_vals)
                        feat_err[c]['error'] = feat_err[c]['error'] + D[i]*float(yhat != y)
                hyp_c[j] = np.sort(feat_err, order=('error', 'cutoff'))[0] # lowest error
            sorted_feat_hyps = np.sort(hyp_c, order=('error', 'feature'))
            self.hyp[t] = sorted_feat_hyps[0] # lowest error
            tol = self.hyp[t]['error'] # tolerance is our error
            if (tol != 0): # in case we have a divide by zero
                best_feature = self.hyp[t]['feature']
                feature_sorted_vals = self.get_feature_set(best_feature)
                self.alpha[t] = .5 * np.log((1-tol)/tol)
                power_ar = np.array([-self.alpha[t] *\
                                     self.calculate_hypothesis(self.get_feat(example,best_feature),
                                                               self.hyp[t]['cutoff'],
                                                               feature_sorted_vals) * self.get_label(example)
                                     for example in self.instances])
                exp_term = np.exp(power_ar)
                Z = np.dot(D, exp_term).sum()
                D = 1/float(Z) * np.multiply(D, exp_term)
            t = t + 1
        self.hyp = self.hyp[0:t]
        self.alpha = self.alpha[0:t]
        pass

    def predict(self, instance):
#        print "NEXT!"
        dtype_predict = [('value', float), ('label', int)]
        possible_labels = np.array([tuple((0, label)) for label in self.unique_labels],
                                   dtype=dtype_predict)
#        print self.hyp
#        print self.get_feature_set(self.hyp[0]['feature'])
#        print self.get_feat(instance, self.hyp[0]['feature'])
        hypo_per_t = np.array([self.calculate_hypothesis(self.get_feat(instance,
                                                                       self.hyp[t]['feature']),
                                                         self.hyp[t]['cutoff'],
                                                         self.get_feature_set(self.hyp[t]['feature']))
                              for t in range(0, self.hyp.shape[0])])
#        print hypo_per_t
        for i in range(0, possible_labels.shape[0]):
            votes = np.array((hypo_per_t == possible_labels[i]['label'])).astype(int) # number of matches
#            print votes
            possible_labels[i]['value'] = np.dot(self.alpha, votes)
#        print possible_labels
        yhat = np.sort(possible_labels, order=('value', 'label'))[possible_labels.shape[0]- 1]
        return self.l2s_key[yhat['label']]
