import numpy as np


class ChainMRFPotentials:
    def __init__(self, data_file):
        with open(data_file) as reader:
            for line in reader:
                if len(line.strip()) == 0:
                    continue

                split_line = line.split(" ")
                try:
                    self._n = int(split_line[0])
                except ValueError:
                    raise ValueError("Unable to convert " + split_line[0] + " to integer.")
                try:
                    self._k = int(split_line[1])
                except ValueError:
                    raise ValueError("Unable to convert " + split_line[1] + " to integer.")
                break

            # create an "(n+1) by (k+1)" list for unary potentials
            self._potentials1 = [[-1.0] * ( self._k + 1) for n in range(self._n + 1)]
            # create a "2n by (k+1) by (k+1)" list for binary potentials
            self._potentials2 = [[[-1.0] * (self._k + 1) for k in range(self._k + 1)] for n in range(2 * self._n)]

            for line in reader:
                if len(line.strip()) == 0:
                    continue

                split_line = line.split(" ")

                if len(split_line) == 3:
                    try:
                        i = int(split_line[0])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[0] + " to integer.")
                    try:
                        a = int(split_line[1])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[1] + " to integer.")
                    if i < 1 or i > self._n:
                        raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
                    if a < 1 or a > self._k:
                        raise Exception("given k=" + str(self._k) + ", illegal value for a: " + str(a))
                    if self._potentials1[i][a] >= 0.0:
                        raise Exception("ill-formed energy file: duplicate keys: " + line)
                    self._potentials1[i][a] = float(split_line[2])
                elif len(split_line) == 4:
                    try:
                        i = int(split_line[0])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[0] + " to integer.")
                    try:
                        a = int(split_line[1])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[1] + " to integer.")
                    try:
                        b = int(split_line[2])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[2] + " to integer.")
                    if i < self._n + 1 or i > 2 * self._n - 1:
                        raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
                    if a < 1 or a > self._k or b < 1 or b > self._k:
                        raise Exception("given k=" + self._k + ", illegal value for a=" + str(a) + " or b=" + str(b))
                    if self._potentials2[i][a][b] >= 0.0:
                        raise Exception("ill-formed energy file: duplicate keys: " + line)
                    self._potentials2[i][a][b] = float(split_line[3])
                else:
                    continue

            # check that all of the needed potentials were provided
            for i in range(1, self._n + 1):
                for a in range(1, self._k + 1):
                    if self._potentials1[i][a] < 0.0:
                        raise Exception("no potential provided for i=" + str(i) + ", a=" + str(a))
            for i in range(self._n + 1, 2 * self._n):
                for a in range(1, self._k + 1):
                    for b in range(1, self._k + 1):
                        if self._potentials2[i][a][b] < 0.0:
                            raise Exception("no potential provided for i=" + str(i) + ", a=" + str(a) + ", b=" + str(b))

    def chain_length(self):
        return self._n

    def num_x_values(self):
        return self._k

    def potential(self, i, a, b = None):
        if b is None:
            if i < 1 or i > self._n:
                raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
            if a < 1 or a > self._k:
                raise Exception("given k=" + str(self._k) + ", illegal value for a=" + str(a))
            return self._potentials1[i][a]

        if i < self._n + 1 or i > 2 * self._n - 1:
            raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
        if a < 1 or a > self._k or b < 1 or b > self._k:
            raise Exception("given k=" + str(self._k) + ", illegal value for a=" + str(a) + " or b=" + str(b))
        return self._potentials2[i][a][b]


class SumProduct:
    def __init__(self, p):
        self._potentials = p
        self.k = self._potentials.num_x_values()
        self.n = self._potentials.chain_length()
        # TODO: EDIT HERE
        # add whatever data structures needed

    def marginal_probability(self, x_i):
        # TODO: EDIT HERE
        # should return a python list of type float, with its length=k+1, and the first value 0
        # store the messages here, as right message, left message, and current factor
        result = np.ones((self.k + 1, 3))
        from pdb import set_trace as pdb
        pdb()
        result[:, 0] = self.get_message(x_i, -1)
        pdb()
        result[:, 1] = self.get_message(x_i, 1)
        pdb()
        result[:, 2] = self.get_current_mu(x_i)
        pdb()
        # This code is used for testing only and should be removed in your implementation.
        # It creates a uniform distribution, leaving the first position 0
        # chop off the first value
        result = np.prod(result, axis=1)
        result = np.divide(result, np.sum(result))
        result[0] = 0
        return result

    # get a message at a particular node, where direction tells us which way to look
    # -1 for left, 1 for right
    def get_message(self, x_i, direction):
        # Want to handle edge case properly in case we start at a border and need
        # the left and right messages, we could get an error if this is not handled
        # if it's the last node to consider
        if (x_i == 1 and direction == -1) or (x_i == self.k and direction == 1):
            return self.get_current_mu(x_i)
        else: # know that we are not at an end, so use recursion here
            marginal = np.zeros((self.k + 1, self.k + 1))
            # make a matrix of all the marginals
            for k1 in range(1, self.k + 1):
                for k2 in range(1, self.k + 1):
                    warn = 0
                    if (direction == -1):
                        warn = 1
                    marginal[k1, k2] = self._potentials.potential(self.n + x_i - warn, k1, k2)
                   
                        
            # sum across it for this x_i
            marginal = marginal.sum(axis=1)
            # take the product element wise with the next node
            result = np.multiply(marginal, self.get_message(x_i + direction, direction))
            return result

    def get_current_mu(self, x_i):
        result = np.zeros((self.k + 1))
        for k in range(1, self.k + 1):
            result[k] = self._potentials.potential(x_i, k)
        return result


class MaxSum:
    def __init__(self, p):
        self._potentials = p
        self._assignments = [0] * (p.chain_length() + 1)
        # TODO: EDIT HERE
        # add whatever data structures needed

    def get_assignments(self):
        return self._assignments

    def max_probability(self, x_i):
        # TODO: EDIT HERE
        return 0.0
