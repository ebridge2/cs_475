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
        self.init_table()
        # TODO: EDIT HERE
        # add whatever data structures needed

    def init_table(self):
        table = np.ones((self.n, self.k, 3))
        # build the left-neighbor direction, and then the right-neighbor
        # direction starting from the opposite end
        # use values precomputed for speed
        # start at the 0th node (1-indexed 1)
        table[0, :, 1] = self.get_current_message(0)
        for i in range(1, self.n):
            # get an array corresponding to the factor we just moved over
            table[i,:,0] = self.get_factor_message(self.n + i - 1,
                np.multiply(table[i-1,:,0], table[i-1,:,1]))
            table[i,:,1]  = self.get_current_message(i)
        table[self.n - 1, :, 1] = self.get_current_message(self.n - 1)
        for i in reversed(range(0, self.n - 1)):
            table[i,:,2] = self.get_factor_reverse(self.n + i,
                np.multiply(table[i+1,:,2], table[i+1,:,1]))
            table[i, :, 1] = self.get_current_message(i)
        self.table = table
        pass

    # note that we keep all code to be 0-indexed internally, and marginal_probability
    # swaps the x_i from 1 to zero indexing
    def marginal_probability(self, x_i):
        # TODO: EDIT HERE
        # should return a python list of type float, with its length=k+1, and the first value 0
        result = self.get_normalization(x_i-1)
        result = np.divide(result, np.sum(result))
        result = np.insert(result, 0, 0)
        return result

    # use this for normalization for sumprod
    def get_normalization(self, x_i):
        result = np.product(self.table[x_i,:,:], axis=1)
        return result

    # use this within maxsum
    def get_total_potentials(self, x_i):
        result = np.sum(np.log(self.table[x_i,:,:]), axis=1)
        return result

    # gets the message of a particular factor as we go from nodes 0 to n.
    # this obtains the message from the preceding node, and multiplies it element wise
    # with all the joints of a particular factor
    def get_factor_reverse(self, factor_id, mu):
        marginal = np.zeros((self.k, self.k))
        for k1 in range(0, self.k):
            for k2 in range(0, self.k):
                marginal[k1, k2] = self._potentials.potential(factor_id + 1, k1 + 1, k2 + 1)*mu[k2]
        marginal = marginal.sum(axis=1)
        return marginal

    # same idea as before
    def get_factor_message(self, factor_id, mu):
        marginal = np.zeros((self.k, self.k))
        for k1 in range(0, self.k):
            for k2 in range(0, self.k):
                marginal[k1, k2] = self._potentials.potential(factor_id + 1, k2 + 1, k1 + 1)*mu[k2]
        marginal = marginal.sum(axis=1)
        return marginal

    def get_current_message(self, x_i):
        result = np.zeros((self.k))
        for k in range(0, self.k):
            result[k] = self._potentials.potential(x_i + 1, k + 1)
        return result


class MaxSum:
    def __init__(self, p):
        self._potentials = p
        self.k = self._potentials.num_x_values()
        self.n = self._potentials.chain_length()
        self.init_table()
        self.sumprod = SumProduct(p)
        # TODO: EDIT HERE
        # add whatever data structures needed

    def init_table(self):
        table = np.zeros((self.n, self.k, 3))
        table[0, :, 1] = self.get_current_message(0)
        for i in range(1, self.n):
            # get an array corresponding to the factor we just moved over
            table[i,:,0] = self.get_factor_message(self.n + i - 1,
                np.add(table[i-1,:,0], table[i-1,:,1]))
            table[i,:,1]  = self.get_current_message(i)
        table[self.n - 1, :, 1] = self.get_current_message(self.n - 1)
        for i in reversed(range(0, self.n - 1)):
            table[i,:,2] = self.get_factor_reverse(self.n + i,
                np.add(table[i+1,:,2], table[i+1,:,1]))
            table[i, :, 1] = self.get_current_message(i)
        self.table = table
        pass

    def get_assignments(self):
        assignments = np.zeros((self.n))
        table = self.table
        for i in range(0, self.n):
            # compute the argmax sum for all values, and add one for 0-indexing
            assignments[i] = np.argmax(table[i,:,:].sum(axis=1)) + 1
        # put our zero pad on the front bc 0-indexing to 1-indexing
        self._assignments = assignments
        return np.insert(self._assignments, 0, 0)

    def max_probability(self, x_i):
        norm = self.sumprod.get_normalization(x_i - 1)
        return np.max(self.table[x_i-1,:,:].sum(axis=1))

    def get_factor_reverse(self, factor_id, mu):
        marginal = np.zeros((self.k, self.k))
        for k1 in range(0, self.k):
            for k2 in range(0, self.k):
                marginal[k1, k2] = np.log(self._potentials.potential(factor_id + 1, k1 + 1, k2 + 1)) + mu[k2]
        return marginal.max(axis=1)

    def get_factor_message(self, factor_id, mu):
        marginal = np.zeros((self.k, self.k))
        for k1 in range(0, self.k):
            for k2 in range(0, self.k):
                marginal[k1, k2] = np.log(self._potentials.potential(factor_id + 1, k2 + 1, k1 + 1)) + mu[k2]
        return marginal.max(axis=1)

    def get_current_message(self, x_i):
        result = np.zeros((self.k))
        for k in range(0, self.k):
            result[k] = self._potentials.potential(x_i + 1, k+1)
        return np.log(result)

