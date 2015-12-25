# coding: utf-8

"""
Implementation of algorithms proposed by:

    H. Huang, et al., "Unsupervised Feature Selection on Data Streams," Proc. of CIKM 2015, pp. 1031-1040 (Oct. 2015).
"""

import numpy as np
import numpy.linalg as ln

class GenFeatWeight:
    """
    Alg. 1: Prototype algorithm for feature weighting (p=2)
    """

    def __init__(self, m, k):
        """
        :param m: number of original features
        :param k: number of singular vectors (this can be the same as the number of clusters in the dataset)
        """

        self.m = m
        self.k = k

    def update(self, Yt):
        """
        Update the weights based on new inputs at time t,
        :param Yt: m-by-n_t input matrix from data stream
        """

        if hasattr(self, 'Y'):
            self.Y = np.hstack((self.Y, Yt))
        else:
            # for Y0, we need to first initialize Y
            self.Y = Yt

        U, s, V = ln.svd(self.Y, full_matrices=False)

        # According to Section 5.1, for all experiments,
        # the authors set alpha = 2^3 * sigma_k based on the pre-experiment
        alpha = (2 ** 3) * s[self.k-1]

        # solve the ridge regression by using the top-k singular values
        # X: m-by-k matrix (k <= ell)
        D = np.diag(s[:self.k] / (s[:self.k] ** 2 + alpha))
        X = np.dot(U[:, :self.k], D)

        return np.amax(abs(X), axis=1)
