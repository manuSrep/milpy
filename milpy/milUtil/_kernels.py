
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

def rbfKernel(b1, b2, gamma):
    """
    The rbf kernel between  between all instances of two bags.
    """

    b1 = np.array(b1)
    b2 = np.array(b2)
    # squared euclidean distance between instances of two bags
    sqFrobNorm = cdist(b1, b2, "sqeuclidean")
    # radial basis function
    k = np.exp(-gamma * sqFrobNorm)
    return k


def rbfKernel_Mat(X, gamma):
    """
    The rbf kernel between all instances of a data set.
    """

    X = np.array(X)
    sqFrobNorm = squareform(pdist(X, "sqeuclidean"))
    # Create GP covariance matrices
    K = np.exp(-gamma * sqFrobNorm) # instance level covariance matrix
    K += np.identity(K.shape[0]) * 1e-7  # add some value to grantee a positive definite kernel

    assert not np.any(np.isnan(K)), 'NaN detected in covMat_K!'
    assert np.allclose(K,  K.T), 'covMat_K is not symmetric'
    assert np.all(np.linalg.eigvals(K) > 0), 'covMat_K is not positive definite'

    return K