import numpy as np
from sklearn.neighbors import NearestNeighbors, KDTree
from scipy.special import digamma

from typing import Optional

from sklearn.utils.parallel import Parallel, delayed


def compute_mi_cc(x: np.ndarray, y: np.ndarray, n_neighbors: int):
    """Compute mutual information between two continuous (potentially multivariate) variables.

    Parameters
    ----------
    x, y : ndarray, shape (n_samples, n_features)
        Samples of two continuous random variables

    n_neighbors : int
        Number of nearest neighbors to search for each point, see [1]_.

    Returns
    -------
    mi : float
        Estimated mutual information in nat units. If it turned out to be
        negative it is replaced by 0.

    Notes
    -----
    True mutual information can't be negative. If its estimate by a numerical
    method is negative, it means (providing the method is adequate) that the
    mutual information is close to 0 and replacing it by 0 is a reasonable
    strategy.

    References
    ----------
    .. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
           information". Phys. Rev. E 69, 2004.
    """

    assert x.shape[0] == y.shape[0], "x and y must have the same number of samples"
    assert x.ndim == 2, "x must be a 2D array"
    assert y.ndim == 2, "y must be a 2D array"

    n_samples = x.shape[0]

    xy = np.hstack((x, y))

    # Here we rely on NearestNeighbors to select the fastest algorithm.
    nn = NearestNeighbors(metric="chebyshev", n_neighbors=n_neighbors)

    nn.fit(xy)
    radius = nn.kneighbors()[0]
    radius = np.nextafter(radius[:, -1], 0)

    # KDTree is explicitly fit to allow for the querying of number of
    # neighbors within a specified radius
    kd = KDTree(x, metric="chebyshev")
    nx = kd.query_radius(x, radius, count_only=True, return_distance=False)
    nx = np.array(nx) - 1.0

    kd = KDTree(y, metric="chebyshev")
    ny = kd.query_radius(y, radius, count_only=True, return_distance=False)
    ny = np.array(ny) - 1.0

    mi = (
        digamma(n_samples)
        + digamma(n_neighbors)
        - np.mean(digamma(nx + 1))
        - np.mean(digamma(ny + 1))
    )

    return max(0, mi)


def compute_mi_cd(c: np.ndarray, d: np.ndarray, n_neighbors: int):
    """Compute mutual information between continuous and discrete variables.

    Parameters
    ----------
    c : ndarray, shape (n_samples, n_features)
        Samples of a continuous random variable.

    d : ndarray, shape (n_samples,)
        Samples of a discrete random variable.

    n_neighbors : int
        Number of nearest neighbors to search for each point, see [1]_.

    Returns
    -------
    mi : float
        Estimated mutual information in nat units. If it turned out to be
        negative it is replaced by 0.

    Notes
    -----
    True mutual information can't be negative. If its estimate by a numerical
    method is negative, it means (providing the method is adequate) that the
    mutual information is close to 0 and replacing it by 0 is a reasonable
    strategy.

    References
    ----------
    .. [1] B. C. Ross "Mutual Information between Discrete and Continuous
       Data Sets". PLoS ONE 9(2), 2014.
    """

    assert c.shape[0] == d.shape[0], "c and d must have the same number of samples"
    assert c.ndim == 2, "c must be a 2D array"
    assert d.ndim == 1, "d must be a 1D array"

    n_samples = c.shape[0]

    radius = np.empty(n_samples)
    label_counts = np.empty(n_samples)
    k_all = np.empty(n_samples)

    nn = NearestNeighbors()
    for label in np.unique(d):
        mask = d == label
        count = np.sum(mask)
        if count > 1:
            k = min(n_neighbors, count - 1)
            nn.set_params(n_neighbors=k)
            nn.fit(c[mask])
            r = nn.kneighbors()[0]
            radius[mask] = np.nextafter(r[:, -1], 0)
            k_all[mask] = k
        label_counts[mask] = count

    # Ignore points with unique labels.
    mask = label_counts > 1
    n_samples = np.sum(mask)
    label_counts = label_counts[mask]
    k_all = k_all[mask]
    c = c[mask]
    radius = radius[mask]

    kd = KDTree(c)
    m_all = kd.query_radius(c, radius, count_only=True, return_distance=False)
    m_all = np.array(m_all)

    mi = (
        digamma(n_samples)
        + np.mean(digamma(k_all))
        - np.mean(digamma(label_counts))
        - np.mean(digamma(m_all))
    )

    return max(0, mi)


def compute_EI_MI(
    excitation: np.ndarray, 
    inhibition: np.ndarray, 
    mi_E_I: bool = False,
    branch_activation: Optional[np.ndarray] = None,
    mi_EI_branch: bool = False,
    labels: Optional[np.ndarray] = None,
    mi_EI_labels: bool = False,
    n_neighbors: int = 10
):
    """
    Compute mutual information between excitatory and inhibitory branch activations.

    Parameters
    ----------
    excitation : np.ndarray, shape (n_samples,)
        Excitatory branch activations.
    inhibition : np.ndarray, shape (n_samples,)
        Inhibitory branch activations.
    mi_E_I : bool
        Whether to compute mutual information between excitatory and inhibitory branch activations.
    branch_activation : np.ndarray, shape (n_samples,)
        Branch activations.
    mi_EI_branch : bool
        Whether to compute mutual information between excitatory and inhibitory branch activations.
    labels : np.ndarray, shape (n_samples,)
        Labels.
    mi_EI_labels : bool
        Whether to compute mutual information between excitatory and inhibitory branch activations.
    n_neighbors : int
        Number of nearest neighbors to search for each point.
    """

    assert excitation.shape[0] == inhibition.shape[0], "excitation and inhibition must have the same number of samples"
    assert excitation.ndim == 1, "excitation must be a 1D array"
    assert inhibition.ndim == 1, "inhibition must be a 1D array"

    excitation = excitation[:, None]
    inhibition = inhibition[:, None]

    mi_dict = {}

    if mi_E_I:
        mi_dict["E_I"] = compute_mi_cc(excitation, inhibition, n_neighbors)
    
    if mi_EI_branch or mi_EI_labels:
        ei = np.hstack((excitation, inhibition))
    
    if mi_EI_branch and branch_activation is not None:
        mi_dict["EI_branch"] = compute_mi_cc(ei, branch_activation[:, None], n_neighbors)
    
    if mi_EI_labels and labels is not None:
        mi_dict["EI_labels"] = compute_mi_cd(ei, labels, n_neighbors)
    
    return mi_dict









