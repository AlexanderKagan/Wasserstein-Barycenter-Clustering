import numpy as np
import cvxpy as cp
from typing import List
from utils import DiscreteDistrib
from scipy.spatial import distance_matrix

from utils import discrete_wasserstein_distance


class D2Cluster:
    def __init__(self, K: int, m: int, p: float = 2):
        self.K = K
        self.p = p
        self.m = m

    def fit(self, Ps: List[DiscreteDistrib]):

        self.Ps = Ps
        self.X = np.vstack([P.x for P in Ps]).T  # d x (m1 + ... m_N)
        self.d = self.X.shape[0]
        self.n = self.X.shape[1]
        self.N = len(Ps)

        ...

    def _solve_full_batch_lp(self, x, cluster_labels):

        cost_mat = distance_matrix(x, self.X, self.p)
        pi = cp.Variable((self.m, self.n), pos=True)
        w = cp.Variable(self.m)

        unique_labels = np.unique(cluster_labels)
        constraints = [cp.sum(w) == 1, cp.sum(pi, 1) == w]
        for k in unique_labels:
            cluster_mask = cluster_labels == k
            constraint = cp.sum(pi[:, cluster_mask], 0) == w[cluster_mask]
            constraints.append(constraint)

        masks = np.vstack([cluster_labels == k for k in unique_labels])
        objective = cp.Minimize(
            cp.sum(np.apply_along_axis(lambda mask: cp.trace(cost_mat[:, mask] @ pi[:, mask].T), 0, masks))
        )
        prob = cp.Problem(objective, constraints)
        _ = prob.solve()
        return w.value, pi.value

    def _find_optimum_support_points(self, pi, w):
        return self.X @ pi.T @ np.diag(w) / self.N

    def _find_centroid(self, cluster_labels, x_init=None, tol=1e-3):
        x = x_init if x_init is not None else np.random.randn(self.d, self.d.m)
        norm_change = np.inf
        while norm_change > tol:
            pi, w = self._solve_full_batch_lp(x, cluster_labels)
            x_new = self._find_optimum_support_points(pi, w)
            norm_change = np.linalg.norm(x - x_new)
            x = x_new
        return x
