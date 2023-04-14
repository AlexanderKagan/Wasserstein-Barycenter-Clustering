import numpy as np
import cvxpy as cp
from typing import List
from scipy.spatial import distance_matrix
import scipy.stats as ss
import ot ## should we use the POT module or use what we wrote
import matplotlib.pyplot as plt


class DiscreteDistrib:
    def __init__(self, w, x):
        assert np.all(w >= 0), "weights should be positive"
        assert abs(np.sum(w) - 1) <= 1e-9, "weights should sum up to one"

        self.w = w
        self.x = x


def discrete_wasserstein_distance(P1: DiscreteDistrib, P2: DiscreteDistrib, p=2,
                                  return_coupling=False):
    w1, x1 = P1.w, P1.x
    w2, x2 = P2.w, P2.x

    cost_matrix = distance_matrix(x1, x2, p)
    pi = cp.Variable((w1.shape[0], w2.shape[0]), pos=True)
    objective = cp.Minimize(cp.sum(cp.multiply(pi, cost_matrix)))
    constraints = [cp.sum(pi, axis=0) == w2.reshape((-1,)), cp.sum(pi, axis=1) == w1.reshape((-1,))]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    dist = prob.value ** (1 / p)
    return (dist, pi.value) if return_coupling else dist


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

    def _solve_full_batch_lp(self, P_index, x):

        #cost_mat = distance_matrix(x, self.X, self.p)
        pi = cp.Variable((self.m, self.n), pos=True)
        w = cp.Variable(self.m)

        unique_labels = np.unique(P_index)
        masks = np.vstack([P_index == k for k in unique_labels])

        constraints = [pi >= 0, w >= 0, w <= 1, cp.sum(w) == 1]
        for k in unique_labels:
            current_pi = pi[:, P_index == k]

            constraint1 = cp.sum(current_pi, 0) == w
            constraint2 = cp.sum(current_pi, 1) == self.Ps[k-1].w

            constraints.append(constraint1)
            constraints.append(constraint2)


        objective = cp.Minimize(
            cp.sum([cp.trace(distance_matrix(x.T, self.Ps[k-1].x, self.p) @ pi[:, masks[i,:]].T) for i in range(k)])
        )
        prob = cp.Problem(objective, constraints)
        _ = prob.solve(verbose = True)
        
        return w.value, pi.value

    
    def _find_optimum_support_points(self, w, pi):
        return self.X @ pi.T @ np.diag(1./w) / self.N

    
    def _find_centroid(self, P_index, x_init=None, tol=1e-3):
        x = x_init if x_init is not None else np.random.randn(self.d, self.d.m)
        norm_change = np.inf
        
        while norm_change > tol:
            
            w, pi = self._solve_full_batch_lp(P_index, x)
            x_new = self._find_optimum_support_points(w, pi)
            norm_change = np.linalg.norm(x - x_new)
            x = x_new
            
        return x



def disc_gauss_generator(n, mean, std, plotting = False):

    # creates a discrete gaussian distribution
    # with plotting option

    cont_normal = np.random.normal(mean, scale=std, size = n)
    count, bin_edges= np.histogram(cont_normal, bins = n)
    bins = (bin_edges[1:] + bin_edges[:-1]) / 2
    prob = count / count.sum() #normalize
    prob = np.reshape(prob, (n,1))
    bins = np.reshape(bins, (n,1)) # disc_wass_dist requires a second dimension

    discgauss = DiscreteDistrib(prob, bins)

    if plotting:
        plt.hist(cont_normal, bins = 10)

    return(discgauss)


