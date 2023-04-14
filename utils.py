import numpy as np
import cvxpy as cp
from scipy.spatial import distance_matrix


class DiscreteDistrib:
    def __init__(self, w, x):
        assert np.all(w >= 0), "weights should be positive"
        assert np.allclose(np.sum(w), 1.), "weights should sum up to one"

        self.w = w
        self.x = x

    def __repr__(self):
        return str(list(zip(self.x, self.w)))


def discrete_wasserstein_distance(P1: DiscreteDistrib, P2: DiscreteDistrib, p=2,
                                  return_coupling=False):
    w1, x1 = P1.w, P1.x
    w2, x2 = P2.w, P2.x

    cost_matrix = distance_matrix(x1, x2, p)
    pi = cp.Variable((w1.shape[0], w2.shape[0]), pos=True)
    objective = cp.Minimize(cp.sum(cp.multiply(pi, cost_matrix)))
    constraints = [cp.sum(pi, axis=0) == w2, cp.sum(pi, axis=1) == w1]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    dist = prob.value ** (1 / p)
    return (dist, pi.value) if return_coupling else dist
