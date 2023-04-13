import numpy as np
from utils import DiscreteDistrib
from scipy import sparse
from scipy.spatial import distance_matrix


def badmm_centroid_update(stride, supp, w, c0: DiscreteDistrib,
                          rho=1e-2, nIter=1000, eps=1e-10,
                          tau=10, badmm_tol=1e-3):
    d = supp.shape[0]
    n = len(stride)
    m = len(w)

    c = c0

    support_size = len(c.w)
    posvec = np.concatenate(([0], np.cumsum(stride)))

    X = np.zeros((support_size, m))
    Y = np.zeros_like(X)
    Z = np.zeros((support_size, np.sum(stride)))

    spIDX_rows = np.zeros(support_size * m, dtype=int)
    spIDX_cols = np.zeros_like(spIDX_rows, dtype=int)

    for i in range(n):
        xx, yy = np.meshgrid(i * support_size + np.arange(0, support_size),
                             np.arange(posvec[i], posvec[i + 1]))
        ii = support_size * posvec[i] + np.arange(support_size * stride[i])
        spIDX_rows[ii] = xx.flatten()
        spIDX_cols[ii] = yy.flatten()

    spIDX = np.kron(np.eye(support_size), np.ones((1, n)))

    for i in range(n):
        Z[:, posvec[i]: posvec[i + 1]] = 1 / (support_size * stride[i])

    C = distance_matrix(c.x.T, supp.T) ** 2

    for iteration in range(nIter):
        # update X
        X = Z * np.exp((C + Y) / (-rho)) + eps
        X = X * (w / np.sum(X, axis=0)).T

        # update Z
        Z0 = Z
        Z = X * np.exp(Y / rho) + eps
        spZ = sparse.csr_matrix((Z.T.ravel(order='F'), (spIDX_rows, spIDX_cols)),
                                shape=(support_size * n, m))

        tmp = np.sum(spZ, axis=1)
        tmp = np.reshape(tmp, (support_size, n))
        dg = c.w / tmp
        dg = sparse.csr_matrix((np.array(dg).flatten(),
                                (np.arange(n * support_size), np.arange(n * support_size))))

        Z = spIDX @ dg @ spZ

        # update Y
        Y = Y + rho * (X - Z)

        # update c.w
        tmp = tmp / np.sum(tmp, axis=0)

        sumW = np.array(np.sum(np.sqrt(tmp), axis=1)) ** 2
        c.w = sumW / np.sum(sumW)
        if iteration % tau == 0:
            c.x = supp @ X.T / np.tile(np.sum(X, axis=1), (d, 1)).T
            C = distance_matrix(c.x.T, supp.T) ** 2
        if iteration % 100 == 0:
            primres = np.linalg.norm(X - Z, 'fro') / np.linalg.norm(Z, 'fro')
            dualres = np.linalg.norm(Z - Z0, 'fro') / np.linalg.norm(Z, 'fro')
            print(f'\t {iter} {np.sum(C * X) / n} {primres} {dualres}', end=' ')
            if np.sqrt(dualres * primres) < badmm_tol:
                break
    return c
