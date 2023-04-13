import numpy as np
from utils import DiscreteDistrib
from scipy import sparse
from scipy.spatial import distance_matrix
from typing import List
from sklearn.cluster import KMeans


def convert_Ps_to_supp_stride_w(Ps: List[DiscreteDistrib]):
    supp = np.vstack([P.x for P in Ps]).T
    stride = np.array([P.x.shape[0] for P in Ps])
    w = np.hstack([P.w for P in Ps])
    return supp, stride, w


def init_kmeans_center(Ps, support_size):
    supp, stride, w = convert_Ps_to_supp_stride_w(Ps)
    kmeans = KMeans(n_clusters=support_size)
    kmeans.fit(supp.T)

    x = kmeans.cluster_centers_.T
    w = np.histogram(kmeans.labels_, bins=support_size, range=(1, support_size + 1), density=True)[0]
    return DiscreteDistrib(w, x)


def badmm_centroid_update(Ps: List[DiscreteDistrib], c0: DiscreteDistrib = None,
                          rho=1e-2, nIter=10000, eps=1e-10,
                          tau=10, badmm_tol=1e-3,
                          verbose_interval=100):
    supp, stride, w = convert_Ps_to_supp_stride_w(Ps)
    d = supp.shape[0]
    n = len(stride)
    m = len(w)

    c = c0 if c0 is not None else init_kmeans_center(Ps, int(np.mean(stride)))

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

        dg = c.w.reshape(-1, 1) / tmp
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
            c.x = supp @ X.T / np.tile(np.sum(X, axis=1), (d, 1))
            C = distance_matrix(c.x.T, supp.T) ** 2
        if iteration % verbose_interval == 0:
            primres = np.linalg.norm(X - Z, 'fro') / np.linalg.norm(Z, 'fro')
            dualres = np.linalg.norm(Z - Z0, 'fro') / np.linalg.norm(Z, 'fro')
            cost = round(np.sum(C * X) / n, 3)
            print(f'Iter: {iteration}, Avg cost {cost}, Primal: {round(primres, 4)}, Dual: {round(dualres, 4)}')
            if np.sqrt(dualres * primres) < badmm_tol:
                print("Early stop activated!")
                break
    return c
