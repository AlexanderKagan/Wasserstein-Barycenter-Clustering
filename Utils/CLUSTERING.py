#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cvxpy as cp
from typing import List
from scipy.spatial import distance_matrix
import scipy.stats as ss
import ot ## should we use the POT module or use what we wrote
import matplotlib.pyplot as plt
from itertools import chain

from UTILS import DiscreteDistrib, discrete_wasserstein_distance
from BADMM import badmm_centroid_update



##################################################
# 4 Small functions needed for clustering function

# 1
def gather_cluster(labels, P_list, cluster_number):

    # Gathers the distributions assigned to a certain cluster.
    #
    
    indices = np.array(labels) == cluster_number
    distributions = np.array(P_list)[indices]
    
    return(distributions)
    


# 2
#def wassdistance(source, target):
#
#    # Computes the Wasserstein distance
#    # Uses Python Optimal Transport package -- is faster than our current one
#
#    M = ot.dist(source.x, target.x, metric = 'minkowski')
#    distance = ot.emd2(source.w, target.w, M)
#
#    return distance


# 3
def init_centroid(K, m, d):
    
    # initialize a set of centroids
    
    centroids = []
    for i in range(K):
        X = np.random.randn(m,d)
        W = np.ones(m) / m

        centroids.append(DiscreteDistrib(W,X))
        
    return(centroids)


# 4
def assign_label(Ps, centroids):

    # assigns clusters to each distribution according to wass dist
        
    labels = []

    for i in range(len(Ps)):

        distances = [discrete_wasserstein_distance(Ps[i], centroids[j]) for j in range(len(centroids))]
        labels.append(distances.index(min(distances)))
        
    
    return(labels)

##################################################


def clustering(Ps, K, numItermax = 1e4, centroid_update_kwargs={}):

    # Clustering framework
    # replace the centroid update step w/ Wasserstein computation algorithm


    m = np.floor(np.mean([P.x.shape[0] for P in Ps])).astype('int') #average of sizes of each P
    d = Ps[0].x.shape[1]
    
    centroids = init_centroid(K, m, d)
    labels = assign_label(Ps, centroids)
    
    for num in range(int(numItermax)):
        
        # centroid update step
        centroids = []

        for k in range(K):
            current_cluster = gather_cluster(labels, Ps, k)
            
            # if a cluster is empty, create random centroid again
            if len(current_cluster) == 0: 
                new_X = np.random.randn(m, d)
                new_W = np.ones(m) / m
                new_centroid = DiscreteDistrib(new_W, new_X)
                
            else:
                new_centroid = badmm_centroid_update(current_cluster, **centroid_update_kwargs)

            centroids.append(new_centroid)
        
        #print('DONE')
        
        # assign step
        prev_labels = labels
        labels = assign_label(Ps, centroids)
        
        # stopping criterion -- can also change to np.allclose
        if (sum(np.not_equal(prev_labels, labels)) < 1): break
    
        
    
    return(labels)



