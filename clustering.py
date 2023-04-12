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

import utilsV2


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
def wassdistance(source, target):

    # Computes the Wasserstein distance
    # Uses Python Optimal Transport package -- is faster than our current one
    
    M = ot.dist(source.x, target.x, metric = 'minkowski')
    dist = ot.emd2(source.w, target.w, M)
    
    return dist

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

        distances = [wassdistance(Ps[i], centroids[j]) for j in range(len(centroids))]
        labels.append(distances.index(min(distances)))
    
    return(labels)

##################################################





def clustering(Ps, K, numItermax = 1e4):

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
                ###### REPLACE with Barycenter computation ######
                new_X = np.mean(np.array([P.x for P in current_cluster]), axis = 0)
                new_W = Ps[0].w
                new_centroid = DiscreteDistrib(new_W, new_X)
                ######
                
            centroids.append(new_centroid)
        
        # assign step
        prev_labels = labels
        labels = assign_label(Ps, centroids)
        
        # stopping criterion -- can also change to np.allclose
        if (sum(np.not_equal(prev_labels, labels)) < 1): break
        
    
    return(labels)


######### TESTING

# 1D Gaussian toy example

def disc_gauss_generator(n, mean, std, plotting = False):

    # creates a discrete gaussian distribution
    # with plotting option

    cont_normal = np.random.normal(mean, scale=std, size = n)
    count, bin_edges= np.histogram(cont_normal, bins = n)
    bins = (bin_edges[1:] + bin_edges[:-1]) / 2
    prob = count / count.sum() #normalize
    prob = np.reshape(prob, (n,))
    bins = np.reshape(bins, (n,1)) # disc_wass_dist requires a second dimension

    discgauss = DiscreteDistrib(prob, bins)
    
    if plotting:
        plt.hist(cont_normal, bins = 10)

    return(discgauss)


# In[31]:


nsamples = 100

P1 = disc_gauss_generator(nsamples, 30, 3)
P2 = disc_gauss_generator(nsamples, 50, 3)
P3 = disc_gauss_generator(nsamples, 1000, 3)
P4 = disc_gauss_generator(nsamples, 1200, 3)


# In[97]:


clustering([P1, P2, P3, P4], 2, numItermax = 100)



