#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn.metrics import pairwise_distances_argmin
from sklearn.metrics.pairwise import euclidean_distances


# In[1]:


def find_clusters(X, n_clusters, rseed=3):
    
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    
    centers = X[i]

    while True:
        
        #возвращает номер объекта, к которому Xi ближе всего
        labels = pairwise_distances_argmin(X, centers)
        
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])
        
        if np.all(centers == new_centers):
            break
            
        centers = new_centers
    return centers, labels


# In[ ]:


def fakt(n):
    r = 1
    for i in range(n):
        i = i + 1
        r*=i;
    return r;

def bci(n,k):
    return fakt(n)/(fakt(k)*fakt(n-k));


# In[2]:


def adjusted_rand_index(confusion_matrix):
    
    length = len(confusion_matrix)
    n_ij = np.array([bci(x, 2) for x in list(confusion_matrix.reshape(length**2))]).sum()
    a_i = np.array([bci(x, 2) for x in list(confusion_matrix.sum(axis=0))]).sum()
    b_j = np.array([bci(x, 2) for x in list(confusion_matrix.sum(axis=1))]).sum()
    
    c = bci(confusion_matrix.sum(),2)
    e = a_i*b_j/c

    ari = (n_ij - e) / (1/2*(a_i+b_j) - e)
    
    return ari


# In[ ]:


def silhouette(x, labels, centers):
    silhouette = []
    
    for index, item in enumerate(x):
        
        label_item = labels[index]
        label_closest_cluster = findClosestCluster(x, index, centers)

        a = np.mean(euclidean_distances([list(item)], x[labels == label_item]))
        b = np.mean(euclidean_distances([list(item)], x[labels == label_closest_cluster]))
            
        s_item = (b - a)/max(a, b)
        
        silhouette.append(s_item)
        
    silhouette = np.array(silhouette)
    return silhouette.mean()


# In[ ]:


def findClosestCluster(x, idx, centers):
    
    distans_to_clusters = list(euclidean_distances([list(x[idx])], centers)[0])
    distans_to_clusters = [(distans_to_clusters[i], i) for i in range(len(distans_to_clusters))]
    data_type = [('dist', float),('label', int)]

    distans_to_clusters = np.array(distans_to_clusters, dtype = data_type)
    distans_to_clusters = np.sort(distans_to_clusters, order = 'dist')
    if(len(distans_to_clusters) == 1):
         return distans_to_clusters[0]['label']
    
    return distans_to_clusters[1]['label']

