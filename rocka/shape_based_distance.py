# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 15:28:32 2019

@author: PB
"""

import numpy as np

__all__ = ['sbd_ele','SBD']

def sbd_ele(values1,values2):
    """
    Given two time seires `values1' and `values2`,cross-correlation slides 
    `values2' to `values1` to compute the inner-product for each shift `s`,the
    range of shift `s` ∈ [-len(values2) + 1, len(values1)-1]
    SBD is based of cross-correlation. SBD ranges from 0 to 2, where 0 means 
    two time series have exactly the same shape. A smaller SBD means higher 
    shape similarity.
    
    Args:
        values1(np.ndarray): time series 1
        values2(np.ndarray): time series 2
        
    Returns:
        np.float32: the SBD between `values1` and `values2`
    """
    #get the 2 norm
    l2_values1 = np.linalg.norm(values1)
    l2_values2 = np.linalg.norm(values2)
    #get the cross-correlation of each shift `s`
    cross_corre = np.convolve(values1,values2,mode = 'full')
    
    #return the SBD between `values1` and `values2`
    return 1 - np.max(cross_corre)/(l2_values1 * l2_values2)

def SBD(values_list,minPts = 4):
    """
    Caculate the shape based distance(SBD) between any two time series for 
    similarity measure. SBD is used for DBSCAN for clustering.
    The main idea of DBSCAN is to find some cores in dense regions,and then 
    expand the cores by transitivity of similarity to form clusters.
    
    Args:
        List(np.ndarray): a list consists of all time series(np.ndarray),the 
            lengths of different time series could be different
        minPts(np.int32): The core `p` in DBSCAN is defined as an object that has
            at least `minPts` objects within a distance of ϵ from it(excluding
            `p`). The default value of `minPts` is 4.
    
    Returns:
        np.ndarray: for each time series, take the SBD between it and its 
        minPts-Nearest-Neighbor(KNN). The SBDs of all time series in `values`
        returned as an np.ndarray.
    """
    
    if len(values_list) < minPts:
        raise ValueError ('`values_list` must contain more than %d time series'\
                          %minPts)
    if len(values_list[0].shape) != 1:
        raise ValueError ('`values` must be a 1-D array')
    if (type(minPts) is not int) or (minPts < 1):
        raise ValueError ('`minPts` must be a positive integar')
    
    #Caculate the SBD between any two time time series
    sbd_matrix = np.zeros((len(values_list),len(values_list)))
    for i in range(len(values_list)):
        for j in range(i,len(values_list)):
            sbd_matrix[i][j] = sbd_ele(values_list[i],values_list[j])
            sbd_matrix[j][i] = sbd_matrix[i][j]
    
    #Return the minPts nearest SBD for each time series(excluding itself)
    ret_sbd = np.zeros(len(values_list))
    for i in range(len(values_list)):
        src_index = np.argsort(sbd_matrix[i])
        ret_sbd[i] = sbd_matrix[i][src_index][minPts]
    
    return sbd_matrix,ret_sbd