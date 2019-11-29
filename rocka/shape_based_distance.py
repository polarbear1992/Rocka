# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 18:05:24 2019

@author: PB
"""

import numpy as np 

__all__ = ['density_radius']


def density_radius(sbd_arr,len_thresh,max_radius,slope_thresh,slope_diff_thresh):
    """
    Given K-Nearest-Neighbor SBDs of each sample,calculate the density radius
    for DESCAN clustering.
    
    Args:
        `sbd_arr`: np.ndarray, array of the K-Nearest-Neighbor SBD of each 
            sample.
        `len_thresh`: np.int32, the length of traget SBDs for candidate radius
            search.
        `max_radius`: np.float32, candidate radius are no larger than 
            `max_radius`.
        `slope_thresh`: np.float32, the slopes on the left and right of 
            candidate point are no larger than `slope_thresh`
        `slope_diff_thresh`: np.float32, the diff between leftslope and right-
            slope of candidate point are no larger than `slope_diff_thresh`
            
    Returns:
        np.float32: the final density radius is the largest value of all 
            candidate radii.
    """
    src_index = np.argsort(sbd_arr)
    sbd_arr_sorted = sbd_arr[src_index][::-1]
    candidates_index = np.argwhere(sbd_arr_sorted<=max_radius)
    start = np.min(candidates_index)
    end = len(sbd_arr_sorted)
    
    def find_candidate_radius(sbd_arr_sorted,start,end,candidates):
        """
        Given reverse sorted K-Nearest-Neighbor SBDs of each sample,calculate the density 
        radius for DESCAN clustering.
        A divide and conquer strategy is used for candidate radius finding.
        
        Args:
            `sbd_arr_sorted`: np.ndarray, reverse sorted array of the K-Nearest
                -Neighbor SBD of each sample.
            `start`: np.int32, the begain index of target SBDs.
            `end`: np.int32, the end index of target SBDs.
            `candidates`: np.ndarray, the indexes of all candidate radii.
            
        Returns:
            `candidates`: np.ndarray, the indexes of all candidate radii.
        """
        if end - start <= len_thresh:
            return
        radius,diff = -1,2
        for i in range(start+1,end):
            leftslope = (sbd_arr_sorted[i]-sbd_arr_sorted[start])/(i-start)
            rightslope = (sbd_arr_sorted[end-1]-sbd_arr_sorted[i])/(end-1-i)
            
            if leftslope > slope_thresh or rightslope > slope_thresh:
                continue
            if np.abs(leftslope - rightslope) < diff:
                diff = leftslope - rightslope
                radius = i
        if diff < slope_diff_thresh:
            np.append(candidates,radius)
        find_candidate_radius(sbd_arr_sorted,start,radius,candidates)
        find_candidate_radius(sbd_arr_sorted,radius+1,end,candidates)
    
    candidate = np.empty((0),np.int32)
    candidates = find_candidate_radius(sbd_arr_sorted,start,end,candidate)
    print(candidates)
    if candidates is not None:
        radius_candidates = np.max(sbd_arr_sorted[candidates])
        return radius_candidates
    else:
        raise ValueError('There is no qualified density raidus.')
    
