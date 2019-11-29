# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 10:04:57 2019

@author: PB
"""
from preprocessing_rocka import standardize_obj
import numpy as np

__all__ = ['smoothing_extreme_values','extract_baseline']

def smoothing_extreme_values(values):
    """
    In general,the ratio of anomaly points in a time series is less than 5%[1].
    As such,simply remove the top5% data which deviate the most from the mean 
    value,and use linear interpolation to fill them.
    
    Args:
        values(np.ndarray) is a time series which has been preprosessed by linear 
        interpolation and standardization(to have zero mean and unit variance)
    
    Returns:
        np.ndarray: The smoothed `values`
    """
    
    values = np.asarray(values, np.float32)
    if len(values.shape) != 1:
        raise ValueError('`values` must be a 1-D array')
#    if (values.mean() != 0) or (values.std() != 1):
#        raise ValueError('`values` must be standardized to have zero mean and unit variance')
    
    #get the deviation of each point from zero mean
    values_deviation = np.abs(values)
    
    #the abnormal portion
    abnormal_portion = 0.05
    
    #replace the abnormal points with linear interpolation
    abnormal_max = np.max(values_deviation)
    abnormal_index = np.argwhere(values_deviation >= abnormal_max * (1-abnormal_portion))
    abnormal = abnormal_index.reshape(len(abnormal_index))
    normal_index = np.argwhere(values_deviation < abnormal_max * (1-abnormal_portion))
    normal = normal_index.reshape(len(normal_index))
    normal_values = values[normal]
    abnormal_values = np.interp(abnormal,normal,normal_values)
    values[abnormal] = abnormal_values
    
    return values
    
def extract_baseline(values,w):
    """
    A simple but effective method for removing noises if to apply moving 
    average with a small sliding window(`w`) on the KPI(`values`),separating 
    its curve into two parts:baseline and residuals.
    For a KPI,T,with a sliding window of length of `w`,stride = 1,for each 
    point x(t),the corresponding point on the baseline,denoted as x(t)*,is the 
    mean of vector (x(t-w+1),...,x(t)).
    Then the diffrence between x(t) and x(t)* is called a residuals.
    
    Args:
        values(np.ndarray): time series after preprocessing and smoothed
        
    Returns:
        tuple(np.ndarray,np.float32,np.float32):
            np.ndarray: the baseline of rawdata;
            np.float32: the mean of input values after moving average;
            np.float32: the std of input values after moving average.
        np.ndarray:the residuals between rawdata between baseline
        
        
    """
    #moving average to get the baseline
    baseline = np.convolve(values,np.ones((w,))/w,mode='valid')
    #get the residuals,the difference between raw series and baseline
    residuals = values[w-1:] - baseline
    
    return standardize_obj(baseline),residuals
    
    
