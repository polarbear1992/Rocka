# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 15:26:44 2019

@author: icebearwang
"""

from sklearn.cluster import dbscan_

class Rocka(object):
    """
    
    """
    def __init__(self,sbd_matrix,density_radius,minPts,metric_rocka='precomputed',
            metric_params_rocka=None,algorithm_rocka='auto',leaf_size_rocka=30,
            p_rocka=2,sample_weight_rocka=None,n_jobs_rocka=None):
        self.sbd_matrix = sbd_matrix
        self.density_radius = density_radius
        self.minPts = minPts
        self.metric_rocka = metric_rocka
        self.metric_params_rocka = metric_params_rocka
        self.algorithm_rocka = algorithm_rocka
        self.leaf_size_rocka = leaf_size_rocka
        self.p_rocka = p_rocka
        self.sample_weight_rocka = sample_weight_rocka
        self.n_jobs_rocka = n_jobs_rocka
        
    def fit(self,sbd_matrix,density_radius,minPts,metric_rocka='precomputed',
            metric_params_rocka=None,algorithm_rocka='auto',leaf_size_rocka
            =30,p_rocka=2,sample_weight_rocka=None,n_jobs_rocka=None):
        """
        Perform DBSCAN clustering from vector array or distance matrix.

        Read more in the :ref:`User Guide <dbscan>`.
    
        Parameters
        ----------
        sbd_matrix : np.ndarray or sparse (CSR) matrix of shape (n_samples, n_features), 
            or \array of shape (n_samples, n_samples)
            A feature array, or array of distances between samples if
            ``metric='precomputed'``.
    
        density_radius : np.float32, 
            The maximum distance between two samples for one to be considered
            as in the neighborhood of the other. This is not a maximum bound
            on the distances of points within a cluster. This is the most
            important DBSCAN parameter to choose appropriately for your data set
            and distance function.
    
        minPts : np.int32, 
            The number of samples (or total weight) in a neighborhood for a point
            to be considered as a core point. This includes the point itself.
    
        metric_rocka : string, or callable
            The metric to use when calculating distance between instances in a
            feature array. If metric is a string or callable, it must be one of
            the options allowed by :func:`sklearn.metrics.pairwise_distances` for
            its metric parameter.
            If metric is "precomputed", X is assumed to be a distance matrix and
            must be square. X may be a sparse matrix, in which case only "nonzero"
            elements may be considered neighbors for DBSCAN.
    
        metric_params_rocka : dict, optional
            Additional keyword arguments for the metric function.
    
            .. versionadded:: 0.19
    
        algorithm_rocka : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
            The algorithm to be used by the NearestNeighbors module
            to compute pointwise distances and find nearest neighbors.
            See NearestNeighbors module documentation for details.
    
        leaf_size_rocka : np.int32, optional (default = 30)
            Leaf size passed to BallTree or cKDTree. This can affect the speed
            of the construction and query, as well as the memory required
            to store the tree. The optimal value depends
            on the nature of the problem.
    
        p_rocka : np.float32, optional
            The power of the Minkowski metric to be used to calculate distance
            between points.
    
        sample_weight_rocka : np.ndarray, shape (n_samples,), optional
            Weight of each sample, such that a sample with a weight of at least
            ``min_samples`` is by itself a core sample; a sample with negative
            weight may inhibit its eps-neighbor from being core.
            Note that weights are absolute, and default to 1.
    
        n_jobs_rocka : bp.int32 or None, optional (default=None)
            The number of parallel jobs to run for neighbors search.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.
    
        Attributes
        ----------
        core_sample_indices_ : array, shape = [n_core_samples]
            Indices of core samples.
    
        components_ : array, shape = [n_core_samples, n_features]
            Copy of each core sample found by training.
    
        labels_ : array, shape = [n_samples]
            Cluster labels for each point in the dataset given to fit().
            Noisy samples are given the label -1.
    
        See also
        --------
        DBSCAN
            An estimator interface for this clustering algorithm.
        OPTICS
            A similar estimator interface clustering at multiple values of eps. Our
            implementation is optimized for memory usage.
        """
        model = dbscan_.DBSCAN(density_radius,minPts,metric=metric_rocka,
            metric_params=metric_params_rocka,algorithm=algorithm_rocka,
            leaf_size=leaf_size_rocka,p=p_rocka,sample_weight=sample_weight_rocka,
            n_jobs=n_jobs_rocka).fit(sbd_matrix)
        
        return model
    
        
        
        