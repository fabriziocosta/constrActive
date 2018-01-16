#!/usr/bin/env python
"""Provides cost estimators."""


import numpy as np
import scipy as sp
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from eden.graph import Vectorizer
from eden.util import describe

import logging
logger = logging.getLogger(__name__)


class InstancesDistanceCostEstimator(object):
    """InstancesDistanceCostEstimator."""

    def __init__(self, vectorizer=Vectorizer()):
        """init."""
        self.desired_distances = None
        self.reference_vecs = None
        self.vectorizer = vectorizer

    def fit(self, desired_distances, reference_graphs):
        """fit."""
        self.desired_distances = desired_distances
        self.reference_vecs = self.vectorizer.transform(reference_graphs)
        return self

    def _avg_distance_diff(self, vector):
        distances = euclidean_distances(vector, self.reference_vecs)[0]
        d = self.desired_distances
        dist_diff = (distances - d)
        avg_dist_diff = np.mean(np.absolute(dist_diff))
        return avg_dist_diff

    def decision_function(self, graphs):
        """predict_distance."""
        x = self.vectorizer.transform(graphs)
        avg_distance_diff = np.array([self._avg_distance_diff(vec)
                                      for vec in x])
        avg_distance_diff = avg_distance_diff.reshape(-1, 1)
        return avg_distance_diff


class InstancesMultiDistanceCostEstimator(object):
    """InstancesMultiDistanceCostEstimator."""

    def __init__(self, vectorizer=Vectorizer()):
        """init."""
        self.desired_distances = None
        self.reference_vecs = None
        self.vectorizer = vectorizer

    def fit(self, desired_distances, reference_graphs):
        """fit."""
        self.desired_distances = desired_distances
        self.reference_vecs = self.vectorizer.transform(reference_graphs)
        return self

    def _multi_distance_diff(self, vector):
        distances = euclidean_distances(vector, self.reference_vecs)[0]
        d = self.desired_distances
        dist_diff = (distances - d)
        return np.absolute(dist_diff).reshape(1, -1)

    def decision_function(self, graphs):
        """predict_distance."""
        x = self.vectorizer.transform(graphs)
        multi_distance_diff = np.vstack([self._multi_distance_diff(vec)
                                         for vec in x])
        return multi_distance_diff


class ClassBiasCostEstimator(object):
    """ClassBiasCostEstimator."""

    def __init__(self, vectorizer, improve=True):
        """init."""
        self.vectorizer = vectorizer
        self.estimator = SGDClassifier(average=True,
                                       class_weight='balanced',
                                       shuffle=True,
                                       n_jobs=1)
        self.improve = improve

    def fit(self, pos_graphs, neg_graphs):
        """fit."""
        graphs = pos_graphs + neg_graphs
        y = [1] * len(pos_graphs) + [-1] * len(neg_graphs)
        x = self.vectorizer.transform(graphs)
        self.estimator = self.estimator.fit(x, y)
        return self

    def decision_function(self, graphs):
        """decision_function."""
        x = self.vectorizer.transform(graphs)
        scores = self.estimator.decision_function(x)
        if self.improve is False:
            scores = np.absolute(scores)
        else:
            scores = - scores
        scores = scores.reshape(-1, 1)
        return scores


class RankBiasCostEstimator(object):
    """RankBiasCostEstimator."""

    def __init__(self, vectorizer, improve=True):
        """init."""
        self.vectorizer = vectorizer
        self.estimator = SGDClassifier(average=True,
                                       class_weight='balanced',
                                       shuffle=True)
        self.improve = improve

    def fit(self, ranked_graphs):
        """fit."""
        x = self.vectorizer.transform(ranked_graphs)
        r, c = x.shape
        pos = []
        neg = []
        for i in range(r - 1):
            for j in range(i + 1, r):
                p = x[i] - x[j]
                n = - p
                pos.append(p)
                neg.append(n)
        y = np.array([1] * len(pos) + [-1] * len(neg))
        pos = sp.sparse.vstack(pos)
        neg = sp.sparse.vstack(neg)
        x_ranks = sp.sparse.vstack([pos, neg])
        logger.debug('fitting: %s' % describe(x_ranks))
        self.estimator = self.estimator.fit(x_ranks, y)
        return self

    def decision_function(self, graphs):
        """decision_function."""
        x = self.vectorizer.transform(graphs)
        scores = self.estimator.decision_function(x)
        if self.improve is False:
            scores = np.absolute(scores)
        else:
            scores = - scores
        scores = scores.reshape(-1, 1)
        return scores


class SizeCostEstimator(object):
    """SizeCostEstimator."""

    def __init__(self):
        """init."""
        pass

    def fit(self, graphs):
        """fit."""
        self.reference_size = np.percentile([len(g) for g in graphs], 50)
        return self

    def decision_function(self, graphs):
        """decision_function."""
        sizes = np.array([len(g) for g in graphs])
        size_diffs = np.absolute(sizes - self.reference_size)
        size_diffs = size_diffs.reshape(-1, 1)
        return size_diffs


class VolumeCostEstimator(object):
    """VolumeCostEstimator."""

    def __init__(self, vectorizer, k=5):
        """init."""
        self.vectorizer = vectorizer
        self.k = k
        self.nn_estimator = NearestNeighbors()

    def fit(self, graphs):
        """fit."""
        x = self.vectorizer.transform(graphs)
        self.nn_estimator = self.nn_estimator.fit(x)
        return self

    def decision_function(self, graphs):
        """decision_function."""
        x = self.vectorizer.transform(graphs)
        distances, neighbors = self.nn_estimator.kneighbors(x, self.k)
        vols = np.mean(distances, axis=1)
        vols = -vols.reshape(-1, 1)
        return vols


class SimilarityCostEstimator(object):
    """SimilarityCostEstimator."""

    def __init__(self, vectorizer):
        """init."""
        self.vectorizer = vectorizer

    def fit(self, graphs):
        """fit."""
        self.vecs = self.vectorizer.transform(graphs)
        return self

    def decision_function(self, graphs):
        """decision_function."""
        x = self.vectorizer.transform(graphs)
        similarity_matrix = cosine_similarity(x, self.vecs)
        sims = np.mean(similarity_matrix, axis=1)
        sims = -sims.reshape(-1, 1)
        return sims


class VarianceCostEstimator(object):
    """VarianceCostEstimator."""

    def __init__(self, vectorizer, improve=True, n_estimators=10):
        """init."""
        self.vectorizer = vectorizer
        self.improve = improve
        self.n_estimators = n_estimators

    def fit(self, pos_graphs, neg_graphs):
        """fit."""
        graphs = pos_graphs + neg_graphs
        y = [1] * len(pos_graphs) + [-1] * len(neg_graphs)
        x = self.vectorizer.transform(graphs)

        self.estimators = []
        for i in range(self.n_estimators):
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.5, random_state=i)
            sgd = SGDClassifier(average=True,
                                class_weight='balanced',
                                shuffle=True)
            self.estimators.append(sgd.fit(x_train, y_train))
        return self

    def decision_function(self, graphs):
        """decision_function."""
        x = self.vectorizer.transform(graphs)
        scores = [estimator.decision_function(x)
                  for estimator in self.estimators]
        scores = np.vstack(scores).T
        avg_scores = np.mean(scores, axis=1)
        std_scores = np.std(scores, axis=1)
        if self.improve is False:
            avg_scores = np.absolute(avg_scores)
        avg_scores = -avg_scores.reshape(-1, 1)
        std_scores = -std_scores.reshape(-1, 1)

        quant_high = np.percentile(scores, 75, axis=1)
        quant_low = np.percentile(scores, 25, axis=1)
        quant_high = -quant_high.reshape(-1, 1)
        quant_low = -quant_low.reshape(-1, 1)
        out = np.hstack([avg_scores, std_scores, quant_high, quant_low])
        return out
