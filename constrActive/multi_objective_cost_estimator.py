#!/usr/bin/env python
"""Provides Pareto optimization of graphs."""


import numpy as np
from scipy.stats import rankdata

from eden.graph import Vectorizer
from eden.util import timeit

from cost_estimator import InstancesDistanceCostEstimator
from cost_estimator import RankBiasCostEstimator
from cost_estimator import SizeCostEstimator
from cost_estimator import ClassBiasCostEstimator
from cost_estimator import VarianceCostEstimator
from cost_estimator import SimilarityCostEstimator
from cost_estimator import VolumeCostEstimator

import logging
logger = logging.getLogger(__name__)


class MultiObjectiveCostEstimator(object):
    """MultiObjectiveCostEstimator."""

    def __init__(self):
        """Initialize."""
        pass

    def set_params(self, estimators):
        """set_params."""
        self.estimators = estimators

    def decision_function(self, graphs):
        """decision_function."""
        cost_vec = [estimator.decision_function(graphs)
                    for estimator in self.estimators]
        costs = np.hstack(cost_vec)
        return costs

    def is_fit(self):
        """is_fit."""
        return self.estimators is not None

    def select(self, graphs, k_best=10, objective=None):
        """select."""
        if k_best > len(graphs):
            return graphs
        costs = self.decision_function(graphs)
        ranks = [rankdata(costs[:, i], method='min')
                 for i in range(costs.shape[1])]
        ranks = np.vstack(ranks).T
        if objective is None:
            agg_ranks = np.sum(ranks, axis=1)
        else:
            agg_ranks = ranks[:, objective]
        ids = np.argsort(agg_ranks)
        k_best_graphs = [graphs[id] for id in ids[:k_best]]
        return k_best_graphs


# -----------------------------------------------------------------------------


class DistRankSizeCostEstimator(MultiObjectiveCostEstimator):
    """DistRankSizeCostEstimator."""

    def __init__(self, r=3, d=3):
        """Initialize."""
        self.vec = Vectorizer(
            r=r,
            d=d,
            normalization=False,
            inner_normalization=False)

    @timeit
    def fit(
            self,
            desired_distances,
            reference_graphs,
            ranked_graphs):
        """fit."""
        d_est = InstancesDistanceCostEstimator(self.vec)
        d_est.fit(desired_distances, reference_graphs)

        b_est = RankBiasCostEstimator(self.vec, improve=True)
        b_est.fit(ranked_graphs)

        s_est = SizeCostEstimator()
        s_est.fit(reference_graphs)

        self.estimators = [d_est, b_est, s_est]
        return self


# -----------------------------------------------------------------------------


class DistBiasSizeCostEstimator(MultiObjectiveCostEstimator):
    """DistBiasSizeCostEstimator."""

    def __init__(self, pos_graphs, neg_graphs, r=3, d=3, improve=True):
        """Initialize."""
        self.vec = Vectorizer(
            r=r,
            d=d,
            normalization=False,
            inner_normalization=False)
        self.b_est = ClassBiasCostEstimator(self.vec, improve=improve)
        self.b_est.fit(pos_graphs, neg_graphs)

    @timeit
    def fit(self, desired_distances, reference_graphs):
        """fit."""
        d_est = InstancesDistanceCostEstimator(self.vec)
        d_est.fit(desired_distances, reference_graphs)

        s_est = SizeCostEstimator()
        s_est.fit(reference_graphs)

        self.estimators = [d_est, self.b_est, s_est]
        return self

# -----------------------------------------------------------------------------


class VarSimVolCostEstimator(MultiObjectiveCostEstimator):
    """VarSimVolCostEstimator."""

    def __init__(self, r=3, d=3, improve=True):
        """Initialize."""
        self.improve = improve
        self.vec = Vectorizer(
            r=r,
            d=d,
            normalization=False,
            inner_normalization=False)

    @timeit
    def fit(self, pos_graphs, neg_graphs):
        """fit."""
        var_est = VarianceCostEstimator(self.vec, improve=self.improve)
        var_est.fit(pos_graphs, neg_graphs)

        s_est = SimilarityCostEstimator(self.vec)
        s_est.fit(pos_graphs + neg_graphs)

        v_est = VolumeCostEstimator(self.vec)
        v_est.fit(pos_graphs + neg_graphs)

        self.estimators = [var_est, s_est, v_est]
        return self
