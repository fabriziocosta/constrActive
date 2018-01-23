#!/usr/bin/env python
"""Provides graph constructor from neighbors."""

from multi_objective_optimizer import LocalLandmarksDistanceOptimizer


def construct_from_neighbors(
        landmark_graphs=None,
        desired_distances=None,
        ranked_graphs=None,
        r=2,
        d=5,
        min_count=1,
        context_size=2,
        expand_max_n_neighbors=None,
        max_size_frontier=None,
        n_iter=20,
        expand_max_frontier=1,
        output_k_best=None,
        adapt_grammar_n_iter=None):
    """construct_from_neighbors."""
    ld_opt = LocalLandmarksDistanceOptimizer(
        r=r,
        d=d,
        min_count=min_count,
        context_size=context_size,
        expand_max_n_neighbors=expand_max_n_neighbors,
        n_iter=n_iter + 1,
        expand_max_frontier=expand_max_frontier,
        max_size_frontier=max_size_frontier,
        output_k_best=output_k_best,
        adapt_grammar_n_iter=adapt_grammar_n_iter)
    graphs = ld_opt.optimize(
        landmark_graphs,
        desired_distances,
        ranked_graphs)
    return graphs
