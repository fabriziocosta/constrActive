#!/usr/bin/env python
"""Provides Pareto optimization of graphs."""


import random
import numpy as np
from toolz.itertoolz import iterate, last
from toolz.curried import compose, pipe, map, concat
from itertools import islice

from graphlearn.lsgg import lsgg

from eden.util import timeit
from sklearn.neighbors import NearestNeighbors
from eden.graph import Vectorizer
import scipy as sp
from sklearn.metrics.pairwise import euclidean_distances
import multiprocessing
from eden import apply_async


from pareto_funcs import get_pareto_set
import constrActive.multi_objective_cost_estimator as mobj

import logging
logger = logging.getLogger(__name__)


def _manage_int_or_float(input_val, ref_val):
    assert (ref_val > 0), 'Error: ref val not >0'
    out_val = None
    if isinstance(input_val, int):
        out_val = min(input_val, ref_val)
    elif isinstance(input_val, float):
        msg = 'val=%.3f should be >0 and <=1'
        assert(0 < input_val <= 1), msg
        out_val = int(input_val * float(ref_val))
    else:
        raise Exception('Error on val type')
    out_val = max(out_val, 2)
    return out_val


class ParetoGraphOptimizer(object):
    """ParetoGraphOptimizer."""

    def __init__(
            self,
            grammar=None,
            multiobj_est=None,
            expand_max_n_neighbors=None,
            n_iter=19,
            expand_max_frontier=20,
            max_size_frontier=30,
            adapt_grammar_n_iter=5,
            random_state=1):
        """init."""
        self.grammar = grammar
        self.adapt_grammar_n_iter = adapt_grammar_n_iter
        self.multiobj_est = multiobj_est
        self.expand_max_n_neighbors = expand_max_n_neighbors
        self.n_iter = n_iter
        self.expand_max_frontier = expand_max_frontier
        self.max_size_frontier = max_size_frontier
        self.pareto_set = dict()
        self.knn_ref_dist = None

        self.curr_iter = 0
        self.prev_costs = None
        random.seed(random_state)
        self._expand_neighbors = compose(list,
                                         concat,
                                         map(self._get_neighbors))

    def set_objectives(self, multiobj_est):
        """set_objectives."""
        self.multiobj_est = multiobj_est

    def _mark_non_visited(self, graphs):
        for g in graphs:
            g.graph['visited'] = False

    @timeit
    def optimize(self, graphs):
        """Optimize iteratively."""
        assert(self.grammar.is_fit())
        assert(self.multiobj_est.is_fit())
        # init
        costs = self.multiobj_est.decision_function(graphs)
        seed_graphs = get_pareto_set(graphs, costs)
        self.pareto_set = seed_graphs
        self._mark_non_visited(self.pareto_set)

        # main cycle
        try:
            last(islice(
                iterate(self._update_pareto_set, seed_graphs),
                self.n_iter))
        except Exception as e:
            msg = 'Terminated at iteration:%d because %s' % (self.curr_iter, e)
            logger.debug(msg)
        finally:
            return self.pareto_set

    def _get_neighbors(self, graph):
        n = self.expand_max_n_neighbors
        if n is None:
            return self.grammar.neighbors(graph)
        else:
            return self.grammar.neighbors_sample(graph, n_neighbors=n)

    def _update_grammar_policy(self):
        if self.curr_iter > 0 and \
            self.adapt_grammar_n_iter is not None and \
                self.curr_iter % self.adapt_grammar_n_iter == 0:
            min_count = self.grammar.get_min_count()
            min_count = min_count + 1
            self.grammar.set_min_count(min_count)
            self.grammar.reset_productions()
            self.grammar.fit(self.pareto_set)
            self._mark_non_visited(self.pareto_set)
            logger.debug(self.grammar)

    def _update_pareto_set_policy(self, neighbor_graphs):
        graphs = self.pareto_set + neighbor_graphs
        costs = self.multiobj_est.decision_function(graphs)
        self.pareto_set = get_pareto_set(graphs, costs)
        if self.max_size_frontier is not None:
            # reduce Pareto set size by taking best ranking subset
            m = self.max_size_frontier
            n = len(self.pareto_set)
            size = _manage_int_or_float(m, n)
            self.pareto_set = self.multiobj_est.select(self.pareto_set, size)
        return costs

    def _update_pareto_set_expansion_policy(self):
        size = _manage_int_or_float(self.expand_max_frontier,
                                    len(self.pareto_set))
        # permute the elements in the frontier
        ids = np.arange(len(self.pareto_set))
        np.random.shuffle(ids)
        ids = list(ids)
        # select non visited elements
        is_visited = lambda g: g.graph.get('visited', False)
        non_visited_ids = [id for id in ids
                           if not is_visited(self.pareto_set[id])]
        if len(non_visited_ids) == 0:
            raise Exception('No non visited elements in frontier, stopping')
        not_yet_visited_graphs = []
        for id in non_visited_ids[:size]:
            self.pareto_set[id].graph['visited'] = True
            not_yet_visited_graphs.append(self.pareto_set[id])
        return not_yet_visited_graphs

    def _update_pareto_set(self, seed_graphs):
        """_update_pareto_set."""
        self._update_grammar_policy()
        neighbor_graphs = self._expand_neighbors(seed_graphs)
        costs = self._update_pareto_set_policy(neighbor_graphs)
        new_seeds = self._update_pareto_set_expansion_policy()
        self._log_update_pareto_set(
            costs, self.pareto_set, neighbor_graphs, new_seeds)
        self.curr_iter += 1
        return new_seeds

    def _log_update_pareto_set(self,
                               costs,
                               pareto_set,
                               neighbor_graphs,
                               new_seed_graphs):
        min_cost0 = min(costs[:, 0])
        par_size = len(pareto_set)
        med_cost0 = np.percentile(costs[:, 0], 50)
        txt = 'iter: %3d \t' % self.curr_iter
        txt += 'current min obj-0: %7.2f \t' % min_cost0
        txt += 'median obj-0: %7.2f \t' % med_cost0
        txt += 'added n neighbors: %4d \t' % len(neighbor_graphs)
        txt += 'obtained pareto set of size: %4d \t' % par_size
        txt += 'next round seeds: %4d ' % len(new_seed_graphs)
        logger.debug(txt)

# -----------------------------------------------------------------------------


class LocalLandmarksDistanceOptimizer(object):
    """LocalLandmarksDistanceOptimizer."""

    def __init__(
            self,
            r=3,
            d=3,
            context_size=2,
            min_count=1,
            expand_max_n_neighbors=None,
            max_size_frontier=None,
            n_iter=2,
            expand_max_frontier=1000,
            output_k_best=None,
            adapt_grammar_n_iter=None):
        """init."""
        self.adapt_grammar_n_iter = adapt_grammar_n_iter
        self.expand_max_n_neighbors = expand_max_n_neighbors
        self.n_iter = n_iter
        self.expand_max_frontier = expand_max_frontier
        self.max_size_frontier = max_size_frontier
        self.output_k_best = output_k_best
        self.grammar = lsgg()
        self.grammar.set_core_size([0, 1, 2, 3])
        self.grammar.set_context(context_size)
        self.grammar.set_min_count(min_count)
        self.multiobj_est = mobj.DistRankSizeCostEstimator(r=r, d=d)

    def fit(self):
        """fit."""
        pass

    def optimize(
            self,
            reference_graphs,
            desired_distances,
            ranked_graphs):
        """optimize."""
        self.grammar.fit(ranked_graphs)
        logger.debug(self.grammar)

        # fit objectives
        self.multiobj_est.fit(desired_distances,
                              reference_graphs,
                              ranked_graphs)

        # setup and run optimizer
        pgo = ParetoGraphOptimizer(
            grammar=self.grammar,
            multiobj_est=self.multiobj_est,
            expand_max_n_neighbors=self.expand_max_n_neighbors,
            expand_max_frontier=self.expand_max_frontier,
            max_size_frontier=self.max_size_frontier,
            n_iter=self.n_iter,
            adapt_grammar_n_iter=self.adapt_grammar_n_iter)
        graphs = pgo.optimize(reference_graphs + ranked_graphs)

        if self.output_k_best is None:
            return graphs
        else:
            # output a selection of the Pareto set
            return self.multiobj_est.select(
                graphs,
                k_best=self.output_k_best,
                objective=0)

# -----------------------------------------------------------------------------


class LandmarksDistanceOptimizer(object):
    """LandmarksDistanceOptimizer."""

    def __init__(
            self,
            r=3,
            d=3,
            context_size=2,
            min_count=1,
            expand_max_n_neighbors=None,
            n_iter=20,
            expand_max_frontier=1,
            output_k_best=5,
            improve=True):
        """init."""
        self.r = r
        self.d = d
        self.context_size = context_size
        self.expand_max_n_neighbors = expand_max_n_neighbors
        self.n_iter = n_iter
        self.expand_max_frontier = expand_max_frontier
        self.output_k_best = output_k_best
        self.improve = improve
        self.grammar = lsgg()
        self.grammar.set_core_size([0, 1, 2, 3])
        self.grammar.set_context(context_size)
        self.grammar.set_min_count(min_count)

    def fit(self, pos_graphs, neg_graphs):
        """fit."""
        self.multiobj_est = mobj.DistBiasSizeCostEstimator(
            pos_graphs,
            neg_graphs,
            r=self.r,
            d=self.d,
            improve=self.improve)

    def optimize(
            self,
            reference_graphs,
            desired_distances,
            loc_graphs):
        """optimize."""
        self.grammar.fit(loc_graphs)
        logger.debug(self.grammar)

        # fit objectives
        self.multiobj_est.fit(desired_distances, reference_graphs)

        # setup and run optimizer
        pgo = ParetoGraphOptimizer(
            grammar=self.grammar,
            multiobj_est=self.multiobj_est,
            expand_max_n_neighbors=self.expand_max_n_neighbors,
            expand_max_frontier=self.expand_max_frontier,
            n_iter=self.n_iter)
        graphs = pgo.optimize(reference_graphs)

        # output a selection of the Pareto set
        graphs = self.multiobj_est.select(graphs,
                                          k_best=self.output_k_best,
                                          objective=0)
        return graphs


# -----------------------------------------------------------------------------


class NearestNeighborsMeanOptimizer(object):
    """NearestNeighborsMeanOptimizer."""

    def __init__(
            self,
            min_count=2,
            context_size=2,
            expand_max_n_neighbors=None,
            r=3,
            d=3,
            n_landmarks=5,
            n_neighbors=100,
            n_iter=20,
            expand_max_frontier=1,
            output_k_best=5,
            improve=True):
        """init."""
        self.n_landmarks = n_landmarks
        self.n_neighbors = n_neighbors
        self.output_k_best = output_k_best
        self.nn_estimator = NearestNeighbors(n_neighbors=n_neighbors)
        self.non_norm_vec = Vectorizer(
            r=r,
            d=d,
            normalization=False,
            inner_normalization=False)
        self.vec = Vectorizer(
            r=r,
            d=d,
            normalization=True,
            inner_normalization=True)
        self.dist_opt = LandmarksDistanceOptimizer(
            r=r,
            d=d,
            min_count=min_count,
            context_size=context_size,
            expand_max_n_neighbors=expand_max_n_neighbors,
            n_iter=n_iter,
            expand_max_frontier=expand_max_frontier,
            output_k_best=output_k_best,
            improve=improve)
        self.sim_est = mobj.VarSimVolCostEstimator(improve=improve)

    def fit(self, pos_graphs, neg_graphs):
        """fit."""
        self.all_graphs = pos_graphs + neg_graphs
        self.all_vecs = self.vec.transform(self.all_graphs)
        self.nn_estimator.fit(self.all_vecs)
        self.dist_opt.fit(pos_graphs, neg_graphs)
        self.sim_est.fit(pos_graphs, neg_graphs)

    def optimize(self, graphs):
        """optimize."""
        seed_graphs = self.select(graphs, max_num=self.output_k_best)

        # run optimization in parallel
        pareto_graphs_list = self._optimize_parallel(seed_graphs)
        self._log_result(pareto_graphs_list)

        # join all pareto sets
        pareto_set_graphs = pipe(pareto_graphs_list, concat, list)

        # pareto filter using similarity of the solutions
        sel_graphs = self.select(pareto_set_graphs, max_num=self.output_k_best)
        logger.debug('#constructed graphs:%5d' % (len(sel_graphs)))
        return sel_graphs

    def select(self, graphs, max_num=30):
        """select."""
        costs = self.sim_est.decision_function(graphs)
        pareto_graphs = get_pareto_set(graphs, costs)
        select_graphs = self.sim_est.select(pareto_graphs, k_best=max_num)
        i, p, s = len(graphs), len(pareto_graphs), len(select_graphs)
        logger.debug('initial:%d  pareto:%d  selected:%d' % (i, p, s))
        return select_graphs

    def _log_result(self, pareto_graphs_list):
        tot_size = sum(len(graphs) for graphs in pareto_graphs_list)
        msg = 'pareto set sizes [%d]: ' % tot_size
        for graphs in pareto_graphs_list:
            msg += '[%d]' % len(graphs)
        logger.debug(msg)

    def _optimize_parallel(self, reference_graphs):
        """optimize_parallel."""
        pool = multiprocessing.Pool()
        res = [apply_async(
            pool, self._optimize, args=(reference_graph,))
            for reference_graph in reference_graphs]
        pareto_set_graphs_list = [p.get() for p in res]
        pool.close()
        pool.join()
        return pareto_set_graphs_list

    def _optimize(self, reference_graph):
        """optimize_single."""
        constraints = self._get_constraints(reference_graph)
        graphs = self.dist_opt.optimize(*constraints)
        return graphs

    def _get_constraints(self, reference_graph):
        reference_vec = self.non_norm_vec.transform([reference_graph])
        # find neighbors
        neighbors = self.nn_estimator.kneighbors(
            reference_vec,
            return_distance=False)
        neighbors = neighbors[0]
        # compute center of mass
        landmarks = neighbors[:self.n_landmarks]
        loc_graphs = [self.all_graphs[i] for i in neighbors]
        reference_graphs = [self.all_graphs[i] for i in landmarks]
        reference_vecs = self.all_vecs[landmarks]
        avg_reference_vec = sp.sparse.csr_matrix.mean(reference_vecs, axis=0)

        reference_vecs = self.non_norm_vec.transform(reference_graphs)
        # compute desired distances
        desired_distances = euclidean_distances(
            avg_reference_vec,
            reference_vecs)
        desired_distances = desired_distances[0]
        return reference_graphs, desired_distances, loc_graphs
