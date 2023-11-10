# An extension of suprb for if-then rule extraction.
#
# Copyright (C) 2023 David Pätzel
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np  # type: ignore
import suprb  # type: ignore
from suprb import rule
import suprb.utils
from suprb.optimizer import solution
from suprb.optimizer import rule
from suprb.optimizer.rule.mutation import HalfnormIncrease
from suprb.rule.initialization import MeanInit
from suprb.rule.fitness import VolumeWu
from sklearn.utils.validation import check_is_fitted  # type: ignore
from sklearn.utils import check_random_state

from .utils import clamp_transform


class SupRB(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        n_iter=32,
        n_initial_rules=0,
        n_rules=4,
        random_state=None,
        verbose=10,
        rd_mutation_sigma=0.1,
        rd_delay=10,
        rd_init_fitness_alpha=0.01,
        # sc_selection=("RouletteWheel", {}),
        sc_selection=("Tournament", {"k": 5}),
        sc_crossover=("NPoint", {"n": 2}),
        # sc_crossover=("Uniform", {}),
        sc_mutation_rate=0.05,
    ):
        self.n_iter = n_iter
        self.n_initial_rules = n_initial_rules
        self.n_rules = n_rules
        self.random_state = random_state
        self.verbose = verbose
        self.rd_mutation_sigma = rd_mutation_sigma
        self.rd_delay = rd_delay
        self.rd_init_fitness_alpha = rd_init_fitness_alpha
        self.sc_selection = sc_selection
        self.sc_crossover = sc_crossover
        self.sc_mutation_rate = sc_mutation_rate

    def fit(self, X, y):
        rd = rule.es.ES1xLambda(
            operator="&",  # early stopping
            # n_iter=12, # default seems to be 10000?
            delay=self.rd_delay,
            init=MeanInit(fitness=VolumeWu(alpha=self.rd_init_fitness_alpha)),
            mutation=HalfnormIncrease(sigma=self.rd_mutation_sigma),
        )

        sc = solution.ga.GeneticAlgorithm(
            # n_iter=, # default is 32
            # population_size # default is 32
            # elitist_ratio # default is 0.17
            # mutation=ga.mutation.BitFlips(), # default is BitFlips (actually
            # the only one implemented as of 2023-10-24)
            crossover=getattr(solution.ga.crossover, self.sc_crossover[0])(
                **self.sc_crossover[1]
            ),
            selection=getattr(solution.ga.selection, self.sc_selection[0])(
                **self.sc_selection[1]
            ),
        )

        self.suprb_ = suprb.SupRB(
            rule_generation=rd,
            solution_composition=sc,
            n_iter=self.n_iter,
            n_initial_rules=self.n_initial_rules,
            n_rules=self.n_rules,
            random_state=self.random_state,
            verbose=self.verbose,
            # This seems to be important esp. for tasks with many training data points.
            n_jobs=1,
        )

        self.suprb_.fit(X, y)

        return self

    def predict(self, X):
        return self.suprb_.predict(X)

    @property
    def rules_(self):
        return self.suprb_.elitist_.subpopulation

    def bounds_(self, X_min, X_max, transformer_X=None):
        rules = self.rules_
        lowers = [rule.match.bounds[:, 0] for rule in self.rules_]
        uppers = [rule.match.bounds[:, 1] for rule in self.rules_]

        return clamp_transform(lowers, uppers, X_min, X_max, transformer_X)
