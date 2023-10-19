# An extension of some sklearn ensembles for if-then rule extraction.
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
import numpy as np  # type: ignore
from sklearn.utils.validation import check_is_fitted  # type: ignore
import sklearn.ensemble

from .tree import extract_rules
from .utils import clamp_transform


class RandomForestRegressor(sklearn.ensemble.RandomForestRegressor):
    @property
    def rules_(self):
        check_is_fitted(self)
        rules = []
        for estimator in self.estimators_:
            for rule in extract_rules(estimator):
                # Note that we deduplicate here!
                if rule not in rules:
                    rules.append(rule)
        return rules

    def bounds_(self, X_min, X_max, transformer_X=None):
        rules = self.rules_
        lowers = np.array([rule["l"] for rule in rules])
        uppers = np.array([rule["u"] for rule in rules])

        return clamp_transform(lowers, uppers, X_min, X_max, transformer_X)
