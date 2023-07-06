# An extension of sklearn decision trees for if-then rule extraction.
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
import sklearn.tree  # type: ignore
from sklearn.tree import _tree  # type: ignore
from sklearn.utils.validation import check_is_fitted  # type: ignore


class DecisionTreeRegressor(sklearn.tree.DecisionTreeRegressor):
    @property
    def rules_(self):

        check_is_fitted(self)

        return extract_rules(self)


# TODO Consider to make this/check whether it already is general (i.e. include
# classification as well)
def extract_rules(estimator):
    if hasattr(estimator, "feature_names_in_"):
        feature_names = estimator.feature_names_in_
    else:
        feature_names = ["feature_" + str(i) for i in range(estimator.n_features_)]

    rules = []

    leq_threshold = "leq_threshold"
    gt_threshold = "gt_threshold"

    def is_leaf(node):
        # TODO Maybe check children_right and children_left for
        # being -1 as well for consistency
        return estimator.tree_.feature[node] == _tree.TREE_UNDEFINED

    def recurse(node, features, rels, thresholds, feature_names):
        if not is_leaf(node):
            features_ = features + [feature_names[estimator.tree_.feature[node]]]
            thresholds_ = thresholds + [estimator.tree_.threshold[node]]

            recurse(
                estimator.tree_.children_left[node],
                features_,
                # If we go down the left subtree, then the data is less
                # than or equal to the threshold.
                rels + [leq_threshold],
                thresholds_,
                feature_names,
            )
            recurse(
                estimator.tree_.children_right[node],
                features_,
                # If we go down the left subtree, then the data is
                # greater than the threshold.
                rels + [gt_threshold],
                thresholds_,
                feature_names,
            )
        else:
            lowers, uppers = [], []
            features = np.array(features)
            # Says whether data <= threshold or data > threshold.
            rels = np.array(rels)
            thresholds = np.array(thresholds)

            for feature_name in feature_names:

                # Lower bound (i.e. `data > threshold`).
                #
                # If no threshold below the data, set lower bound to
                # negative infinity.
                if not np.any(rels == gt_threshold):
                    lowers.append(-np.inf)
                else:
                    # Otherwise, lower bound is highest of the lower thresholds.
                    lowers.append(np.max(thresholds[rels == gt_threshold]))

                # Upper bound (i.e. `data <= threshold`).
                #
                # If no threshold above the data, set to upper bound to
                # positive infinity.
                if not np.any(rels == leq_threshold):
                    uppers.append(np.inf)
                else:
                    # Otherwise, upper bound is lowest of the upper thresholds.
                    uppers.append(np.min(thresholds[rels == leq_threshold]))

            rules.append(
                dict(
                    l=np.array(lowers),
                    u=np.array(uppers),
                    pred=estimator.tree_.value[node],
                    exp=estimator.tree_.n_node_samples[node],
                )
            )

    recurse(0, [], [], [], feature_names)

    return rules
