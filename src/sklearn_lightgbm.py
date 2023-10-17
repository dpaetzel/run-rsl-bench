import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin  # type: ignore
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

import lightgbm as lgb


class LinearTreeRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
    ):
        pass

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        self.n_features_in_ = X.shape[1]

        # data_lgb = lgb.Dataset(X, label=y, params={"linear_tree": True})
        data_lgb = lgb.Dataset(X, label=y, params={})

        params = {
            "boosting_type": "gbdt",
            "objective": "regression",
            "linear_tree": True,
            # https://lightgbm.readthedocs.io/en/v3.3.2/Parameters.html#linear_tree
            # says that regression_l1 objective is not supported with linear
            # tree boosting
            # "metric": "mae",
            "metric": "mse",
            # "num_leaves": 30,
            # "learning_rate": 0.1,
            # "verbosity": -1,
            # Only build a single tree.
            "num_trees": 100,
        }

        self.booster_ = lgb.train(params, data_lgb)

        return self

    def predict(self, X, start_iteration=0, n_iterations=0):

        check_is_fitted(self)

        X = check_array(X)

        return self.booster_.predict(
            X, start_iteration=start_iteration, num_iteration=n_iterations
        )

    @property
    def rules_(self):
        check_is_fitted(self)

        tinfo = self.booster_.dump_model()["tree_info"]

        rule_sets = []
        for tree in tinfo:
            rule_sets.append(
                extract_rules(tree["tree_structure"], DX=self.n_features_in_)
            )

        return sum(rule_sets, [])


def extract_rules(tree, DX):
    rules = []

    leq_threshold = "<="
    gt_threshold = ">"

    def is_leaf(tree):
        return "leaf_index" in tree

    def recurse(tree, features, rels, thresholds):
        if not is_leaf(tree):
            if tree["decision_type"] != "<=":
                raise RuntimeError(f"Unexpected decision_type: {tree['decision_type']}")
            features_ = features + [tree["split_feature"]]
            thresholds_ = thresholds + [tree["threshold"]]

            recurse(tree["left_child"], features_, rels + [leq_threshold], thresholds_)
            recurse(tree["right_child"], features_, rels + [gt_threshold], thresholds_)
        else:
            lowers, uppers = [], []
            features = np.array(features)
            # Says whether data <= threshold or data > threshold.
            rels = np.array(rels)
            thresholds = np.array(thresholds)

            for idx_feature in range(DX):

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
                    pred={
                        k: tree[k] for k in ["leaf_value", "leaf_const", "leaf_coeff"]
                    },
                    exp=tree["leaf_count"],
                )
            )

    recurse(tree, [], [], [])

    return rules


# if False:
#     import lightgbm as lgb
#     import matplotlib.pyplot as plt

#     lgb.plot_tree(b, tree_index=0)
#     # lgb.plot_tree(b, tree_index=1)
#     plt.show()
