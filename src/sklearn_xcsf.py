# An almost scikit-learn–compatible wrapper for Preen's xcsf library.
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
#
# Based on
# https://github.com/heidmic/suprb-experimentation/blob/xcsf_experiment/runs/examples/xcsf_final.py
# which was originally based on
# https://github.com/berbl-dev/berbl-exp/blob/main/src/experiments/xcsf.py .
import json
import warnings

import numpy as np
import toolz
import xcsf
from sklearn.base import BaseEstimator, RegressorMixin  # type: ignore
from sklearn.metrics import mean_absolute_error
from sklearn.utils import check_random_state  # type: ignore
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


wiggle_room = 0.05
X_min = -1.0 - wiggle_room
X_max = 1.0 + wiggle_room


def fromNone(val, default):
    return default if val is None else val


def set_xcs_params(xcs, params):
    xcs.OMP_NUM_THREADS = params["OMP_NUM_THREADS"]
    xcs.POP_INIT = params["POP_INIT"]
    xcs.POP_SIZE = params["POP_SIZE"]
    xcs.MAX_TRIALS = params["MAX_TRIALS"]
    xcs.PERF_TRIALS = params["PERF_TRIALS"]
    xcs.LOSS_FUNC = params["LOSS_FUNC"]
    xcs.HUBER_DELTA = params["HUBER_DELTA"]
    xcs.E0 = params["E0"]
    xcs.ALPHA = params["ALPHA"]
    xcs.NU = params["NU"]
    xcs.BETA = params["BETA"]
    xcs.DELTA = params["DELTA"]
    xcs.THETA_DEL = params["THETA_DEL"]
    xcs.INIT_FITNESS = params["INIT_FITNESS"]
    xcs.INIT_ERROR = params["INIT_ERROR"]
    xcs.M_PROBATION = params["M_PROBATION"]
    xcs.STATEFUL = params["STATEFUL"]
    xcs.SET_SUBSUMPTION = params["SET_SUBSUMPTION"]
    xcs.THETA_SUB = params["THETA_SUB"]
    xcs.COMPACTION = params["COMPACTION"]
    xcs.TELETRANSPORTATION = params["TELETRANSPORTATION"]
    xcs.GAMMA = params["GAMMA"]
    xcs.P_EXPLORE = params["P_EXPLORE"]
    xcs.EA_SELECT_TYPE = params["EA_SELECT_TYPE"]
    xcs.EA_SELECT_SIZE = params["EA_SELECT_SIZE"]
    xcs.THETA_EA = params["THETA_EA"]
    xcs.LAMBDA = params["LAMBDA"]
    xcs.P_CROSSOVER = params["P_CROSSOVER"]
    xcs.ERR_REDUC = params["ERR_REDUC"]
    xcs.FIT_REDUC = params["FIT_REDUC"]
    xcs.EA_SUBSUMPTION = params["EA_SUBSUMPTION"]
    xcs.EA_PRED_RESET = params["EA_PRED_RESET"]


def get_xcs_params(xcs):
    return {
        "OMP_NUM_THREADS": xcs.OMP_NUM_THREADS,
        "POP_INIT": xcs.POP_INIT,
        "POP_SIZE": xcs.POP_SIZE,
        "MAX_TRIALS": xcs.MAX_TRIALS,
        "PERF_TRIALS": xcs.PERF_TRIALS,
        "LOSS_FUNC": xcs.LOSS_FUNC,
        "HUBER_DELTA": xcs.HUBER_DELTA,
        "E0": xcs.E0,
        "ALPHA": xcs.ALPHA,
        "NU": xcs.NU,
        "BETA": xcs.BETA,
        "DELTA": xcs.DELTA,
        "THETA_DEL": xcs.THETA_DEL,
        "INIT_FITNESS": xcs.INIT_FITNESS,
        "INIT_ERROR": xcs.INIT_ERROR,
        "M_PROBATION": xcs.M_PROBATION,
        "STATEFUL": xcs.STATEFUL,
        "SET_SUBSUMPTION": xcs.SET_SUBSUMPTION,
        "THETA_SUB": xcs.THETA_SUB,
        "COMPACTION": xcs.COMPACTION,
        "TELETRANSPORTATION": xcs.TELETRANSPORTATION,
        "GAMMA": xcs.GAMMA,
        "P_EXPLORE": xcs.P_EXPLORE,
        "EA_SELECT_TYPE": xcs.EA_SELECT_TYPE,
        "EA_SELECT_SIZE": xcs.EA_SELECT_SIZE,
        "THETA_EA": xcs.THETA_EA,
        "LAMBDA": xcs.LAMBDA,
        "P_CROSSOVER": xcs.P_CROSSOVER,
        "ERR_REDUC": xcs.ERR_REDUC,
        "FIT_REDUC": xcs.FIT_REDUC,
        "EA_SUBSUMPTION": xcs.EA_SUBSUMPTION,
        "EA_PRED_RESET": xcs.EA_PRED_RESET,
    }


def default_xcs_params(DX):
    xcs = xcsf.XCS(DX, 1, 1)
    return get_xcs_params(xcs)


class XCSF(BaseEstimator, RegressorMixin):
    """
    Almost a correct sklearn wrapper for ``xcsf.XCS``. For example, it can't yet
    be pickled and some parameters are missing
    """

    def __init__(
        self,
        random_state=None,
        # Note that we intentionally keep the same ordering here as in the wiki
        # at
        # https://github.com/rpreen/xcsf/wiki/Python-Library-Usage#initialising-general-parameters
        n_threads=None,
        # POP_INIT
        n_pop_size=None,
        n_iter=None,
        # PERF_TRIALS
        # LOSS_FUNC
        # HUBER_DELTA
        epsilon0=None,
        alpha=None,
        nu=None,
        beta=None,
        ea_delta=None,
        ea_theta_del=None,
        # INIT_FITNESS
        # INIT_ERROR
        # M_PROBATION
        # STATEFUL
        ea_set_subsumption=None,
        ea_theta_sub=None,
        ea_compaction=None,
        # TELETRANSPORTATION
        # GAMMA
        # P_EXPLORE
        ea_select_type=None,
        ea_select_size=None,
        ea_theta_ea=None,
        ea_lambda=None,
        ea_p_crossover=None,
        ea_err_reduc=None,
        ea_fit_reduc=None,
        ea_subsumption=None,
        # EA_PRED_RESET
        condition="csr",
        spread_min=None,
        check_convergence=True,
        tol=None,
        n_iter_no_change=1000,
    ):
        """
        Parameters
        ----------
        spread_min : float > 0 or None
        check_convergence : bool
            Whether to use the `tol` `n_iter_no_change` mechanism to check for
            convergence based on model scores on the training data.
        tol : float > 0 or None
            Tolerance for the optimization. When the loss (MAE on training data)
            is not improving by at least `tol` for `n_iter_no_change`
            consecutive iterations convergence is considered to be reached and
            training stops. If `None`, then set `tol` to `epsilon0`.
        n_iter_no_change : int
            Maximum number of epochs to not meet `tol` improvement.
        """
        self.random_state = random_state

        self.n_threads = n_threads
        # POP_INIT
        self.n_pop_size = n_pop_size
        self.n_iter = n_iter
        # PERF_TRIALS
        # LOSS_FUNC
        # HUBER_DELTA
        self.epsilon0 = epsilon0
        self.alpha = alpha
        self.nu = nu
        self.beta = beta
        self.ea_delta = ea_delta
        self.ea_theta_del = ea_theta_del
        # INIT_FITNESS
        # INIT_ERROR
        # M_PROBATION
        # STATEFUL
        self.ea_set_subsumption = ea_set_subsumption
        self.ea_theta_sub = ea_theta_sub
        self.ea_compaction = ea_compaction
        # TELETRANSPORTATION
        # GAMMA
        # P_EXPLORE
        self.ea_select_type = ea_select_type
        self.ea_select_size = ea_select_size
        self.ea_theta_ea = ea_theta_ea
        self.ea_lambda = ea_lambda
        self.ea_p_crossover = ea_p_crossover
        self.ea_err_reduc = ea_err_reduc
        self.ea_fit_reduc = ea_fit_reduc
        self.ea_subsumption = ea_subsumption
        # EA_PRED_RESET

        self.condition = condition
        self.spread_min = spread_min

        self.check_convergence = check_convergence
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change

    def _parse_params(self, DX):
        """
        Given an input space dimension, parse the parameters, replacing `None`s
        with the default parameter given by the XCSF library.
        """
        params_default = default_xcs_params(DX)

        # Extract defaults for each of input parameters.
        self.n_threads_ = fromNone(self.n_threads, params_default["OMP_NUM_THREADS"])
        # POP_INIT
        self.n_pop_size_ = fromNone(self.n_pop_size, params_default["POP_SIZE"])
        self.n_iter_ = fromNone(self.n_iter, params_default["MAX_TRIALS"])
        # PERF_TRIALS
        # LOSS_FUNC
        # HUBER_DELTA
        self.epsilon0_ = fromNone(self.epsilon0, params_default["E0"])
        self.alpha_ = fromNone(self.alpha, params_default["ALPHA"])
        self.nu_ = fromNone(self.nu, params_default["NU"])
        self.beta_ = fromNone(self.beta, params_default["BETA"])
        self.ea_delta_ = fromNone(self.ea_delta, params_default["DELTA"])
        self.ea_theta_del_ = fromNone(self.ea_theta_del, params_default["THETA_DEL"])
        # INIT_FITNESS
        # INIT_ERROR
        # M_PROBATION
        # STATEFUL
        self.ea_set_subsumption_ = fromNone(
            self.ea_set_subsumption, params_default["SET_SUBSUMPTION"]
        )
        self.ea_theta_sub_ = fromNone(self.ea_theta_sub, params_default["THETA_SUB"])
        self.ea_compaction_ = fromNone(self.ea_compaction, params_default["COMPACTION"])
        # TELETRANSPORTATION
        # GAMMA
        # P_EXPLORE
        self.ea_select_type_ = fromNone(
            self.ea_select_type, params_default["EA_SELECT_TYPE"]
        )
        self.ea_select_size_ = fromNone(
            self.ea_select_size, params_default["EA_SELECT_SIZE"]
        )
        self.ea_theta_ea_ = fromNone(self.ea_theta_ea, params_default["THETA_EA"])
        self.ea_lambda_ = fromNone(self.ea_lambda, params_default["LAMBDA"])
        self.ea_p_crossover_ = fromNone(
            self.ea_p_crossover, params_default["P_CROSSOVER"]
        )
        self.ea_err_reduc_ = fromNone(self.ea_err_reduc, params_default["ERR_REDUC"])
        self.ea_fit_reduc_ = fromNone(self.ea_fit_reduc, params_default["FIT_REDUC"])
        self.ea_subsumption_ = fromNone(
            self.ea_subsumption, params_default["EA_SUBSUMPTION"]
        )
        # EA_PRED_RESET

        # Do a very rough check of whether we misspelled any of the boilerplate.
        # Better than nothing, I guess.
        for k_ in toolz.keyfilter(lambda x: x.endswith("_"), self.__dict__):
            k = k_.removesuffix("_")
            v_ = self.__dict__[k_]
            v = self.__dict__[k]
            assert (v is None and v_ is not None) or (v == v_)

        self.condition_ = "hyperrectangle_" + self.condition
        self.spread_min_ = self.spread_min
        self.tol_ = self.tol if self.tol is not None else self.epsilon0_ / 10

    def _init_xcs(self, DX):
        random_state = check_random_state(self.random_state)

        xcs = xcsf.XCS(DX, 1, 1)  # only 1 (dummy) action
        seed = random_state.randint(np.iinfo(np.int32).max)
        print("XCSF seed:", seed)
        xcs.seed(seed)

        xcs.OMP_NUM_THREADS = self.n_threads_
        # xcs.POP_INIT = ...
        xcs.POP_SIZE = self.n_pop_size_
        # We set this to `n_iter_no_change` so we can check for convergence
        # after that many iterations.
        xcs.MAX_TRIALS = self.n_iter_no_change
        # xcs.PERF_TRIALS = ...
        # xcs.LOSS_FUNC = ...
        # xcs.HUBER_DELTA = ...
        xcs.E0 = self.epsilon0_
        xcs.ALPHA = self.alpha_
        xcs.NU = self.nu_
        xcs.BETA = self.beta_
        xcs.DELTA = self.ea_delta_
        xcs.THETA_DEL = self.ea_theta_del_
        # xcs.INIT_FITNESS = ...
        # xcs.INIT_ERROR = ...
        # xcs.M_PROBATION = ...
        # xcs.STATEFUL = ...
        xcs.SET_SUBSUMPTION = self.ea_set_subsumption_
        xcs.THETA_SUB = self.ea_theta_sub_
        xcs.COMPACTION = self.ea_compaction_
        # xcs.TELETRANSPORTATION = ...
        # xcs.GAMMA = ...
        # xcs.P_EXPLORE = ...
        xcs.EA_SELECT_TYPE = self.ea_select_type_
        xcs.EA_SELECT_SIZE = self.ea_select_size_
        xcs.THETA_EA = self.ea_theta_ea_
        xcs.LAMBDA = self.ea_lambda_
        xcs.P_CROSSOVER = self.ea_p_crossover_
        xcs.ERR_REDUC = self.ea_err_reduc_
        xcs.FIT_REDUC = self.ea_fit_reduc_
        xcs.EA_SUBSUMPTION = self.ea_subsumption_
        # xcs.EA_PRED_RESET = ...

        xcs.action("integer")  # (dummy) integer actions

        if self.spread_min is None:
            vol_input_space = (X_max - X_min) ** DX
            # Assume a maximum of half the population size cubic rules to cover
            # input space.
            vol_min_rule = vol_input_space / (self.n_pop_size_ / 2.0)
            # The DX'th root is equal to the side length of a cube with
            # `vol_min_rule` volume.
            width_min_rule_cubic = vol_min_rule ** (1 / DX)
            spread_min_rule_cubic = width_min_rule_cubic / 2.0
            self.spread_min_ = spread_min_rule_cubic
        else:
            self.spread_min_ = self.spread_min

        args = {
            "min": X_min,  # minimum value of a lower bound
            "max": X_max,  # maximum value of an upper bound
            "spread_min": self.spread_min_,  # minimum initial spread
            "eta": 0,  # disable gradient descent of centers towards matched input mean
        }
        xcs.condition(self.condition_, args)

        args = {
            "x0": 1,  # bias attribute
            "scale_factor": 1000,  # initial diagonal values of the gain-matrix
            "lambda": 1,  # forget rate (small values may be unstable)
        }
        prediction_string = "rls_linear"
        xcs.prediction(prediction_string, args)

        return xcs

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        # This is required so that XCS does not (silently!?) segfault (see
        # https://github.com/rpreen/xcsf/issues/17 ).
        y = y.reshape((len(X), -1))

        N, DX = X.shape

        self._parse_params(DX)
        xcs = self._init_xcs(DX)

        n_fits = self.n_iter / self.n_iter_no_change
        if n_fits != int(n_fits):
            warnings.warn(
                "XCSF.n_iter_no_change not a divisor of n_iter, "
                "running up to n_iter_no_change more iterations"
            )
            n_fits = np.ceil(n_fits)
        n_fits = int(n_fits)

        score = np.inf
        delta_scores = [np.inf] * 15
        for i in range(n_fits):

            xcs.fit(X, y, True)

            if self.check_convergence:
                score_last = score
                score = xcs.score(X, y, cover=[0.0])
                delta_score = np.abs(score_last - score)
                delta_scores.append(delta_score)
                del delta_scores[0]
                delta_scores_mean = np.mean(delta_scores)
                if delta_scores_mean < self.tol_:
                    self.delta_scores_mean_final_ = delta_scores_mean
                    break
        self.n_updates_ = i * self.n_iter_no_change

        self.xcs_ = xcs

        return self

    def predict(self, X):
        check_is_fitted(self)

        X = check_array(X)

        return self.xcs_.predict(X, cover=[0.0])

    @property
    def rules_(self):
        check_is_fitted(self)

        return_condition = True
        return_action = True
        return_prediction = True
        json_string = self.xcs_.json(return_condition, return_action, return_prediction)
        pop = json.loads(json_string)
        rules = pop["classifiers"]
        return rules


def bounds(rules, transformer_X=None):
    """
    Extracts from the given rectangular condition–based population the lower and
    upper condition bounds.

    Note that center-spread conditions are allowed to have lower and upper
    bounds outside of the input space (e.g. if the center lies closer to the
    edge of the input space than the spread's width). These are fixed by
    clipping the resulting lower and upper bounds at the edges of the input
    space.

    Parameters
    ----------
    rules : list of dict
        A list of interval-based rules as created by `sklearn_xcsf.XCSF.rules_`
        (i.e. as exported by `xcs.json(True, True, True)`).
    transformer_X : object supporting `inverse_transform`
        Apply the given transformer's inverse transformation to the resulting
        lower and upper bounds (*after* they have been clipped to XCSF's input
        space edges).

    Returns
    -------
    list, list
        Two lists of NumPy arrays, the first one being lower bounds, the second
        one being upper bounds.
    """
    lowers = []
    uppers = []
    for rule in rules:
        cond = rule["condition"]
        if cond["type"] == "hyperrectangle_ubr":
            bound1 = np.array(rule["condition"]["bound1"])
            bound2 = np.array(rule["condition"]["bound2"])
            lower = np.min([bound1, bound2], axis=0)
            upper = np.max([bound1, bound2], axis=0)

        elif cond["type"] == "hyperrectangle_csr":
            center = np.array(rule["condition"]["center"])
            spread = np.array(rule["condition"]["spread"])
            lower = center - spread
            upper = center + spread
        else:
            raise NotImplementedError(
                "bounds_ only exists for hyperrectangular conditions"
            )

        lower = np.clip(lower, X_min, X_max)
        upper = np.clip(upper, X_min, X_max)

        lowers.append(lower)
        uppers.append(upper)

    if transformer_X is not None:
        lowers = transformer_X.inverse_transform(lowers)
        uppers = transformer_X.inverse_transform(uppers)
    else:
        lowers = np.array(lowers)
        uppers = np.array(lowers)

    return lowers, uppers
