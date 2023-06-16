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

import numpy as np
import xcsf
from sklearn.base import BaseEstimator, RegressorMixin  # type: ignore
from sklearn.metrics import mean_absolute_error
from sklearn.utils import check_random_state  # type: ignore
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


wiggle_room = 0.05
X_min = -1.0 - wiggle_room
X_max = 1.0 + wiggle_room


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
            random_state,
            n_iter=1000,
            n_pop_size=200,
            # TODO expose other important ones here as well (epsilon0 etc.)
            nu=5,
            p_crossover=0.8,
            theta_ea=50,
            ea_subsumption=False,
            theta_sub=50,
            ea_select_type="tournament",
            compaction=False,
            condition="hyperrectangle_csr",
            n_threads=8):
        self.random_state = random_state

        self.n_iter = n_iter
        self.n_pop_size = n_pop_size
        self.nu = nu
        self.p_crossover = p_crossover
        self.theta_ea = theta_ea
        self.ea_subsumption = ea_subsumption
        self.theta_sub = theta_sub
        self.ea_select_type = ea_select_type
        self.compaction = compaction
        self.condition = condition
        self.n_threads = n_threads

    def _init_xcs(self, X):
        random_state = check_random_state(self.random_state)

        xcs = xcsf.XCS(X.shape[1], 1, 1)  # only 1 (dummy) action
        seed = random_state.randint(np.iinfo(np.int32).max)
        print("XCSF seed:", seed)
        xcs.seed(seed)

        params = default_xcs_params(X.shape[1]) | {
            "MAX_TRIALS": self.n_iter,
            "POP_SIZE": self.n_pop_size,
            "NU": self.nu,
            "P_CROSSOVER": self.p_crossover,
            "THETA_EA": self.theta_ea,
            "EA_SUBSUMPTION": self.ea_subsumption,
            "THETA_SUB": self.theta_sub,
            "EA_SELECT_TYPE": self.ea_select_type,
            "COMPACTION": self.compaction,
            "OMP_NUM_THREADS" : self.n_threads,
        }
        set_xcs_params(xcs, params)

        xcs.action("integer")  # (dummy) integer actions

        N, DX = X.shape
        vol_input_space = (X_max - X_min)**DX
        # Assume a maximum of 1000 (arbitrary over-the-head number) cubic rules
        # to cover input space.
        vol_min_rule = vol_input_space / 1000.0
        # The DX'th root is equal to the side length of a cube with
        # `vol_min_rule` volume.
        width_min_rule_cubic = vol_min_rule**(1 / DX)
        spread_min_rule_cubic = width_min_rule_cubic / 2.0

        args = {
            "min": X_min,  # minimum value of a lower bound
            "max": X_max,  # maximum value of an upper bound
            "spread_min": spread_min_rule_cubic,  # minimum initial spread
            "eta":
            0,  # disable gradient descent of centers towards matched input mean
        }
        xcs.condition(self.condition, args)

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

        xcs = self._init_xcs(X)

        xcs.fit(X, y, True)

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
        json_string = self.xcs_.json(return_condition, return_action,
                                     return_prediction)
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
                "bounds_ only exists for hyperrectangular conditions")

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
