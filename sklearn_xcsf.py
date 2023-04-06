# Based on
# https://github.com/heidmic/suprb-experimentation/blob/xcsf_experiment/runs/examples/xcsf_final.py
# which was originally based on
# https://github.com/berbl-dev/berbl-exp/blob/main/src/experiments/xcsf.py .
import json

import numpy as np
import xcsf
from sklearn.base import BaseEstimator, RegressorMixin  # type: ignore
from sklearn.utils import check_random_state  # type: ignore
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


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


def default_xcs_params():
    xcs = xcsf.XCS(1, 1, 1)
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
            p_explore=0.9,
            theta_ea=50,
            ea_subsumption=False,
            ea_select_type="tournament",
            compaction=False,
            condition="hyperrectangle_csr"):
        self.random_state = random_state

        self.n_iter = n_iter
        self.n_pop_size = n_pop_size
        self.nu = nu
        self.p_crossover = p_crossover
        self.p_explore = p_explore
        self.theta_ea = theta_ea
        self.ea_subsumption = ea_subsumption
        self.ea_select_type = ea_select_type
        self.compaction = compaction
        self.condition = condition

    def _init_xcs(self, X):
        random_state = check_random_state(self.random_state)

        xcs = xcsf.XCS(X.shape[1], 1, 1)  # only 1 (dummy) action
        xcs.seed(random_state.randint(np.iinfo(np.int32).max))

        params = default_xcs_params() | {
            "MAX_TRIALS": self.n_iter,
            "POP_SIZE": self.n_pop_size,
            "NU": self.nu,
            "P_CROSSOVER": self.p_crossover,
            "P_EXPLORE": self.p_explore,
            "THETA_EA": self.theta_ea,
            "EA_SUBSUMPTION": self.ea_subsumption,
            "EA_SELECT_TYPE": self.ea_select_type,
            "COMPACTION": self.compaction
        }
        set_xcs_params(xcs, params)

        xcs.action("integer")  # (dummy) integer actions

        wiggle_room = 0.05
        args = {
            "min": -1.0 - wiggle_room,  # minimum value of a lower bound
            "max": 1.0 + wiggle_room,  # maximum value of an upper bound
            "spread_min": 0.1,  # minimum initial spread
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

        xcs = self._init_xcs(X)

        xcs.fit(X, y, True)

        self.xcs_ = xcs

        return self

    def predict(self, X):
        check_is_fitted(self)

        X = check_array(X)
        # self.xcs_.print_pset(True, True, True)

        return self.xcs_.predict(X)

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


def bounds(rules):
    lowers = []
    uppers = []
    for rule in rules:
        cond = rule["condition"]
        if cond["type"] == "hyperrectangle_ubr":
            lower = np.array(rule["condition"]["bound1"])
            upper = np.array(rule["condition"]["bound2"])
        elif cond["type"] == "hyperrectangle_csr":
            center = np.array(rule["condition"]["center"])
            spread = np.array(rule["condition"]["spread"])
            lower = center - spread
            upper = center + spread
        else:
            raise NotImplementedError(
                "bounds_ only exists for hyperrectangular conditions")
        lowers.append(lower)
        uppers.append(upper)
    return lowers, uppers
