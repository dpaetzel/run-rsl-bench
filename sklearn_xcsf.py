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
            condition="hyperrectangle_csr"):
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

    def _init_xcs(self, X):
        random_state = check_random_state(self.random_state)

        xcs = xcsf.XCS(X.shape[1], 1, 1)  # only 1 (dummy) action
        xcs.seed(random_state.randint(np.iinfo(np.int32).max))

        params = default_xcs_params(X.shape[1]) | {
            "MAX_TRIALS": self.n_iter,
            "POP_SIZE": self.n_pop_size,
            "NU": self.nu,
            "P_CROSSOVER": self.p_crossover,
            "THETA_EA": self.theta_ea,
            "EA_SUBSUMPTION": self.ea_subsumption,
            "THETA_SUB": self.theta_sub,
            "EA_SELECT_TYPE": self.ea_select_type,
            "COMPACTION": self.compaction
        }
        set_xcs_params(xcs, params)

        xcs.action("integer")  # (dummy) integer actions

        wiggle_room = 0.05
        self.X_min_ = -1.0 - wiggle_room
        self.X_max_ = 1.0 + wiggle_room

        N, DX = X.shape
        vol_input_space = (self.X_max_ - self.X_min_)**DX
        # Assume a maximum of 1000 (arbitrary over-the-head number) cubic rules
        # to cover input space.
        vol_min_rule = vol_input_space / 1000.0
        # The DX'th root is equal to the side length of a cube with
        # `vol_min_rule` volume.
        width_min_rule_cubic = vol_min_rule**(1 / DX)
        spread_min_rule_cubic = width_min_rule_cubic / 2.0

        args = {
            "min": self.X_min_,  # minimum value of a lower bound
            "max": self.X_max_,  # maximum value of an upper bound
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

        # Make room for the default rule that we will add later.
        xcs.POP_SIZE = self.n_pop_size - 1

        xcs.fit(X, y, True)

        self.xcs_ = xcs

        # Simply predict the overall training data output mean everywhere (which
        # should be 0.0 due to standardization).
        pred = 0.0
        if self.condition == "hyperrectangle_ubr":
            bound1 = [self.X_min_] * DX
            bound2 = [self.X_max_] * DX
        elif self.condition == "hyperrectangle_csr":
            bound1 = [(self.X_min_ + self.X_max_) / 2.0] * DX
            bound2 = [(self.X_max_ - self.X_min_) / 2.0] * DX
        else:
            raise ValueError("Only hyperrectangle_{ubr,csr} are supported")
        classifier = {
            # TODO Check these defaults for being sensible
            # "error": np.mean([r["error"] for r in self.rules_]),
            "error": mean_absolute_error([pred] * N, y),
            # "fitness": np.mean([r["fitness"] for r in self.rules_]),
            # Should have a very small impact where other rules match.
            # TODO Consider computing fitness using the XCSF formula
            "fitness": 1e-10 * np.mean([r["fitness"] for r in self.rules_]),
            "accuracy": np.mean([r["accuracy"] for r in self.rules_]),
            "set_size": np.mean([r["set_size"] for r in self.rules_]),
            "numerosity": 1,
            "experience": N,
            "time": self.n_iter + 1,
            "samples_seen": N,
            "samples_matched": N,
            "condition": {
                "type": self.condition,
                "bound1": bound1,
                "bound2": bound2,
                "mutation": [0.0]
            },
            "action": {
                "type": "integer",
                "action": 0,
                "mutation": [0.0]
            },
            "prediction": {
                "type": "rls_linear",
                "weights": [pred] + [0.0] * DX
            }
        }

        json_str: str = json.dumps(classifier)  # dictionary to JSON

        # json_insert_cl performs roulette wheel deletion if XCSF's population
        # size is exceeded. We therefore have to make sure that our default rule
        # does not get deleted just after having been added to the population.
        #
        # Setting a very high fitness to the default rule does not work: Albeit
        # the probability of deletion is very small, it is not zero. Also,
        # fitness also serves as the rule's mixing weight (and we don't want to
        # bias the system prediction towards the default rule too much).
        #
        # Therefore we simply increase the population size (note that we
        # decreased the population size by one before fitting so the
        # population's size is now as the user configured).
        xcs.POP_SIZE = self.n_pop_size
        xcs.json_insert_cl(json_str)

        return self

    def predict(self, X):
        check_is_fitted(self)

        X = check_array(X)

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
