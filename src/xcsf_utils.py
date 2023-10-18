import json
import numpy as np


def rules(xcs):
    return_condition = True
    return_action = True
    return_prediction = True
    json_string = xcs.json(return_condition, return_action, return_prediction)
    pop = json.loads(json_string)
    rules = pop["classifiers"]
    return rules


def bounds(xcs, X_min, X_max, transformer_X=None):
    return _bounds(rules(xcs), X_min, X_max, transformer_X)


def _bounds(rules, X_min, X_max, transformer_X=None):
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
