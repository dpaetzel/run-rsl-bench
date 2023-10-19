# An extension of xcsf for if-then rule extraction.
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
import xcsf

import json
import numpy as np

from .utils import clamp_transform


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

        lowers.append(lower)
        uppers.append(upper)

    return lowers, uppers


class XCS(xcsf.XCS):
    @property
    def rules_(self):
        return_condition = True
        return_action = True
        return_prediction = True
        json_string = self.json(return_condition, return_action, return_prediction)
        pop = json.loads(json_string)
        return pop["classifiers"]

    def bounds_(self, X_min, X_max, transformer_X=None):
        lowers, uppers = _bounds(
            self.rules_, X_min=X_min, X_max=X_max, transformer_X=transformer_X
        )

        return clamp_transform(lowers, uppers, X_min, X_max, transformer_X)
