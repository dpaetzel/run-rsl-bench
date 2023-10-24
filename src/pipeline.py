# Unified pipeline used in all experiments.
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
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# TODO Unify X_MIN vs X_min vs x_min
X_MIN, X_MAX = -1.0, 1.0


regressor_name = "ttregressor"


def make_pipeline(regressor, cachedir=None):
    """
    Plug the given regressor into our unified pipeline and configure a cache
    directory (see `Pipeline` docs) for storing transformers so that they are
    not fitted repeatedly on the training data (e.g. relevant for CV).
    """
    estimator = Pipeline(
        steps=[
            ("minmaxscaler", MinMaxScaler(feature_range=(X_MIN, X_MAX))),
            (
                regressor_name,
                TransformedTargetRegressor(
                    regressor=regressor, transformer=StandardScaler()
                ),
            ),
        ],
        memory=cachedir,
    )

    return estimator
