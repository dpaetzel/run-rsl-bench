# Protocol for storing and retrieving experiment artifacts in mlflow.
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

import json
import re
import tempfile

import joblib
import mlflow
import numpy as np


def _artifact_dir(artifact_uri, tracking_uri=None):
    """
    This assumes that there is only a single occurrence of `mlruns` in both the
    `artifact_uri` as well as the current `tracking_uri`.

    Parameters
    ----------
    tracking_uri : str or None
        If None, use the currently active tracking URI.
    """

    artifact_uri = re.compile("^(/.*)?mlruns/").sub("mlruns/", artifact_uri)

    if tracking_uri is None:
        tracking_uri = mlflow.get_tracking_uri()

    assert tracking_uri.endswith("mlruns") or tracking_uri.endswith("mlruns/"), (
        "Valid tracking URIs should " 'have the suffix "mlruns"'
    )
    assert artifact_uri.startswith("mlruns/")

    path = tracking_uri.removesuffix("/").removesuffix("mlruns") + artifact_uri
    return path


# TODO Consider to use mlflow's save_model instead (but Preen's XCS is
# incompatible with that as of 2023-05-03 and would need to be wrapped)
def log_scalers(scaler_X, scaler_y):
    with tempfile.TemporaryDirectory(prefix=f"scalers-") as tempdir_name:
        for obj, name in [(scaler_X, "scaler_X"), (scaler_y, "scaler_y")]:
            fname = f"{tempdir_name}/{name}.pkl"
            joblib.dump(obj, fname)
            mlflow.log_artifact(fname)


def load_scalers(artifact_uri):
    path = _artifact_dir(artifact_uri)
    # TODO Consider not hardcoding filenames twice
    scaler_X = joblib.load(f"{path}/scaler_X.pkl")
    scaler_y = joblib.load(f"{path}/scaler_y.pkl")
    return scaler_X, scaler_y


def log_population(model, label):
    return_condition = True
    return_action = True
    return_prediction = True
    json_string = model.xcs_.json(return_condition, return_action, return_prediction)
    mlflow.log_text(json_string, artifact_file=f"population.{label}.json")


def load_population(label):
    def _get_results(artifact_uri):
        path = _artifact_dir(artifact_uri)

        # TODO Consider not hardcoding filenames twice
        with open(path + f"/population.{label}.json", "r") as f:
            out = json.load(f)
        return out

    return _get_results


# Copied from berbl.utils.
def log_arrays(artifact_name, **arrays):
    with tempfile.TemporaryDirectory(prefix=f"{artifact_name}-") as tempdir_name:
        fname = f"{tempdir_name}/{artifact_name}.npz"
        np.savez(fname, **arrays)
        mlflow.log_artifact(fname)


def load_array(label, array_name):
    def _get_results(artifact_uri):
        path = _artifact_dir(artifact_uri)
        # TODO Consider not hardcoding filenames twice
        data = np.load(path + f"/results.{label}.npz", allow_pickle=True)
        out = data[array_name]
        data.close()
        return out

    return _get_results


def load_dict(dict_name, tracking_uri=None):
    """
    Parameters
    ----------
    tracking_uri : str or None
        If None, use the currently active tracking URI.
    """

    def _get_results(artifact_uri):
        path = _artifact_dir(artifact_uri, tracking_uri=tracking_uri)
        # Open the file in read mode
        with open(path + f"/{dict_name}.json", "r") as file:
            # Load the contents of the file as JSON
            out = json.load(file)
        return out

    return _get_results
