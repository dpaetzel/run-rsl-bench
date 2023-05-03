"""
Defines the protocol of how different things are stored to and later loaded from
mlflow.
"""
import json
import tempfile

import joblib
import mlflow


def _artifact_dir(artifact_uri):
    tracking_uri = mlflow.get_tracking_uri()
    assert tracking_uri.endswith("mlruns"), ("Valid tracking URIs should "
                                             "have the suffix \"mlruns\"")
    assert artifact_uri.startswith("mlruns")

    path = tracking_uri.removesuffix("mlruns") + artifact_uri
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
    json_string = model.xcs_.json(return_condition, return_action,
                                  return_prediction)
    mlflow.log_text(json_string, artifact_file=f"population.{label}.json")


def load_population(label):

    def _get_results(artifact_uri):
        path = _artifact_dir(artifact_uri)

        # TODO Consider not hardcoding filenames twice
        with open(path + f"/population.{label}.json", "r") as f:
            out = json.load(f)
        return out

    return _get_results
