# Perform hyperparameter search and runs on a set of `syn-rsl-benchs` data sets.
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

import tempfile
import time
from shutil import rmtree

import click
import mlflow
import numpy as np
import optuna.distributions
import store
import toolz
from dataset import file_digest, get_test, get_train
from mlflow.models.signature import infer_signature
from mlflow.sklearn import load_model, log_model
from optuna.integration import OptunaSearchCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import check_random_state
from sklearn_xcsf import XCSF

best_params_fname = "best_params.json"
best_params_all_fname = "best_params_all.json"

defaults = dict(n_iter=100000, timeout=10)


def _non_primitive_to_string(obj):
    if isinstance(obj, (int, float, str, bool)):
        return obj
    else:
        return str(obj)


def randseed(random_state: np.random.RandomState):
    """
    Sometimes we need to generate a new random seed from a `RandomState` due to
    different APIs (e.g. NumPy wants the new RNG API, scikit-learn uses the
    legacy NumPy `RandomState` API etc.).
    """
    # Highest possible seed is `2**32 - 1` for NumPy legacy generators.
    return random_state.randint(2**32 - 1)


params_dt = {
    "criterion": optuna.distributions.CategoricalDistribution(
        ["squared_error", "friedman_mse", "absolute_error"]
    ),
    # ValueError: Some value(s) of y are negative which is not allowed for Poisson regression.
    # , "poisson"]),
    # TODO Probably set max max_depth input dimension dependent?
    "max_depth": optuna.distributions.IntDistribution(1, 20),
    "min_samples_split": optuna.distributions.IntDistribution(2, 5),
    "min_samples_leaf": optuna.distributions.IntDistribution(1, 5),
    # min_impurity_decrease
}


params_fixed_xcsf = {
    "n_threads": 4,
    "n_iter": 200000,
    "condition": "csr",
    "ea_select_type": "tournament",
    "ea_lambda": 2,
}


def spread_min_cubic(DX, K=100):
    """
    Given an input space dimension and a rule count, returns the spread that
    each rule would have if each rule were cubic and the rules fully covered the
    input space without overlaps.
    """
    vol_input_space = (X_MAX - X_MIN) ** DX
    # Assume K cubic rules to cover the input space.
    vol_min_rule = vol_input_space / K
    # The DX'th root is equal to the side length of a cube with
    # `vol_min_rule` volume.
    width_min_rule_cubic = vol_min_rule ** (1 / DX)
    spread_min_rule_cubic = width_min_rule_cubic / 2.0
    return spread_min_rule_cubic


def params_var_xcsf(DX, n_pop_size):
    spread_min_cubic_ = spread_min_cubic(DX, n_pop_size)
    return {
        "spread_min": optuna.distributions.CategoricalDistribution(
            [factor * spread_min_cubic_ for factor in [0.5, 0.75, 1, 2]]
        ),
        "epsilon0": optuna.distributions.CategoricalDistribution(
            [0.01, 0.05, 0.1, 0.2]
        ),
        "beta": optuna.distributions.CategoricalDistribution([0.01, 0.05, 0.1]),
    }


regressor_name = "ttregressor"

X_MIN, X_MAX = -1.0, 1.0


def make_pipeline(model, cachedir):

    estimator = Pipeline(
        steps=[
            ("minmaxscaler", MinMaxScaler(feature_range=(X_MIN, X_MAX))),
            (
                regressor_name,
                TransformedTargetRegressor(
                    regressor=model, transformer=StandardScaler()
                ),
            ),
        ],
        memory=cachedir,
    )

    return estimator


def make_xcsf_triple(DX, n_pop_size):
    return (
        f"XCSF{n_pop_size}",
        XCSF(n_pop_size=n_pop_size, **params_fixed_xcsf),
        params_var_xcsf(DX, n_pop_size=n_pop_size),
    )


def models(DX, n_sample):
    return [
        ("Ridge", Ridge(), {"alpha": optuna.distributions.FloatDistribution(0.0, 1.0)}),
        (
            "KNeighborsRegressor",
            KNeighborsRegressor(),
            {
                "n_neighbors": optuna.distributions.IntDistribution(1, 10),
                "weights": optuna.distributions.CategoricalDistribution(
                    ["uniform", "distance"]
                ),
            },
        ),
        make_xcsf_triple(DX, n_pop_size=50),
        make_xcsf_triple(DX, n_pop_size=100),
        make_xcsf_triple(DX, n_pop_size=200),
        make_xcsf_triple(DX, n_pop_size=400),
        make_xcsf_triple(DX, n_pop_size=800),
    ]


# Copied from berbl-exp.experiments.utils.
@click.group()
@click.argument("NPZFILE")
@click.pass_context
def cli(ctx, npzfile):
    ctx.ensure_object(dict)

    data = np.load(npzfile)

    # Load training and test data.
    X, y = get_train(data)
    X_test, y_test = get_test(data)
    N, DX = X.shape

    # Since `y` should have shape (N, 1) and some of the sklearn estimators
    # used warn if such a shape is passed to them instead of (N,), we
    # flatten.
    y = y.ravel()
    y_test = y_test.ravel()
    assert len(y) == len(X)
    assert len(y_test) == len(X_test)

    # Load ground truth.
    centers_true = data["centers"]
    K = len(centers_true)

    ctx.obj["npzfile"] = npzfile
    ctx.obj["data"] = data
    ctx.obj["X"] = X
    ctx.obj["y"] = y
    ctx.obj["X_test"] = X_test
    ctx.obj["y_test"] = y_test
    ctx.obj["DX"] = DX
    ctx.obj["K"] = K
    ctx.obj["N"] = N
    ctx.obj["sha256"] = file_digest(npzfile)


@cli.command()
@click.option(
    "-t",
    "--timeout",
    default=defaults["timeout"],
    type=int,
    show_default=True,
    help="Compute budget (in seconds) for the hyperparameter optimization",
)
@click.option(
    "--seed",
    default=None,
    type=int,
    show_default=True,
    help="Seed to initialize the algorithms' RNGs",
)
@click.option("--run-name", type=str, default=None)
@click.option("--tracking-uri", type=str, default="mlruns")
@click.option("--experiment-name", type=str, default="optparams")
@click.pass_context
def optparams(ctx, timeout, seed, run_name, tracking_uri, experiment_name):
    """
    TODO
    """
    npzfile = ctx.obj["npzfile"]
    data = ctx.obj["data"]
    X = ctx.obj["X"]
    y = ctx.obj["y"]
    X_test = ctx.obj["X_test"]
    y_test = ctx.obj["y_test"]
    DX = ctx.obj["DX"]
    K = ctx.obj["K"]
    N = ctx.obj["N"]
    sha256 = ctx.obj["sha256"]

    print(f"Initializing RNG from seed {seed} …")
    random_state = check_random_state(seed)

    print(f"Logging to mlflow tracking URI {tracking_uri}.")
    mlflow.set_tracking_uri(tracking_uri)

    print(f'Setting experiment name to "{experiment_name}".')
    mlflow.set_experiment(experiment_name)

    # scoring="neg_mean_squared_error"
    scoring = "neg_mean_absolute_error"

    # Create a cache for transformers (this way, the pipeline does not fit
    # the MinMaxScaler over and over again).
    cachedir = tempfile.mkdtemp()

    def early_stopping(study, trial):
        n_min_trials = 100
        rate_window = 0.1

        trials = study.trials
        n_trials = len(trials)

        if n_trials < n_min_trials:
            return
        else:
            len_window = int(rate_window * n_trials)
            window = trials[-len_window:]
            scores_window = np.array(list(map(lambda x: x.values[0], window)))
            if np.all(scores_window <= study.best_value):
                print(
                    "Stopping early due to no change in best score for "
                    f"{len_window} trials."
                )
                study.stop()

    def tune_model(model, label, param_distributions):
        estimator = make_pipeline(model, cachedir)

        param_distributions = toolz.keymap(
            lambda x: f"{regressor_name}__regressor__" + x, param_distributions
        )

        search = OptunaSearchCV(
            estimator,
            param_distributions=param_distributions,
            cv=5,
            # This may suffer from GIL otherwise since multithreading is
            # implemented via `threading`.
            n_jobs=1,
            # Note that this is the RNG used by OptunaSearchCV itself (i.e.
            # for subsampling data, which we don't use, as well as for
            # sampling the parameter distributions), it does not get passed
            # down to the estimators.
            #
            # Note that fixing the seed does *not* guarantee the same results
            # because we end tuning based on time.
            random_state=random_state,
            return_train_score=True,
            scoring=scoring,
            subsample=1.0,
            # Only use timeout.
            n_trials=None,
            # Seconds.
            timeout=timeout,
            callbacks=[early_stopping],
        )
        search.fit(X, y)

        return search

    ms = models(DX=DX, n_sample=len(X))

    for label, model, params in ms:
        print(f'Setting run name to "{run_name}".')
        with mlflow.start_run(run_name=run_name) as run:
            print(f"Run ID is {run.info.run_id}.")

            try:
                model.set_params(random_state=None)
                print(f"Set model RNG.")
            except ValueError:
                print(f"Model {label} is deterministic, no RNG set.")

            mlflow.log_params(
                {
                    "seed": seed,
                    "timeout": timeout,
                    "scoring": scoring,
                    "algorithm": label,
                    "data.fname": npzfile,
                    "data.sha256": sha256,
                    # Python 3.11 and onwards we can simply do:
                    # hashlib.file_digest(npzfile, digest="sha256")
                    "data.N": N,
                    "data.DX": DX,
                    "data.K": K,
                    "data.linear_model_mse": data["linear_model_mse"],
                    "data.linear_model_mae": data["linear_model_mae"],
                    "data.linear_model_rsquared": data["linear_model_rsquared"],
                    "data.rsl_model_mse": data["rsl_model_mse"],
                    "data.rsl_model_mae": data["rsl_model_mae"],
                    "data.rsl_model_rsquared": data["rsl_model_rsquared"],
                }
            )

            if not params:
                print(
                    f"Fitting {label} without tuning b/c no "
                    "hyperparameter distributions given …"
                )

                estimator = make_pipeline(model, cachedir)
                scores = cross_val_score(estimator, X, y, cv=4, scoring=scoring)

                best_params_ = {}
                best_estimator_ = estimator.fit(X, y)
                n_trials_ = 1
                # As of 2023-05-31, `OptunaSearchCV.best_score_` is the mean of
                # the cv test scores. We thus use the same for untuned models.
                best_score_ = np.mean(scores)
            else:
                print(f"Tuning {label} …")

                search = tune_model(
                    model=model,
                    label=label,
                    param_distributions=params,
                )

                best_params_ = search.best_params_
                best_estimator_ = search.best_estimator_
                best_score_ = search.best_score_
                n_trials_ = search.n_trials_

            mlflow.log_metric(f"n_trials", n_trials_)
            print(f"Finished after {n_trials_} trials.")
            mlflow.log_dict(best_params_, best_params_fname)
            # This is meant mostly for later sanity checking.  Since
            # best_params_ only contains values for the hyperparameters that
            # we're optimizing over and not all the hyperparameters of the
            # estimator, we also store all the hyperparameters in another dict
            # (with non-primitives converted to strings). Note that we do not
            # store the estimator itself to be more light on disk space (some of
            # the estimators are in the range of tens of MBs if serialized fully
            # and we perform a lot of runs); also, future sklearn versions may
            # not be able to deal with estimators serialized like this anyway.
            # And: Some estimators are not sklearn-style serializable without
            # additional effort (looking at you, XCSF).
            best_params_all_ = toolz.valmap(
                _non_primitive_to_string, best_estimator_.get_params()
            )
            mlflow.log_dict(best_params_all_, best_params_all_fname)
            print(f"Best hyperparameters for {label}: {best_params_}")
            mlflow.log_metric(f"best_score", best_score_)
            print(f"Best score for {label}: {best_score_}")

    # Remove cached transformers.
    rmtree(cachedir)


@cli.command()
@click.option(
    "--seed",
    default=None,
    type=int,
    show_default=True,
    help="Seed to initialize the algorithms' RNGs",
)
@click.option("--run-name", type=str, default=None)
@click.option("--tracking-uri", type=str, default="mlruns")
@click.option("--experiment-name", type=str, default="runbest")
@click.option("--tuning-uri", type=str, required=True)
@click.option("--tuning-experiment-name", type=str, default="optparams")
@click.pass_context
def runbest(
    ctx,
    seed,
    run_name,
    tuning_uri,
    tuning_experiment_name,
    tracking_uri,
    experiment_name,
):
    """
    Read the best hyperparameter values for the algorithms considered from the
    mlflow tracking URI given by --tuning-uri, then perform one run for each
    algorithm using those hyperparameter values on data given by NPZFILE.
    """

    npzfile = ctx.obj["npzfile"]
    data = ctx.obj["data"]
    X = ctx.obj["X"]
    y = ctx.obj["y"]
    X_test = ctx.obj["X_test"]
    y_test = ctx.obj["y_test"]
    DX = ctx.obj["DX"]
    K = ctx.obj["K"]
    N = ctx.obj["N"]
    sha256 = ctx.obj["sha256"]

    print(f"Loading runs from {tuning_uri} …")
    mlflow.set_tracking_uri(tuning_uri)
    df = mlflow.search_runs(experiment_names=[tuning_experiment_name])

    # Create a cache for transformers (this way, the pipeline does not fit
    # the MinMaxScaler over and over again).
    cachedir = tempfile.mkdtemp()

    print(f"Setting mlflow tracking URI to {tracking_uri} …")
    mlflow.set_tracking_uri(tracking_uri)

    print(f"Initializing RNG from seed {seed} …")
    random_state = check_random_state(seed)

    print(f'Setting experiment name to "{experiment_name}".')
    mlflow.set_experiment(experiment_name)

    ms = models(n_sample=N)

    for label, model, _ in ms:
        print()
        print(f"Running for {label} …")
        print(f'Setting run name to "{run_name}".')
        with mlflow.start_run(run_name=run_name) as run:
            print(f"Run ID is {run.info.run_id}.")

            print(f"Loading tuning data for {label} …")
            row = df[
                (df["params.algorithm"] == label) & (df["params.data.sha256"] == sha256)
            ]
            assert (
                not row.empty
            ), f"Algorithm seems to not have been run on the file {npzfile}"
            assert len(row) == 1, (
                f"Algorithm was run multiple times on the file {npzfile}, "
                "possibly ambiguous tuning results"
            )
            row = row.iloc[0]

            print(f"Creating pipeline …")
            estimator = make_pipeline(model, cachedir)

            print(f"Loading estimator with best hyperparameters from tuning data …")
            best_params = store.load_dict("best_params", tracking_uri=tuning_uri)(
                row["artifact_uri"]
            )
            print(f"Setting hyperparameters of {label} …")
            estimator.set_params(**best_params)

            print(f"Drawing and setting model RNG …")
            seed_model = randseed(random_state)
            try:
                estimator.set_params(
                    **{f"{regressor_name}__regressor__random_state": seed_model}
                )
                print(f"RNG seed for {label} set to {seed_model}.")
            except ValueError:
                print(f"Model {label} is deterministic, no seed set.")

            mlflow.log_params(
                {
                    "algorithm": label,
                    # Seed used to generate the seeds for the models.
                    "seed": seed,
                    # Seed used for this specific model.
                    "seed_model": seed_model,
                    "data.fname": npzfile,
                    "data.sha256": sha256,
                    # Python 3.11 and onwards we can simply do:
                    # hashlib.file_digest(npzfile, digest="sha256")
                    "data.N": N,
                    "data.DX": DX,
                    "data.K": K,
                    "data.linear_model_mse": data["linear_model_mse"],
                    "data.linear_model_mae": data["linear_model_mae"],
                    "data.linear_model_rsquared": data["linear_model_rsquared"],
                    "data.rsl_model_mse": data["rsl_model_mse"],
                    "data.rsl_model_mae": data["rsl_model_mae"],
                    "data.rsl_model_rsquared": data["rsl_model_rsquared"],
                }
            )

            print(f"Fitting {label} …")
            t_start = time.time()
            estimator.fit(X, y)
            t_end = time.time()
            duration_fit = t_end - t_start
            print(f"Fitting {label} took {duration_fit} seconds.")

            print(f"Performing predictions with {label} …")
            y_pred = estimator.predict(X)
            mae_train = mean_absolute_error(y_pred, y)
            mse_train = mean_squared_error(y_pred, y)
            y_test_pred = estimator.predict(X_test)
            mae_test = mean_absolute_error(y_test_pred, y_test)
            mse_test = mean_squared_error(y_test_pred, y_test)
            mlflow.log_metrics(
                {
                    "mae.train": mae_train,
                    "mse.train": mse_train,
                    "mae.test": mae_test,
                    "mse.test": mse_test,
                    "duration_fit": duration_fit,
                }
            )
            print(f"Achieved a test MAE of {mae_test}.")

            print(f"Storing predictions made by {label} …")
            store.log_arrays(
                f"pred",
                y_pred=y_pred,
                y_test_pred=y_test_pred,
            )

            if False:
                try:
                    print(f"Storing {label} estimator …")
                    signature = infer_signature(X, y_pred)
                    log_model(estimator, "estimator", signature=signature)
                except:
                    print("Failed to store estimator. Continuing anyway.")

    # Remove cached transformers.
    rmtree(cachedir)


if __name__ == "__main__":
    cli()
