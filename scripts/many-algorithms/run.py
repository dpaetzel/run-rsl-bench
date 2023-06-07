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
from sklearn.linear_model import ARDRegression, BayesianRidge, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import check_random_state

best_params_fname = "best_params.json"

defaults = dict(n_iter=100000, timeout=10)


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

regressor_name = "ttregressor"


def make_pipeline(model, cachedir):

    estimator = Pipeline(
        steps=[
            ("minmaxscaler", MinMaxScaler(feature_range=(-1.0, 1.0))),
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


class MyMLPRegressor(MLPRegressor):
    def __init__(
        self,
        size_layer1=10,
        size_layer2=10,
        # The following are just copied from the MLPRegressor definition.
        activation="relu",
        *,
        solver="adam",
        alpha=0.0001,
        batch_size="auto",
        learning_rate="constant",
        learning_rate_init=0.001,
        power_t=0.5,
        max_iter=200,
        shuffle=True,
        random_state=None,
        tol=0.0001,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,
        n_iter_no_change=10,
        max_fun=15000,
    ):
        super().__init__(
            hidden_layer_sizes=[size_layer1, size_layer2],
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            power_t=power_t,
            max_iter=max_iter,
            shuffle=shuffle,
            random_state=random_state,
            tol=tol,
            verbose=verbose,
            warm_start=warm_start,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            n_iter_no_change=n_iter_no_change,
            max_fun=max_fun,
        )

        self.size_layer1 = size_layer1
        self.size_layer2 = size_layer2

    def set_params(self, **params):
        if "size_layer1" in params:
            self.size_layer1 = params["size_layer1"]
        if "size_layer2" in params:
            self.size_layer2 = params["size_layer2"]
        self.hidden_layer_sizes = [self.size_layer1, self.size_layer2]
        super().set_params(
            **{k: params[k] for k in params if k not in ["size_layer1", "size_layer2"]}
        )
        return self


def models(n_sample):
    return [
        ("GaussianProcessRegressor", GaussianProcessRegressor(), {}),
        (
            "AdaBoostRegressor",
            AdaBoostRegressor(DecisionTreeRegressor()),
            {
                "n_estimators": optuna.distributions.IntDistribution(2, 100),
                # https://stats.stackexchange.com/a/444972/321460 says
                # learning_rate should be < 0.1.
                "learning_rate": optuna.distributions.FloatDistribution(0.0, 0.2),
                "loss": optuna.distributions.CategoricalDistribution(
                    ["linear", "square", "exponential"]
                ),
            }
            | toolz.keymap(lambda x: "base_estimator__" + x, params_dt),
        ),
        ("DecisionTreeRegressor", DecisionTreeRegressor(), {} | params_dt),
        (
            "RandomForestRegressor",
            RandomForestRegressor(),
            {
                "n_estimators": optuna.distributions.IntDistribution(2, 100),
                # "bootstrap" True by default
                # "oob_score" False by default
                # "max_samples" Number of samples to draw from X to train each base estimator, default is len(X).
            }
            | params_dt,
        ),
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
        (
            "BayesianRidge",
            BayesianRidge(),
            {
                # TODO Maybe hyperparameters here
            },
        ),
        (
            "ARDRegression",
            ARDRegression(),
            {
                # TODO Maybe hyperparameters here
            },
        ),
        (
            "MLPRegressor",
            MyMLPRegressor(
                activation="relu",
                max_iter=1000000,
                early_stopping=True,
                validation_fraction=0.1,
            ),
            {
                "size_layer1": optuna.distributions.IntDistribution(10, 1010, step=100),
                "size_layer2": optuna.distributions.IntDistribution(10, 1010, step=100),
                "solver": optuna.distributions.CategoricalDistribution(
                    ["adam", "lbfgs"]
                ),
                # The default for alpha (the L2 regularization factor) is 1e-4.
                "alpha": optuna.distributions.CategoricalDistribution(
                    [1e-3, 5e-4, 3e-4, 1e-4, 5e-5, 3e-5, 1e-5]
                ),
                "batch_size": optuna.distributions.CategoricalDistribution(
                    [16, 32, 64, 128, 256]
                ),
                "learning_rate_init": optuna.distributions.CategoricalDistribution(
                    [1e-3, 5e-4, 3e-4, 1e-4, 5e-5, 3e-5, 1e-5]
                ),
            },
        ),
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
            cv=4,
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

    ms = models(n_sample=len(X))

    for label, model, params in ms:
        print(f'Setting run name to "{run_name}".')
        with mlflow.start_run(run_name=run_name) as run:
            print(f"Run ID is {run.info.run_id}.")

            print(f"Drawing and setting model RNG …")
            seed_model = randseed(random_state)
            try:
                model.set_params(random_state=seed_model)
                print(f"RNG seed for {label} set to {seed_model}.")
            except ValueError:
                print(f"Model {label} is deterministic, no seed set.")

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
            print(f"Best hyperparameters for {label}: {best_params_}")
            mlflow.log_metric(f"best_score", best_score_)
            print(f"Best score for {label}: {best_score_}")

            # Since best_params_ only contains values for the hyperparameters
            # that we're optimizing over and not all the hyperparameters, we
            # also store the estimator itself (which then contains *all* the
            # possible hyperparameters).
            print(f"Storing best {label} estimator …")
            y_pred = best_estimator_.predict(X)
            signature = infer_signature(X, y_pred)
            log_model(best_estimator_, "best_estimator", signature=signature)

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
            best_estimator = load_model(
                f'{store._artifact_dir(row["artifact_uri"], tracking_uri=tuning_uri)}/best_estimator'
            )
            print(f"Extracting hyperparameters from best estimator …")
            params = best_estimator.get_params()
            print(f"Setting hyperparameters of {label} …")
            estimator.set_params(**params)

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

            print(f"Storing {label} estimator …")
            signature = infer_signature(X, y_pred)
            log_model(estimator, "estimator", signature=signature)

    # Remove cached transformers.
    rmtree(cachedir)


if __name__ == "__main__":
    cli()
