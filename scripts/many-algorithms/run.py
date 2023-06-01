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
from shutil import rmtree

import click
import mlflow
import numpy as np
import optuna.distributions
import toolz
from dataset import file_digest, get_test, get_train
from optuna.integration import OptunaSearchCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import ARDRegression, BayesianRidge, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor


def best_params_fname(label):
    return f"{label}-best_params_.json"


defaults = dict(n_iter=100000, n_threads=4, timeout=10)

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
                # TODO
                "alpha": optuna.distributions.FloatDistribution(1e-6, 1e-3),
                "batch_size": optuna.distributions.IntDistribution(50, n_sample // 2),
                # Only used when solver=sgd
                # "learning_rate" : ["constant", "invscaling", "adaptive"],
                "learning_rate_init": optuna.distributions.FloatDistribution(
                    1e-5, 1e-2
                ),
            },
        ),
    ]


# Copied from berbl-exp.experiments.utils.
@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--n-threads",
    default=defaults["n_threads"],
    type=int,
    show_default=True,
    help="Number of threads to use while fitting XCSF",
)
@click.option(
    "-t",
    "--timeout",
    default=defaults["timeout"],
    type=int,
    show_default=True,
    help="Compute budget (in seconds) for the hyperparameter optimization",
)
@click.option("--run-name", type=str, default=None)
@click.option("--tracking-uri", type=str, default="mlruns")
@click.option("--experiment-name", type=str, default="optparams")
@click.argument("NPZFILE")
def optparams(n_threads, timeout, run_name, tracking_uri, experiment_name, npzfile):
    """

    Note that we, for now, parallelize this at the level of optuna (and only
    start a single Slurm job that uses a certain amount of cores).
    """

    print(f"Logging to mlflow tracking URI {tracking_uri}.")
    mlflow.set_tracking_uri(tracking_uri)

    print(f'Setting experiment name to "{experiment_name}".')
    mlflow.set_experiment(experiment_name)

    # TODO One run per tuning would be better (at least wrt to mlflow ui)

    print(f'Setting run name to "{run_name}".')
    with mlflow.start_run(run_name=run_name) as run:
        print(f"Run ID is {run.info.run_id}.")

        data = np.load(npzfile)
        mlflow.log_params(
            {
                "data.fname": npzfile,
                "data.sha256": file_digest(npzfile)
                # Python 3.11 and onwards we can simply do:
                # hashlib.file_digest(npzfile, digest="sha256")
            }
        )

        # Load training data.
        X, y = get_train(data)

        N, DX = X.shape
        mlflow.log_params(
            {
                "data.N": N,
                "data.DX": DX,
            }
        )

        # Load test data.
        X_test, y_test = get_test(data)

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

        mlflow.log_params(
            {
                "data.K": K,
                "data.linear_model_mse": data["linear_model_mse"],
                "data.linear_model_mae": data["linear_model_mae"],
                "data.linear_model_rsquared": data["linear_model_rsquared"],
                "data.rsl_model_mse": data["rsl_model_mse"],
                "data.rsl_model_mae": data["rsl_model_mae"],
                "data.rsl_model_rsquared": data["rsl_model_rsquared"],
            }
        )

        # Create a cache for transformers (this way, the pipeline does not fit
        # the MinMaxScaler over and over again).
        cachedir = tempfile.mkdtemp()

        def tune_model(model, label, param_distributions):
            estimator = make_pipeline(model, cachedir)

            param_distributions = toolz.keymap(
                lambda x: f"{regressor_name}__regressor__" + x, param_distributions
            )

            search = OptunaSearchCV(
                estimator,
                param_distributions=param_distributions,
                cv=4,
                n_jobs=n_threads,
                # Note that this is the RNG used by OptunaSearchCV itself (i.e.
                # for subsampling data, which we don't use, as well as for
                # sampling the parameter distributions), it does not get passed
                # down to the estimators.
                # random_state=1,
                return_train_score=True,
                # scoring="neg_mean_squared_error",
                scoring="neg_mean_absolute_error",
                subsample=1.0,
                # Only use timeout.
                n_trials=None,
                # Seconds.
                timeout=timeout,
                callbacks=[],
            )
            search.fit(X, y)

            return search

        ms = models(n_sample=len(X))

        for label, model, params in ms:
            if not params:
                print(f"Fitting {label} without tuning b/c no "
                      "hyperparameter distributions given …")
                from sklearn.model_selection import cross_val_score
                scores = cross_val_score(model, X, y, cv=4, n_jobs=n_threads)
                mlflow.log_dict({}, best_params_fname(label))
                print(f"Best parameters for {label}: {{}}")
                # As of 2023-05-31, `OptunaSearchCV.best_score_` is the mean of
                # the cv test scores. We thus use the same for untuned models.
                best_score_ = np.mean(scores)
            else:
                print(f"Tuning {label} …")
                search = tune_model(
                    model=model, label=label, param_distributions=params
                )

                mlflow.log_dict(search.best_params_, best_params_fname(label))
                print(f"Best parameters for {label}: {search.best_params_}")
                best_score_ = search.best_score_
            mlflow.log_metric(f"{label}.best_score_", best_score_)
            print(f"Best score for {label}: {best_score_}")

        # Remove cached transformers.
        rmtree(cachedir)


if __name__ == "__main__":
    cli()
