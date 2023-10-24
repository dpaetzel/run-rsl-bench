# Perform hyperparameter search and runs on a set of `RSLModels.jl` data sets.
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
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
import store
import toolz
import lineartree
from dataset import file_digest, get_test, get_train
from mlflow.models.signature import infer_signature
from mlflow.sklearn import load_model, log_model
from optuna.integration import OptunaSearchCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import check_random_state

from pipeline import make_pipeline, regressor_name, X_MIN, X_MAX
from rule_support.ensemble import RandomForestRegressor
from rule_support.tree import DecisionTreeRegressor
from rule_support.xcsf import XCS
from rule_support.suprb import SupRB

best_params_fname = "best_params.json"
best_params_all_fname = "best_params_all.json"

defaults = dict(n_iter=100000, timeout=10)


# We try to parallelize stuff at the algorithm level fourways. I.e. XCSF.
# SupRB's `n_jobs` seems not to behave well with larger amounts of training
# data.
N_JOBS = 4


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
    "criterion": CategoricalDistribution(
        ["squared_error", "friedman_mse", "absolute_error"]
    ),
    # ValueError: Some value(s) of y are negative which is not allowed for Poisson regression.
    # , "poisson"]),
    # TODO Probably set max max_depth input dimension dependent?
    "max_depth": IntDistribution(1, 20),
    "min_samples_split": IntDistribution(2, 5),
    "min_samples_leaf": IntDistribution(1, 5),
    # min_impurity_decrease
}


def params_xcsf_condition(spread_min):
    return {
        "type": "hyperrectangle_csr",
        "args": {
            # Minimum initial spread.
            "spread_min": spread_min,
            # Minimum value of a center/bound. We add a bit of wiggle room.
            "min": X_MIN - 0.05,
            # Maximum value of a center/bound. We add a bit of wiggle room.
            "max": X_MAX + 0.05,
            # Gradient descent rate for moving centers to mean inputs matched.
            "eta": 0,
        },
    }


def params_xcsf(DX, n_pop_size, n_train, seed):
    return {
        "x_dim": DX,
        "y_dim": 1,
        "condition": params_xcsf_condition(spread_min_cubic(DX, n_pop_size)),
        # Constant local models.
        "prediction": {
            "type": "constant",
        },
        # We do regression and only have a single (“dummy”) action.
        "n_actions": 1,
        "action": {"type": "integer"},
        "pop_size": n_pop_size,
        "max_trials": 200000,
        "random_state": seed,
        "omp_num_threads": N_JOBS,
        # Don't load an existing population.
        "population_file": "",
        # Whether to seed the population with random rules.
        "pop_init": True,
        # `perf_trials > max_trials` means don't output performance stats.
        "perf_trials": 1000000,
        "loss_func": "mae",
        "set_subsumption": False,
        # Only relevant if `set_subsumption`.
        "theta_sub": 100,
        # Target error below which accuracy is set to 0.
        "e0": 0.01,
        # Accuracy offset for rules with errors above `e0`.
        "alpha": 0.1,
        # Accuracy slope.
        "nu": 5,
        # Learning rate for updating error, fitness and set size.
        "beta": 0.1,
        # Fraction of least fit rule to increase deletion vote.
        "delta": 0.1,
        # Min experience before fitness used in probability of deletion.
        # TODO Make theta_del depend on the number training examples?
        "theta_del": 20,
        # Initial rule fitness.
        "init_fitness": 0.01,
        # Initial rule error.
        "init_error": 0,
        # Trials since creation a rule must match at least 1 input or be
        # deleted. Any rule must match at least one of the training data points.
        "m_probation": n_train,
        # Rules should retain state between trials.
        "stateful": True,
        # “If enabled and system error < e0, the largest of 2 roulette spins is
        # deleted.”
        "compaction": False,
        "ea": {
            "select_type": "tournament",
            # Fraction of set size for tournament parental selection.
            "select_size": 0.4,
            # Average set time between EA invocations.
            "theta_ea": 50,
            # Number of offspring to create each EA invocation (use multiples of 2).
            "lambda": 2,
            # Probability of applying crossover.
            "p_crossover": 0.8,
            # Factor to reduce created offspring's error by (1=disabled).
            "err_reduc": 1,
            # Factor to reduce created offspring's fitness by (1=disabled).
            "fit_reduc": 0.1,
            # Whether to try and subsume offspring rules.
            "subsumption": False,
            # Whether to reset offspring predictions instead of copying.
            "pred_reset": False,
        },
    }


def spread_min_cubic(DX, K=100, X_MIN=X_MIN, X_MAX=X_MAX):
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
        # Note that this CategoricalDistribution raises a UserWarning because
        # it's a distribution over `dict`s which are not supported by Optuna's
        # storages. However, we do not use those storages so we're fine. We
        # could only circumvent this by either changing XCS's parameters to
        # being flat (i.e. condition not being expected to be a `dict`) or by
        # switching to Optuna's default interface.
        "condition": CategoricalDistribution(
            [
                params_xcsf_condition(factor * spread_min_cubic_)
                for factor in [0.5, 0.75, 1, 2]
            ]
        ),
        "e0": CategoricalDistribution([0.01, 0.05, 0.1, 0.2]),
        "beta": CategoricalDistribution([0.01, 0.05, 0.1]),
    }


def params_var_suprb(DX):
    return {
        "rd_mutation_sigma": FloatDistribution(0, np.sqrt(DX)),
        "rd_delay": IntDistribution(10, 100),
        "rd_init_fitness_alpha": FloatDistribution(0.01, 0.2),
        "sc_selection": CategoricalDistribution(
            [
                # Consider to add RouletteWheel multiple times to balance/mimic
                # SupRB paper's tuning?
                # ("RouletteWheel", {}),
                # ("RouletteWheel", {}),
                # ("RouletteWheel", {}),
                ("RouletteWheel", {}),
                ("Tournament", {"k": 3}),
                ("Tournament", {"k": 5}),
                ("Tournament", {"k": 7}),
                ("Tournament", {"k": 10}),
            ]
        ),
        "sc_crossover": CategoricalDistribution(
            [
                ("NPoint", {"n": 1}),
                ("NPoint", {"n": 2}),
                ("NPoint", {"n": 4}),
                ("NPoint", {"n": 7}),
                ("NPoint", {"n": 10}),
                ("Uniform", {}),
                # Consider to add Uniform multiple times to balance/mimic SupRB
                # paper's tuning?
                # ("Uniform", {}),
                # ("Uniform", {}),
                # ("Uniform", {}),
                # ("Uniform", {}),
            ]
        ),
        "sc_mutation_rate": FloatDistribution(0, 0.1),
    }


def make_xcsf_triple(DX, n_pop_size, n_train, seed=0):
    return (
        f"XCSF{n_pop_size}",
        XCS().set_params(
            **params_xcsf(
                DX=DX,
                n_pop_size=n_pop_size,
                n_train=n_train,
                seed=seed,
            )
        ),
        params_var_xcsf(DX=DX, n_pop_size=n_pop_size),
    )


def models(DX, n_train):
    return [
        (
            "SupRB",
            SupRB(),
            params_var_suprb(DX),
        ),
        (
            "RandomForestRegressor30",
            RandomForestRegressor(n_estimators=30),
            # TODO Sensible vals here
            # TODO Use above DT defaults
            {"max_depth": IntDistribution(2, 5)},
        ),
        (
            "DecisionTreeRegressor",
            DecisionTreeRegressor(),
            # TODO Sensible vals here
            # TODO Use above DT defaults
            {"max_depth": IntDistribution(2, 5)},
        ),
        make_xcsf_triple(DX=DX, n_pop_size=50, n_train=n_train),
        # make_xcsf_triple(DX=DX, n_pop_size=100, n_train=n_train),
        # make_xcsf_triple(DX=DX, n_pop_size=200, n_train=n_train),
        # make_xcsf_triple(DX=DX, n_pop_size=400, n_train=n_train),
        # make_xcsf_triple(DX=DX, n_pop_size=800, n_train=n_train),
        # ("Ridge", Ridge(), {"alpha": FloatDistribution(0.0, 1.0)}),
        # (
        #     "KNeighborsRegressor",
        #     KNeighborsRegressor(),
        #     {
        #         "n_neighbors": IntDistribution(1, 10),
        #         "weights": CategoricalDistribution(
        #             ["uniform", "distance"]
        #         ),
        #     },
        # ),
    ]


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

    ctx.obj["npzfile"] = npzfile
    ctx.obj["data"] = data
    ctx.obj["X"] = X
    ctx.obj["y"] = y
    ctx.obj["X_test"] = X_test
    ctx.obj["y_test"] = y_test
    ctx.obj["DX"] = DX
    ctx.obj["N"] = N
    ctx.obj["sha256"] = file_digest(npzfile)
    ctx.obj["hash"] = data["hash"]


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
    N = ctx.obj["N"]
    sha256 = ctx.obj["sha256"]
    hash_data = ctx.obj["hash"]

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

    ms = models(DX=DX, n_train=len(X))

    for label, model, params in ms:
        print(f'Setting run name to "{run_name}".')
        with mlflow.start_run(run_name=run_name) as run:
            print(f"Run ID is {run.info.run_id}.")

            try:
                if type(model) == XCS:
                    # Unsetting XCSF's RNG seed corresponds to setting it to 0.
                    model.set_params(random_state=0)
                else:
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
                    "data.hash": hash_data,
                    "data.N": N,
                    "data.DX": DX,
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
    N = ctx.obj["N"]
    sha256 = ctx.obj["sha256"]
    hash_data = ctx.obj["hash"]

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

    ms = models(DX=DX, n_train=N)

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
                    "data.hash": hash_data,
                    "data.N": N,
                    "data.DX": DX,
                }
            )

            mlflow.log_params(best_params)

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

            try:
                print(f"Trying to store rules created by {label} …")
                estimator_inner = estimator[1].regressor_
                lowers, uppers = estimator_inner.bounds_(
                    X_min=X_MIN, X_max=X_MAX, transformer_X=estimator[0]
                )
                store.log_arrays("bounds", lowers=lowers, uppers=uppers)
                print(f"Stored rules created by {label}.")
            except AttributeError:
                print(f"{label} did not create rules, skipping rule storing …")

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
