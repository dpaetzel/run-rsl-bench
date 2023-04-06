import json
import os

import matplotlib.pyplot as plt
import tempfile
from sklearn.metrics import mean_squared_error
import click
import mlflow
import numpy as np
from sklearn.compose import TransformedTargetRegressor  # type: ignore
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import check_random_state

from sklearn_xcsf import XCSF, bounds


def get_train(data):
    X, y = data["X"], data["y"]
    y = y.reshape(len(X), -1)
    return X, y


def get_test(data):
    X_test = data["X_test"]
    try:
        y_test = data["y_test_true"]
    except KeyError:
        y_test = data["y_test"]
    y_test = y_test.reshape(len(X_test), -1)
    return X_test, y_test


# Copied from berbl-exp.experiments.utils.
def log_plot(fname, fig):
    # store the figure (e.g. so we can run headless)
    fig_folder = "plots"
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)
    fig_file = f"{fig_folder}/{fname}-{mlflow.active_run().info.run_id}.pdf"
    print(f"Storing plot in {fig_file}")
    fig.savefig(fig_file)
    mlflow.log_artifact(fig_file)


# Copied from berbl.utils.
def log_arrays(artifact_name, **arrays):
    with tempfile.TemporaryDirectory(
            prefix=f"{artifact_name}-") as tempdir_name:
        fname = f"{tempdir_name}/{artifact_name}.npz"
        np.savez(fname, **arrays)
        mlflow.log_artifact(fname)


def intersection(l1, u1, l2, u2):
    """
    Computes the intersection between two intervals.

    Parameters
    ----------
    int1, int2: array
        An array of shape (2, `dimension`). I.e. `int1[0]` is the lower
        bound and `int1[1]` is the upper bound of the interval.

    Returns
    -------
    array or None
        If the intervals do not overlap, return `None`. Otherwise return the
        intersection interval.
    """
    l = np.max([l1, l2], axis=0)
    u = np.min([u1, u2], axis=0)

    if np.any(u < l):
        return None
    else:
        return l, u


def volume(l, u):
    """
    Computes the volume of the given interval.
    """
    return np.prod(u - l)


def subsethood(l1, u1, l2, u2):
    intersect = intersection(l1=l1, u1=u1, l2=l2, u2=u2)
    # If not intersecting, subsethood is 0.
    if intersect is None:
        return 0.0
    # If intersecting …
    else:
        # … and the first interval is degenerate that interval is still fully
        # contained in the second and subsethood is 1.0.
        if volume(l1, u1) == 0:
            return 1.0
        else:
            l, u = intersect
            return volume(l, u) / volume(l1, u1)


def interval_similarity_mean(l1, u1, l2, u2):
    """
    Robust interval similarity metric proposed by (Huidobro et al., 2022).
    """
    ssh1 = subsethood(l1=l1, u1=u1, l2=l2, u2=u2)
    ssh2 = subsethood(l1=l2, u1=u2, l2=l1, u2=u1)
    return (ssh1 + ssh2) / 2.0


def similarities(lowers, uppers, lowers_true, uppers_true):
    K_true = len(lowers_true)
    K = len(lowers)

    vols_overlap = np.full((K_true, K), -1.0, dtype=float)
    similarity = np.full((K_true, K), -1.0, dtype=float)
    for i in range(K):
        for j in range(K_true):
            lu = intersection(l1=lowers[i],
                              u1=uppers[i],
                              l2=lowers_true[j],
                              u2=uppers_true[j])
            if lu is None:
                vols_overlap[j, i] = 0.0
            else:
                vols_overlap[j, i] = volume(*lu)
            # Note that we do not use vols_overlap currently.

            sim = interval_similarity_mean(l1=lowers[i],
                                           u1=uppers[i],
                                           l2=lowers_true[j],
                                           u2=uppers_true[j])
            similarity[j, i] = sim
    return similarity


def score(lowers, uppers, lowers_true, uppers_true):

    similarity = similarities(lowers=lowers,
                              uppers=uppers,
                              lowers_true=lowers_true,
                              uppers_true=uppers_true)

    similarity[similarity == -1.0] = 0.0

    # The score of a solution rule is the highest similarity score value it
    # received.
    scores = np.max(similarity, axis=0)

    return scores


@click.group()
def cli():
    pass


@cli.command()
@click.option("-s",
              "--startseed",
              type=click.IntRange(min=0),
              default=0,
              show_default=True,
              help="First seed to use for initializing RNGs")
@click.option("-e",
              "--endseed",
              type=click.IntRange(min=0),
              default=9,
              show_default=True,
              help="Last seed to use for initializing RNGs")
@click.option("--n-iter",
              default=10000,
              type=int,
              show_default=True,
              help="Number of iterations to run the metaheuristic for")
@click.option("--pop-size",
              default=200,
              type=int,
              show_default=True,
              help="Population size to be used by the metaheuristic")
@click.option("--compact/--no-compact",
              default=False,
              type=bool,
              show_default=True,
              help="Whether to try to compact the final solution")
@click.option("--run-name", type=str, default=None)
@click.option("--tracking-uri", type=str, default="mlruns")
@click.option("--experiment-name", type=str, required=True)
@click.argument("NPZFILE")
@click.pass_context
def runmany(ctx, startseed, endseed, n_iter, pop_size, compact, run_name,
            tracking_uri, experiment_name, npzfile):

    for seed in range(startseed, endseed + 1):
        ctx.invoke(run,
                   seed=seed,
                   n_iter=n_iter,
                   pop_size=pop_size,
                   compact=compact,
                   run_name=run_name,
                   tracking_uri=tracking_uri,
                   experiment_name=experiment_name,
                   npzfile=npzfile)


@cli.command()
@click.option("-s",
              "--seed",
              type=click.IntRange(min=0),
              default=0,
              show_default=True,
              help="Seed to use for initializing RNGs")
@click.option("--n-iter",
              default=10000,
              type=int,
              show_default=True,
              help="Number of iterations to run the metaheuristic for")
@click.option("--pop-size",
              default=200,
              type=int,
              show_default=True,
              help="Population size to be used by the metaheuristic")
@click.option("--compact/--no-compact",
              default=False,
              type=bool,
              show_default=True,
              help="Whether to try to compact the final solution")
@click.option("--run-name", type=str, default=None)
@click.option("--tracking-uri", type=str, default="mlruns")
@click.option("--experiment-name", type=str, required=True)
@click.argument("NPZFILE")
def run(seed, n_iter, pop_size, compact, run_name, tracking_uri,
        experiment_name, npzfile):
    """
    Run XCSF on the data in NPZFILE, logging results using mlflow under the
    given EXPERIMENT_NAME and outputting plots of the found solutions.
    """
    print(f"Logging to mlflow tracking URI {tracking_uri}.")
    mlflow.set_tracking_uri(tracking_uri)

    print(f"Setting experiment name to \"{experiment_name}\".")
    mlflow.set_experiment(experiment_name)

    print(f"Setting run name to \"{run_name}\".")
    with mlflow.start_run(run_name=run_name) as run:
        print(f"Run ID is {run.info.run_id}.")

        # TODO Log data hash (i.e. npzfile hash)

        print(f"RNG seed is {seed}.")
        random_state = check_random_state(seed)
        mlflow.log_param("seed", seed)

        data = np.load(npzfile)
        mlflow.log_param("data.fname", npzfile)

        # Load train data.
        X, y = get_train(data)
        scaler_X = MinMaxScaler(feature_range=(-1, 1))
        X = scaler_X.fit_transform(X).reshape(X.shape)
        scaler_y = StandardScaler()
        y = scaler_y.fit_transform(y)

        N, DX = X.shape
        mlflow.log_params({
            "data.N": N,
            "data.DX": DX,
            "pop_size": pop_size,
            "n_iter": n_iter,
            "compact": compact,
        })

        # Load test data.
        X_test, y_test = get_test(data)
        X_test = scaler_X.transform(X_test)
        y_test = scaler_y.transform(y_test)
        # Sort test data for more straightforward prediction plotting.
        perm = np.argsort(X_test.ravel())
        X_test = X_test[perm]
        y_test = y_test[perm]

        # Load ground truth.
        centers_true = data["centers"]
        spreads_true = data["spreads"]
        lowers_true = centers_true - spreads_true
        uppers_true = centers_true + spreads_true
        K = len(centers_true)
        mlflow.log_params({
            "data.K":
            K,
            "data.linear_model_mse":
            data["linear_model_mse"],
            "data.linear_model_mae":
            data["linear_model_mae"],
            "data.linear_model_rsquared":
            data["linear_model_rsquared"],
            "data.rsl_model_mse":
            data["rsl_model_mse"],
            "data.rsl_model_mae":
            data["rsl_model_mae"],
            "data.rsl_model_rsquared":
            data["rsl_model_rsquared"],
        })

        def eval_model(model, label):
            model.fit(X, y)

            y_test_pred = model.predict(X_test)
            y_pred = model.predict(X)

            mse_test = mean_squared_error(y_test_pred, y_test)
            print(f"MSE test ({label}):", mse_test)
            mse_train = mean_squared_error(y_pred, y)
            print(f"MSE train ({label}):", mse_train)
            mlflow.log_metrics({
                f"mse.test.{label}": mse_test,
                f"mse.train.{label}": mse_train
            })

            # Score solution relative to ground truth.
            lowers, uppers = bounds(model.rules_)
            scores = score(lowers=lowers,
                           uppers=uppers,
                           lowers_true=lowers_true,
                           uppers_true=uppers_true)

            log_arrays(f"results.{label}",
                       y_pred=y_pred,
                       y_test_pred=y_test_pred,
                       scores=scores)

            return y_pred, y_test_pred, scores

        model_ubr = XCSF(n_pop_size=pop_size,
                         n_iter=n_iter,
                         compaction=compact,
                         random_state=random_state,
                         condition="hyperrectangle_ubr",
                         ea_subsumption=True)
        model_csr = XCSF(n_pop_size=pop_size,
                         n_iter=n_iter,
                         compaction=compact,
                         random_state=random_state,
                         condition="hyperrectangle_ubr",
                         ea_subsumption=True)
        y_pred_ubr, y_test_pred_ubr, scores_ubr = eval_model(model_ubr, "ubr")
        y_pred_csr, y_test_pred_csr, scores_csr = eval_model(model_csr, "csr")

        if DX == 1:
            fig, ax = plt.subplots(2, layout="constrained")
            ax[0].scatter(X_test, y_test, color="C0", marker="+")
            ax[0].plot(X_test, y_test_pred_ubr, color="C1")
            ax[0].set_title("ubr")
            ax[1].scatter(X_test, y_test, color="C0", marker="+")
            ax[1].plot(X_test, y_test_pred_csr, color="C1")
            ax[1].set_title("csr")
            log_plot(f"preds", fig)

        fig, ax = plt.subplots(1, layout="constrained")
        ax.hist(scores_ubr,
                bins=50,
                density=True,
                histtype="step",
                label="ubr",
                cumulative=True)
        ax.hist(scores_csr,
                bins=50,
                density=True,
                histtype="step",
                label="csr",
                cumulative=True)
        ax.legend()
        log_plot("hist-scores", fig)

        # 1d

        # 5d

        import IPython
        IPython.embed(banner1="")
        import sys
        sys.exit(1)
        # consider running `globals().update(locals())` in the shell to fix not being
        # able to put scopes around variables

        # TODO Initialize xcs2 properly so that we can make predictions (extract
        # init from sklearn_xcsf)


if __name__ == "__main__":
    cli()
