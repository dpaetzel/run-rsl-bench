import hashlib
import os
import tempfile

import click
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import scoring
import store
from sklearn_xcsf import XCSF, bounds


def file_digest(fname):
    with open(fname, 'rb') as f:
        hash_object = hashlib.sha256()
        # Avoid loading large files into memory by reading in chunks.
        for chunk in iter(lambda: f.read(4096), b''):
            hash_object.update(chunk)
    return hash_object.hexdigest()


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


@click.group()
def cli():
    pass


defaults = dict(n_iter=100000, pop_size=200)


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
              default=defaults["n_iter"],
              type=int,
              show_default=True,
              help="Number of iterations to run the metaheuristic for")
@click.option("--pop-size",
              default=defaults["pop_size"],
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
              default=defaults["n_iter"],
              type=int,
              show_default=True,
              help="Number of iterations to run the metaheuristic for")
@click.option("--pop-size",
              default=defaults["pop_size"],
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

        print(f"RNG seed is {seed}.")
        mlflow.log_param("seed", seed)

        data = np.load(npzfile)
        mlflow.log_params({
            "data.fname":
            npzfile,
            "data.sha256": file_digest(npzfile)
            # Python 3.11 and onwards we can simply do:
            # hashlib.file_digest(npzfile, digest="sha256")
        })

        # Load and transform training data.
        X, y = get_train(data)
        scaler_X = MinMaxScaler(feature_range=(-1.0, 1.0))
        X = scaler_X.fit_transform(X).reshape(X.shape)
        scaler_y = StandardScaler()
        y = scaler_y.fit_transform(y.reshape(len(X), -1))

        # Store training data transformers.
        store.log_scalers(scaler_X=scaler_X, scaler_y=scaler_y)

        N, DX = X.shape
        mlflow.log_params({
            "data.N": N,
            "data.DX": DX,
            "pop_size": pop_size,
            "n_iter": n_iter,
            "compact": compact,
        })

        # Load and transform test data.
        X_test, y_test = get_test(data)
        X_test = scaler_X.transform(X_test)
        y_test = scaler_y.transform(y_test)

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

            print("Performing predictions on test data …")
            y_test_pred = model.predict(X_test)
            print("Performing predictions on training data …")
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

            experiences = np.array([r["experience"] for r in model.rules_])
            if np.sum(experiences) < len(X):
                print(f"WARNING: Training may have failed, the sum of "
                      f"rule experiences is {np.sum(experiences)} for "
                      f"{len(X)} training data points.")

            store.log_arrays(f"results.{label}",
                       y_pred=y_pred,
                       y_test_pred=y_test_pred,
                       scores=scores,
                       experiences=experiences)

            store.log_population(model, label)

            return y_pred, y_test_pred, scores

        model_ubr = XCSF(n_pop_size=pop_size,
                         n_iter=n_iter,
                         compaction=compact,
                         random_state=seed,
                         condition="hyperrectangle_ubr")
        model_csr = XCSF(n_pop_size=pop_size,
                         n_iter=n_iter,
                         compaction=compact,
                         random_state=seed,
                         condition="hyperrectangle_csr")
        y_pred_ubr, y_test_pred_ubr, scores_ubr = eval_model(model_ubr, "ubr")
        y_pred_csr, y_test_pred_csr, scores_csr = eval_model(model_csr, "csr")

        if DX == 1:
            # Sort test data for more straightforward prediction plotting.
            perm = np.argsort(X_test.ravel())
            X_test = X_test[perm]
            y_test = y_test[perm]
            y_test_pred_csr = y_test_pred_csr[perm]
            y_test_pred_ubr = y_test_pred_ubr[perm]

            fig, ax = plt.subplots(2, layout="constrained")
            ax[0].scatter(X_test, y_test, color="C0", marker="+")
            ax[0].plot(X_test, y_test_pred_ubr, color="C1")
            ax[0].set_title("ubr")
            ax[1].scatter(X_test, y_test, color="C0", marker="+")
            ax[1].plot(X_test, y_test_pred_csr, color="C1")
            ax[1].set_title("csr")
            log_plot(f"preds", fig)
            plt.close("all")

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
        plt.close("all")


if __name__ == "__main__":
    cli()
