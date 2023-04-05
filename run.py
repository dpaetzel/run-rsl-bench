import os

import json
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

from sklearn_xcsf import XCSF


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
    if intersect is None:
        return None
    else:
        l, u = intersect
        return volume(l, u) / volume(l1, u1)


def interval_similarity_mean(l1, u1, l2, u2):
    """
    Robust interval similarity metric proposed by (Huidobro et al., 2022).
    """
    ssh1 = subsethood(l1=l1, u1=u1, l2=l2, u2=u2)
    ssh2 = subsethood(l1=l2, u1=u2, l2=l1, u2=u1)
    if ssh1 is None or ssh2 is None:
        return None
    else:
        return (ssh1 + ssh2) / 2.0


def similarities(rules, lowers_true, uppers_true):
    lowers = []
    uppers = []
    for rule in rules:
        center = np.array(rule["condition"]["center"])
        spread = np.array(rule["condition"]["spread"])
        lower = center - spread
        upper = center + spread
        lowers.append(lower)
        uppers.append(upper)

    return similarities_(lowers=lowers,
                         uppers=uppers,
                         lowers_true=lowers_true,
                         uppers_true=uppers_true)


def similarities_(lowers, uppers, lowers_true, uppers_true):
    K_true = len(lowers_true)
    K = len(lowers)

    vols_overlap = np.full((K_true, K), None)
    similarity = np.full((K_true, K), None)
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


def score(rules, lowers_true, uppers_true):
    similarity = similarities(rules=rules,
                              lowers_true=lowers_true,
                              uppers_true=uppers_true)

    similarity[similarity == None] = 0.0

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

        X, y = get_train(data)

        N, DX = X.shape
        mlflow.log_param("data.N", N)
        mlflow.log_param("data.DX", DX)

        scaler_X = MinMaxScaler(feature_range=(-1, 1))
        X = scaler_X.fit_transform(X).reshape(X.shape)
        scaler_y = StandardScaler()
        y = scaler_y.fit_transform(y)

        mlflow.log_params({
            "pop_size": pop_size,
            "n_iter": n_iter,
            "compact": compact,
        })

        model = XCSF(n_pop_size=pop_size,
                     n_iter=n_iter,
                     compaction=compact,
                     random_state=random_state)
        model.fit(X, y)

        X_test, y_test = get_test(data)

        X_test = scaler_X.transform(X_test)
        y_test = scaler_y.transform(y_test)

        perm = np.argsort(X_test.ravel())
        X_test = X_test[perm]
        y_test = y_test[perm]

        y_test_pred = model.predict(X_test)
        y_pred = model.predict(X)

        mse_test = mean_squared_error(y_test_pred, y_test)
        print("MSE test:", mse_test)
        mse_train = mean_squared_error(y_pred, y)
        print("MSE train:", mse_train)
        mlflow.log_metrics({"mse.test": mse_test, "mse.train": mse_train})

        return_condition = True
        return_action = True
        return_prediction = True
        json_string = model.xcs_.json(return_condition, return_action,
                                      return_prediction)
        pop = json.loads(json_string)
        rules = pop["classifiers"]

        centers_true = data["centers"]
        spreads_true = data["spreads"]

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

        # Build a mapping that maps each index of a ground truth center to a
        # list of the indexes of solution rules whose centers are closest to it.
        mapping_norms = {i: [] for i in range(len(centers_true))}
        dists = {i: [] for i in range(len(centers_true))}
        for i, rule in enumerate(rules):
            center = np.array(rule["condition"]["center"])
            norms = np.linalg.norm(centers_true - center, axis=1)
            idx = norms.argmin()
            mapping_norms[idx].append(i)
            dists[idx].append(norms[idx])

        lowers_true = centers_true - spreads_true
        uppers_true = centers_true + spreads_true

        scores = score(rules, lowers_true=lowers_true, uppers_true=uppers_true)

        # Perform the compaction.
        min_exp = 100
        mlflow.log_param("min_exp", min_exp)

        idxs_exp = np.where(
            list(map(lambda r: r["experience"] > min_exp, rules)))
        rules2 = list(np.array(rules)[idxs_exp])

        scores2 = score(rules2,
                        lowers_true=lowers_true,
                        uppers_true=uppers_true)

        fig, ax = plt.subplots(2, layout="constrained")
        ax[0].hist(scores, bins=50)
        ax[1].hist(scores2, bins=50)
        log_plot("hist-scores", fig)

        _, DX = X_test.shape

        # Recompute metrics.
        #
        # Scores of the ground truth are expected to be 1.
        sim_true = similarities_(lowers_true, uppers_true, lowers_true,
                                 uppers_true)
        sim_true[sim_true == None] = 0.0
        assert np.all(np.max(sim_true, axis=0) == 1.0)

        random_state = check_random_state(seed)
        xcs2 = model._init_xcs(X)
        pop2 = { "classifiers" : rules2 }
        with open("pop2.json", "w") as outfile:
            json.dump(pop2, outfile)
        xcs2.json_read("pop2.json")
        y_test_pred2 = xcs2.predict(X_test)
        y_pred2 = xcs2.predict(X)

        log_arrays("results",
                   y_pred=y_pred,
                   y_test_pred=y_test_pred,
                   y_pred2=y_pred2,
                   y_test_pred2=y_test_pred2,
                   scores=scores,
                   scores2=scores2)

        mse_test2 = mean_squared_error(y_test_pred2, y_test)
        print("MSE test (modified):", mse_test2)
        mse_train2 = mean_squared_error(y_pred2, y)
        print("MSE train (modified):", mse_train2)
        mlflow.log_metrics({"mse2.test": mse_test2, "mse2.train": mse_train2})

        if DX == 1:
            plt.scatter(X_test, y_test, color="C0", marker="+")
            plt.plot(X_test, y_test_pred, color="C1")
            plt.plot(X_test, y_test_pred2, color="C2")
            log_plot("pred", fig)


        # 1d

        # 5d


        import IPython; IPython.embed(banner1=""); import sys; sys.exit(1)
        # consider running `globals().update(locals())` in the shell to fix not being
        # able to put scopes around variables


        # TODO Initialize xcs2 properly so that we can make predictions (extract
        # init from sklearn_xcsf)


if __name__ == "__main__":
    cli()
