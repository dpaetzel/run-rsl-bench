import os
import re

import arviz as az
import click
import cmpbayes
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_palette("colorblind")


def plot_dist(df, ax):
    sns.histplot(df, bins=50, ax=ax)
    sns.kdeplot(df, cut=0, ax=ax)


@click.group()
def cli():
    pass


@cli.command()
@click.argument("PATH")
def datasets(path):

    fnames = os.listdir(path)

    def entry(fname):
        data = np.load(f"{path}/{fname}", allow_pickle=True)
        X = data["X"]
        N, DX = X.shape
        centers = data["centers"]
        K = len(centers)

        return dict(N=N,
                    DX=DX,
                    K=K,
                    linear_model_mse=data["linear_model_mse"][()],
                    linear_model_rsquared=data["linear_model_rsquared"][()],
                    rsl_model_mse=data["rsl_model_mse"][()],
                    rsl_model_rsquared=data["rsl_model_rsquared"][()])

    index = ["DX", "K", "N"]
    df = pd.DataFrame(map(entry, fnames))

    summary = df.set_index(index).sort_index().index
    print(np.unique(summary, return_counts=True))

    fig, ax = plt.subplots(2, layout="constrained", figsize=(10, 3 * 10))
    sns.boxplot(
        data=df,
        x="DX",
        hue="K",
        y="linear_model_mse",
        ax=ax[0],
    )
    sns.boxplot(
        data=df,
        x="DX",
        hue="K",
        y="rsl_model_mse",
        ax=ax[1],
    )
    sns.stripplot(data=df,
                  x="DX",
                  hue="K",
                  y="linear_model_mse",
                  ax=ax[0],
                  dodge=True,
                  palette="dark:0",
                  size=3)
    sns.stripplot(data=df,
                  x="DX",
                  hue="K",
                  y="rsl_model_mse",
                  ax=ax[1],
                  dodge=True,
                  palette="dark:0",
                  size=3)
    plt.show()

    means = df.groupby(index).mean().round(2)
    n_datasets = df.groupby(index).apply(len).unique()
    print(f"Mean of the {n_datasets} datasets per {index} combination")
    print(means[["linear_model_mse", "rsl_model_mse"]].to_latex())

    import IPython
    IPython.embed(banner1="")
    import sys
    sys.exit(1)
    # consider running `globals().update(locals())` in the shell to fix not being
    # able to put scopes around variables


@cli.group()
@click.pass_context
@click.option("--tracking-uri", type=str, default="mlruns")
def eval(ctx, tracking_uri):
    ctx.ensure_object(dict)

    print(f"Setting mlflow tracking URI to {tracking_uri} …")
    mlflow.set_tracking_uri(tracking_uri)

    print(f"Loading runs …")
    df = mlflow.search_runs(experiment_names=["runmany"])

    df = df[df.status == "FINISHED"]
    df["params.data.K"] = df["params.data.K"].apply(int)
    df["params.data.DX"] = df["params.data.DX"].apply(int)
    df["params.data.N"] = df["params.data.N"].apply(int)

    def data_seed(fname):
        return int(
            re.match(r"^.*/rsl-.*-.*-.*-(.*)\.npz", fname)[1])
    df["params.data.seed"] = df["params.data.fname"].apply(data_seed)

    df = df.set_index("run_id")

    df["scores_ubr"] = df["artifact_uri"].apply(get_field("ubr", "scores"))
    df["scores_csr"] = df["artifact_uri"].apply(get_field("csr", "scores"))

    df["scores_ubr_median"] = df["scores_ubr"].apply(np.median)
    df["scores_csr_median"] = df["scores_csr"].apply(np.median)
    df["scores_ubr_mean"] = df["scores_ubr"].apply(np.mean)
    df["scores_csr_mean"] = df["scores_csr"].apply(np.mean)

    ctx.obj["df"] = df
    print(f"Sucessfully loaded {len(df)} runs with FINISHED status.")


def get_results(label):

    def _get_results(uri):
        return np.load(uri + f"/results.{label}.npz", allow_pickle=True)

    return _get_results


def get_field(label, field):

    def _get_results(uri):
        tracking_uri = mlflow.get_tracking_uri()
        assert tracking_uri.endswith("/mlruns"), (
            "Valid tracking URIs should "
            "have the suffix \"/mlruns\"")
        assert uri.startswith("mlruns/")

        data = np.load(tracking_uri.removesuffix("mlruns") + uri
                       + f"/results.{label}.npz",
                       allow_pickle=True)
        out = data[field]
        data.close()
        return out

    return _get_results


@eval.command()
@click.pass_context
def hists_scores_pooled(ctx):
    """
    Pool *all* the scores (one score per rule in the final population of each
    run, i.e. `pop_size` scores per run) for ubr as well as csr and plot
    histograms.

    Disregard input dimension, number of components etc. of the learning tasks.
    Probably not that helpful.
    """
    df = ctx.obj["df"]

    n_runs = len(df)
    scores_pool_ubr = []
    scores_pool_csr = []
    # fig, ax = plt.subplots(n_runs, 2, layout="constrained")
    for i in range(n_runs):
        run = df.iloc[i]
        results_ubr = get_results("ubr")(run["artifact_uri"])
        results_csr = get_results("csr")(run["artifact_uri"])
        scores_ubr = get_field("ubr", "scores")(run["artifact_uri"])
        scores_csr = get_field("csr", "scores")(run["artifact_uri"])
        for score in scores_ubr:
            scores_pool_ubr.append(score)
        for score in scores_csr:
            scores_pool_csr.append(score)

    fig, ax = plt.subplots(1, layout="constrained")
    ax.hist(scores_pool_ubr,
            bins=100,
            cumulative=True,
            histtype="step",
            density=True,
            label="ubr")
    ax.hist(scores_pool_csr,
            bins=100,
            cumulative=True,
            histtype="step",
            density=True,
            label="csr")
    ax.legend()
    fig.savefig("plots/eval/hists-scores-pooled.pdf")
    plt.show()


@eval.command()
@click.pass_context
def hists_scores_stats(ctx):
    """
    For each run, compute a single score by computing a statistic over the
    scores of the final population. Plot histograms of these scores for csr as
    well as ubr.
    """
    df = ctx.obj["df"]

    fig, ax = plt.subplots(2, layout="constrained")
    ax[0].hist(df["scores_ubr_median"],
               bins=50,
               density=True,
               histtype="step",
               label="ubr",
               cumulative=True)
    ax[0].hist(df["scores_csr_median"],
               bins=50,
               density=True,
               histtype="step",
               label="csr",
               cumulative=True)
    ax[0].legend()
    ax[0].set_title("Distribution of median score of final population")
    ax[1].hist(df["scores_ubr_mean"],
               bins=50,
               density=True,
               histtype="step",
               label="ubr",
               cumulative=True)
    ax[1].hist(df["scores_csr_mean"],
               bins=50,
               density=True,
               histtype="step",
               label="csr",
               cumulative=True)
    ax[1].legend()
    ax[1].set_title("Distribution of mean score of final population")
    fig.savefig("plots/eval/hists-scores-stats.pdf")
    plt.show()


@eval.command()
@click.pass_context
def hists_mses_pooled(ctx):
    """
    Pool *all* the out-of-sample MSEs (one MSE per run) for ubr as well as csr
    and plot histograms.

    Disregard input dimension, number of components etc. of the learning tasks.
    Probably not that helpful.
    """
    df = ctx.obj["df"]

    fig, ax = plt.subplots(2, layout="constrained")
    ax[0].hist(
        [df["metrics.mse.test.ubr"], df["metrics.mse.test.csr"]],
        data=df,
        bins=50,
        density=True,
        label="csr",
    )
    ax[0].legend()
    ax[0].set_title("Distribution of MSEs on hold out test data")
    ax[1].hist(df["metrics.mse.test.ubr"],
               bins=50,
               density=True,
               histtype="step",
               label="ubr",
               cumulative=True)
    ax[1].hist(df["metrics.mse.test.csr"],
               bins=50,
               density=True,
               histtype="step",
               label="csr",
               cumulative=True)
    ax[1].legend()
    ax[1].set_title(
        "Cumulative representation of the distribution of MSEs on hold out test data"
    )
    fig.savefig("plots/eval/hists_mses_pooled.pdf")
    plt.show()


@eval.command()
@click.pass_context
def withintasks(ctx):
    """
    Very preliminary analysis of MSE variance within a single task vs. variance
    within a K/DX configuration.

    TODO Analyse this further using BDA
    TODO Also analyse variance in *scores*
    """
    df = ctx.obj["df"]

    g = sns.FacetGrid(data=df,
                      row="params.data.DX",
                      col="params.data.K",
                      hue="params.data.fname")
    g.map(sns.swarmplot, "metrics.mse.test.csr")
    plt.show()

    g = sns.FacetGrid(data=df,
                      row="params.data.DX",
                      col="params.data.K",
                      hue="params.data.fname")
    g.map(sns.swarmplot, "metrics.mse.test.ubr")
    plt.show()

    index = ["params.data.DX", "params.data.K", "params.data.fname"]
    pertask = df.groupby(index)["metrics.mse.test.csr"].var()
    perDXK = df.groupby(["params.data.DX",
                         "params.data.K"])["metrics.mse.test.csr"].var()
    print(
        "TODO Compare pertask and perDXK and also include mean in that analysis, probably using BDA"
    )

    import IPython
    IPython.embed(banner1="")
    import sys
    sys.exit(1)
    # consider running `globals().update(locals())` in the shell to fix not being
    # able to put scopes around variables


@eval.command()
@click.pass_context
def sanitycheck(ctx):
    """
    Perform a quick sanity check of the run data. This is meant as a kind of
    test suite for run data that captures problems that I encountered earlier.
    """
    df = ctx.obj["df"]

    for i in range(len(df)):
        run = df.iloc[i]
        exps_ubr = get_field("ubr", "experiences")(run["artifact_uri"])
        exps_csr = get_field("csr", "experiences")(run["artifact_uri"])
        assert np.sum(exps_ubr) > 0
        assert np.sum(exps_csr) > 0



@eval.command()
@click.pass_context
def mean_mse_tendencies(ctx):
    """
    Plot rough tendencies (means) of out-of-sample MSE behaviour with respect to
    altering `K` and `DX` for ubr and csr.
    """
    df = ctx.obj["df"]

    fig, ax = plt.subplots(2, layout="constrained")
    sns.pointplot(data=df,
                  x="params.data.K",
                  y="metrics.mse.test.ubr",
                  hue="params.data.DX",
                  ax=ax[0])
    sns.pointplot(data=df,
                  x="params.data.K",
                  y="metrics.mse.test.csr",
                  hue="params.data.DX",
                  ax=ax[1])
    # TODO Plot linear model errors here as well
    ax[0].set_title("metrics.mse.test.ubr")
    ax[1].set_title("metrics.mse.test.csr")
    fig.savefig("plots/eval/mean-mse-tendencies.pdf")
    plt.show()


@eval.command()
@click.pass_context
def all(ctx):
    df = ctx.obj["df"]

    print(az.summary(model.infdata_))

    diff = model.infdata_.posterior.mean2_minus_mean1.to_numpy().ravel()

    rope = 0.01

    probs = {
        "p(MSE_test(csr) > MSE_test(ubr))" : np.sum(rope < diff) / len(diff),
        "p(MSE_test(ubr) > MSE_test(csr))" : np.sum(diff < -rope) / len(diff),
        "p(MSE_test(csr) = MSE_test(ubr))" : np.sum((-rope <= diff) & (diff <= rope)) / len(diff),
    }
    print(probs)

    diff = np.sort(diff)
    l = diff[int(0.025 * len(diff))]
    u = diff[int(0.975 * len(diff))]
    print(f"95% HDPI of MSE_test(ubr) - MSE_test(csr): [{l:.2}, {u:.2}]")

    import IPython
    IPython.embed(banner1="")
    import sys
    sys.exit(1)
    # consider running `globals().update(locals())` in the shell to fix not being
    # able to put scopes around variables


if __name__ == "__main__":
    cli()
