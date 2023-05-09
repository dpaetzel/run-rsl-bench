# eval.py analyses the results recorded by run.py
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
import os
import re

import arviz as az
import click
import cmpbayes
import json
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns

from tqdm import tqdm

import store

sns.set_palette("colorblind")
pd.options.display.max_rows = 10000
tqdm.pandas()

pretty = {
    "params.data.DX": "$\mathcal{D}_\mathcal{X}$",
    "params.data.K": "$K$",
    "params.data.seed": "Data Seed"
}

import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


# https://github.com/mwaskom/seaborn/issues/915#issuecomment-971204836
def fixed_boxplot(x, y, *args, label=None, **kwargs):
    """
    sns.boxplot usable with sns.FacetGrid.
    """
    sns.boxplot(x=x, y=y, *args, **kwargs, labels=[label])


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

    # A more space saving representation.
    means = means[["linear_model_mse", "rsl_model_mse"
                   ]].reset_index().set_index(["DX", "N", "K"]).unstack("K")
    print(means.to_latex())

    import IPython
    IPython.embed(banner1="")
    import sys
    sys.exit(1)
    # consider running `globals().update(locals())` in the shell to fix not being
    # able to put scopes around variables


@cli.group()
@click.pass_context
@click.option("--tracking-uri", type=str, default="mlruns")
@click.option("--exp-name", type=str, default="runmany")
def eval(ctx, tracking_uri, exp_name):
    ctx.ensure_object(dict)

    print(f"Setting mlflow tracking URI to {tracking_uri} …")
    mlflow.set_tracking_uri(tracking_uri)

    print(f"Loading runs …")
    df = mlflow.search_runs(experiment_names=[exp_name])

    df = df[df.status == "FINISHED"]

    df["params.data.K"] = df["params.data.K"].apply(int)
    df["params.data.DX"] = df["params.data.DX"].apply(int)
    df["params.data.N"] = df["params.data.N"].apply(int)
    df["params.data.linear_model_mae"] = df[
        "params.data.linear_model_mae"].apply(float)
    df["params.data.linear_model_mse"] = df[
        "params.data.linear_model_mse"].apply(float)
    df["params.data.linear_model_rsquared"] = df[
        "params.data.linear_model_rsquared"].apply(float)
    df["params.data.rsl_model_mae"] = df["params.data.rsl_model_mae"].apply(
        float)
    df["params.data.rsl_model_mse"] = df["params.data.rsl_model_mse"].apply(
        float)
    df["params.data.rsl_model_rsquared"] = df[
        "params.data.rsl_model_rsquared"].apply(float)
    df["params.pop_size"] = df["params.pop_size"].apply(int)

    df["metrics.scores.ubr"] = df["artifact_uri"].apply(
        store.load_array("ubr", "scores"))
    df["metrics.scores.csr"] = df["artifact_uri"].apply(
        store.load_array("csr", "scores"))

    regex = re.compile(r"^.*/rsl-.*-.*-.*-(.*)\.npz")

    def data_seed(fname):
        return int(regex.match(fname)[1])

    df["params.data.seed"] = df["params.data.fname"].apply(data_seed)

    df = df.set_index("run_id")

    ctx.obj["df"] = df
    print(f"Sucessfully loaded {len(df)} runs with FINISHED status.")


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
        raise NotImplementedError("Use load_array instead of get_results here")
        exps_ubr = get_results("ubr", "experiences")(run["artifact_uri"])
        exps_csr = get_results("csr", "experiences")(run["artifact_uri"])
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
def durations(ctx):
    df = ctx.obj["df"]

    # Analyse run durations.
    df["duration"] = (df.end_time
                      - df.start_time).apply(lambda x: x.total_seconds())
    df["duration_min"] = df["duration"] / 60
    g = sns.FacetGrid(data=df, col="params.data.DX")
    g.map(sns.histplot, "duration_min")
    plt.show()


@eval.command()
@click.pass_context
def scores_per_task(ctx):
    df = ctx.obj["df"]

    index = ["params.data.DX", "params.data.K", "params.data.seed"]

    variants = ["ubr", "csr"]

    df = df.set_index(index)

    df_metrics = pd.DataFrame()
    for var in variants:
        df_metrics[f"metrics.scores.max.{var}"] = df[
            f"metrics.scores.{var}"].apply(max)
        df_metrics[f"metrics.scores.min.{var}"] = df[
            f"metrics.scores.{var}"].apply(min)
        df_metrics[f"metrics.scores.mean.{var}"] = df[
            f"metrics.scores.{var}"].apply(np.mean)
        df_metrics[f"metrics.scores.median.{var}"] = df[
            f"metrics.scores.{var}"].apply(np.median)

    new_columns = [
        tuple(col.rsplit(".", maxsplit=1)) for col in df_metrics.columns
    ]
    df_metrics.columns = pd.MultiIndex.from_tuples(new_columns)

    df_metrics = df_metrics.stack().stack().reset_index().rename(
        columns={
            "level_3": "Algorithm",
            "level_4": "Metric",
            0: "Value"
        } | pretty)

    g = sns.FacetGrid(
        data=df_metrics[df_metrics["Metric"] == metric],
        col=pretty["params.data.DX"],
        row=pretty["params.data.K"],
        hue="Algorithm",
        hue_order=["ubr", "csr"],
        sharey=False,
        margin_titles=True)

    for m in ["max", "median", "min"]:
        metric = f"metrics.scores.{m}"
        g.data = df_metrics[df_metrics["Metric"] == metric]

        g.map(
            sns.pointplot,
            pretty["params.data.seed"],
            "Value",
            order=np.sort(df_metrics[pretty["params.data.seed"]].unique()),
            errorbar=("ci", 95),
            capsize=0.3,
            errwidth=2.0,
        )

    g.add_legend()
    plt.savefig("plots/eval/scores-per-task.pdf")
    plt.show()

@eval.command()
@click.pass_context
def mses_per_task(ctx):
    df = ctx.obj["df"]

    index = ["params.data.DX", "params.data.K", "params.data.seed"]
    metrics = ["metrics.mse.test.csr", "metrics.mse.test.ubr"]
    df_metrics = df.reset_index()[index + metrics]
    df_metrics = df_metrics.set_index(index).stack().reset_index().rename(
        columns={
            0: "Test MSE",
            "level_3": "Algorithm",
        } | pretty)
    labels = {
        "metrics.mse.test.ubr": "UBR",
        "metrics.mse.test.csr": "CSR",
    }
    df_metrics["Algorithm"] = df_metrics["Algorithm"].apply(
        lambda s: labels[s])
    g = sns.FacetGrid(data=df_metrics,
                      col=pretty["params.data.DX"],
                      row=pretty["params.data.K"],
                      hue="Algorithm",
                      hue_order=list(labels.values()),
                      sharey=False,
                      margin_titles=True)
    g.map(
        sns.pointplot,
        pretty["params.data.seed"],
        "Test MSE",
        order=np.sort(df_metrics[pretty["params.data.seed"]].unique()),
        errorbar=("ci", 95),
        capsize=0.3,
        errwidth=2.0,
    )
    g.add_legend()
    plt.savefig("plots/eval/mses-per-task.pdf")
    plt.show()


@eval.command()
@click.pass_context
def mses_per_task_with_lin(ctx):
    df = ctx.obj["df"]

    index = ["params.data.DX", "params.data.K", "params.data.seed"]
    metrics = [
        "metrics.mse.test.csr", "metrics.mse.test.ubr",
        "params.data.linear_model_mse"
    ]
    df_metrics = df.reset_index()[index + metrics]
    df_metrics = df_metrics.set_index(index).stack().reset_index().rename(
        columns={
            0: "Test MSE",
            "level_3": "Algorithm",
        } | pretty)
    labels = {
        "metrics.mse.test.ubr": "UBR",
        "metrics.mse.test.csr": "CSR",
        "params.data.linear_model_mse": "Linear"
    }
    df_metrics["Algorithm"] = df_metrics["Algorithm"].apply(
        lambda s: labels[s])
    g = sns.FacetGrid(data=df_metrics,
                      col=pretty["params.data.DX"],
                      row=pretty["params.data.K"],
                      hue="Algorithm",
                      hue_order=list(labels.values()),
                      sharey=False,
                      margin_titles=True)
    g.map(
        sns.pointplot,
        pretty["params.data.seed"],
        "Test MSE",
        order=np.sort(df_metrics[pretty["params.data.seed"]].unique()),
        errorbar=("ci", 95),
        capsize=0.3,
        errwidth=2.0,
    )
    g.add_legend()
    plt.savefig("plots/eval/mses-per-task-with-lin.pdf")
    plt.show()

    import IPython
    IPython.embed(banner1="")
    import sys
    sys.exit(1)
    # consider running `globals().update(locals())` in the shell to fix not being
    # able to put scopes around variables


@eval.command()
@click.pass_context
def variances(ctx):
    df = ctx.obj["df"]

    index = ["params.data.DX", "params.data.K", "params.data.seed"]
    metrics = ["metrics.mse.test.csr", "metrics.mse.test.ubr"]
    std_per_task = df.groupby(index).std()[metrics]
    std_per_task = std_per_task.stack().reset_index().rename(
        columns={
            0: "Std of OOS MSEs",
            "level_3": "Algorithm"
        })
    std_per_task["Algorithm"] = std_per_task["Algorithm"].apply(
        lambda s: s.replace("metrics.mse.test.", "").upper())
    print("WARNING: The following values are not normalized!")
    print(
        f"Min OOS MSE std per task:\n{std_per_task.groupby('Algorithm').min()}"
    )
    print(
        f"Max OOS MSE std per task:\n{std_per_task.groupby('Algorithm').max()}"
    )

    sns.histplot(data=std_per_task,
                 x="Std of OOS MSEs",
                 hue="Algorithm",
                 element="step",
                 cumulative=True,
                 fill=False)
    plt.show()

    # Plot per (DX,K) the histogram of stds within each task for that (DX,K)
    # pair. I.e. how much the algorithm seed matters for MSE.
    g = sns.FacetGrid(data=std_per_task,
                      col="params.data.DX",
                      row="params.data.K",
                      hue="Algorithm",
                      sharey=False,
                      sharex=False,
                      margin_titles=True)
    g.map(
        sns.histplot,
        "Std of OOS MSEs",
        element="step",
        bins=50,
        # element="poly",
        cumulative=True,
        fill=False)
    #       stat="density")
    # g.map(sns.kdeplot,
    #       "Std of OOS MSEs",
    #       # element="step",
    #       cumulative=True,
    #       fill=False,
    #       cut=0)
    g.add_legend()
    plt.savefig("plots/eval/variances-std-per-task.pdf")
    plt.show()

    import IPython
    IPython.embed(banner1="")
    import sys
    sys.exit(1)
    # consider running `globals().update(locals())` in the shell to fix not being
    # able to put scopes around variables


@eval.command()
@click.pass_context
def interactive(ctx):
    df = ctx.obj["df"]

    import IPython
    IPython.embed(banner1="")
    import sys
    sys.exit(1)
    # consider running `globals().update(locals())` in the shell to fix not being
    # able to put scopes around variables


if __name__ == "__main__":
    cli()
