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
    df["params.data.linear_model_mae"] = df["params.data.linear_model_mae"].apply(float)
    df["params.data.linear_model_mse"] = df["params.data.linear_model_mse"].apply(float)
    df["params.data.linear_model_rsquared"] = df["params.data.linear_model_rsquared"].apply(float)
    df["params.data.rsl_model_mae"] = df["params.data.rsl_model_mae"].apply(float)
    df["params.data.rsl_model_mse"] = df["params.data.rsl_model_mse"].apply(float)
    df["params.data.rsl_model_rsquared"] = df["params.data.rsl_model_rsquared"].apply(float)
    df["params.pop_size"] = df["params.pop_size"].apply(int)

    regex = re.compile(r"^.*/rsl-.*-.*-.*-(.*)\.npz")

    def data_seed(fname):
        return int(regex.match(fname)[1])

    df["params.data.seed"] = df["params.data.fname"].apply(data_seed)

    df = df.set_index("run_id")

    ctx.obj["df"] = df
    print(f"Sucessfully loaded {len(df)} runs with FINISHED status.")


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
        scores_ubr = get_results("ubr", "scores")(run["artifact_uri"])
        scores_csr = get_results("csr", "scores")(run["artifact_uri"])
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
def scores_per_task_pooled(ctx):
    """
    For each task, pool the scores of all the rules of all the runs for that
    task and plot their mean and bootstrapped CI.
    """
    df = ctx.obj["df"]

    index = ["params.data.DX", "params.data.K", "params.data.seed"]
    scores = ["scores_ubr", "scores_csr"]
    df_scores = df.reset_index()[index + scores]
    # df_scores.groupby(index)["scores_ubr"].apply(lambda x: np.concatenate(x.to_numpy()))
    df_scores_pooled = pd.DataFrame()
    df_scores_pooled["scores_ubr"] = df_scores.groupby(
        index)["scores_ubr"].apply(lambda x: np.concatenate(x.to_numpy()))
    df_scores_pooled["scores_csr"] = df_scores.groupby(
        index)["scores_csr"].apply(lambda x: np.concatenate(x.to_numpy()))

    df_scores_pooled = df_scores_pooled.stack().reset_index().rename(
        columns={
            0: "Rule Similarity Score",
            "level_3": "Algorithm",
        } | pretty)
    df_scores_pooled["Algorithm"] = df_scores_pooled["Algorithm"].apply(
        lambda s: s.replace("scores_", "").upper())
    df_scores_pooled = df_scores_pooled.explode("Rule Similarity Score")
    g = sns.FacetGrid(data=df_scores_pooled,
                      col=pretty["params.data.DX"],
                      row=pretty["params.data.K"],
                      hue="Algorithm",
                      hue_order=["UBR", "CSR"],
                      sharey=False,
                      margin_titles=True)
    g.map(
        sns.pointplot,
        pretty["params.data.seed"],
        "Rule Similarity Score",
        order=np.sort(df_scores_pooled[pretty["params.data.seed"]].unique()),
        errorbar=("ci", 95),
        capsize=0.3,
        errwidth=2.0,
    )
    g.add_legend()
    plt.savefig("plots/eval/scores-per-task-pooled.pdf")
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
def nonnegative(ctx):
    """
    Per learning task, fit `cmpbayes.NonNegative` to ubr and csr out-of-sample
    MSE data and compute the posterior of their expected difference. Based on
    that posterior, compute probabilities of one outperforming the other as well
    as practically equivalent performance (uses a rope; see code for its value).
    """
    df = ctx.obj["df"]

    def fit(group):
        try:
            model = cmpbayes.NonNegative(
                group["metrics.mse.test.ubr"].to_numpy(),
                group["metrics.mse.test.csr"].to_numpy()).fit(
                    random_seed=1337, num_samples=10000)
        except:
            print("Exception in fit(group)")
            import IPython
            IPython.embed(banner1="")
            import sys
            sys.exit(1)
            # consider running `globals().update(locals())` in the shell to fix not being
            # able to put scopes around variables

        summary = az.summary(model.infdata_)
        print(summary)

        diff = model.infdata_.posterior.mean2_minus_mean1.to_numpy().ravel()

        rope = 0.01

        probs = {
            "p(MSE_test(csr) < MSE_test(ubr))":
            np.sum(diff < -rope) / len(diff),
            "p(MSE_test(csr) = MSE_test(ubr))":
            np.sum((-rope <= diff) & (diff <= rope)) / len(diff),
            "p(MSE_test(ubr) < MSE_test(csr))":
            np.sum(rope < diff) / len(diff),
        }
        print(probs)

        diff = np.sort(diff)
        l = diff[int(0.025 * len(diff))]
        u = diff[int(0.975 * len(diff))]
        print(f"95% HDPI of MSE_test(csr) - MSE_test(ubr): [{l:.2}, {u:.2}]")

        # If I return `model` for all the groups, I run out of RAM.
        # return model, probs, l, u, diff
        return probs, l, u, diff, summary

    def explode_fit_data(df):
        df = df.rename("sample").reset_index()

        df[["probs", "l", "u", "sample",
            "summary"]] = pd.DataFrame(df["sample"].tolist(), index=df.index)
        names = list(df["probs"].iloc[0].keys())
        df[names] = pd.DataFrame(df["probs"].to_list(), index=df.index)
        return df

    index = ["params.data.DX", "params.data.K", "params.data.seed"]

    # Let's only look at DX=5 for now.
    print("WARNING: Only looking at DX=5 for now!")
    print("WARNING: Only looking at DX=5 for now!")
    df = df[df["params.data.DX"] == 5]

    diffs = df.groupby(index).apply(fit)
    diffs = explode_fit_data(diffs)
    if False:
        diffs.to_pickle("plots/eval/nonnegative-diffs-DX5.pkl")

    names = list(diffs["probs"].iloc[0].keys())
    print(diffs.set_index(index)[names].round(2))

    diffs_rounded = diffs.set_index(index)[names].round(2)
    # Sort each K-group ascendingly acc. to p(csr < ubr).
    diffs_rounded = diffs_rounded.sort_values(
        "p(MSE_test(csr) < MSE_test(ubr))").reset_index(
            2).sort_index().set_index("params.data.seed", append=True)

    import IPython
    IPython.embed(banner1="")
    import sys
    sys.exit(1)

    # consider running `globals().update(locals())` in the shell to fix not being
    # able to put scopes around variables

    def plot_probs(df, ax):
        df_ = df.filter(regex="^p\(.*").to_numpy()
        ax.matshow(df_)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["csr < ubr", "csr = ubr", "ubr < csr"])
        ax.set_yticks(df["params.data.seed"])
        ax.set_yticklabels(df["params.data.seed"])
        # https://stackoverflow.com/a/20998634
        for (i, j), z in np.ndenumerate(df_):
            ax.text(j,
                    i,
                    '{:0.2f}'.format(z),
                    ha='center',
                    va='center',
                    bbox=dict(boxstyle='round',
                              facecolor='white',
                              edgecolor='0.3'))

        return ax
        # fig.savefig("plots/eval/nonnegative-k-dx-plot-probs.pdf")
        # plt.show()

    diffs_rounded_ = diffs_rounded.reset_index()
    Ks = diffs_rounded_["params.data.K"].unique()
    fig, ax = plt.subplots(1,
                           len(Ks),
                           layout="constrained",
                           figsize=(len(Ks) * 2, 10))
    for i, K in enumerate(Ks):
        plot_probs(diffs_rounded_[diffs_rounded_["params.data.K"] == K],
                   ax=ax[i])

    plt.show()

    # g = sns.FacetGrid(data=diffs_rounded.reset_index(), col="params.data.K")
    # g.map(plot_probs, diffs_rounded.keys())
    # plt.show()
    # diffs_rounded.groupby("params.data.K").apply(plot_probs)

    # mses_mean = df.groupby().mean().filter(
    #     regex="^metrics\.mse\.test\..*")
    # ax[1].matshow(mses_mean.round(2).to_numpy())
    # ax[1].set_xticks(range(2))
    # ax[1].set_xticklabels(["ubr", "csr"])
    # ax[1].set_yticks(range(len(df.index)))
    # ax[0].set_yticklabels(
    #     list(map(lambda tpl: f"DX={tpl[0]}, K={tpl[1]}", df.index)))
    # # https://stackoverflow.com/a/20998634
    # for (i, j), z in np.ndenumerate(mses_mean):
    #     ax[1].text(j,
    #             i,
    #             '{:0.2f}'.format(z),
    #             ha='center',
    #             va='center',
    #             bbox=dict(boxstyle='round',
    #                         facecolor='white',
    #                         edgecolor='0.3'))

    samples_small = diffs.set_index(index)["sample"].apply(
        lambda s: np.random.choice(s, size=10000))
    samples_small = samples_small.explode()
    samples_small = samples_small.reset_index()
    g = sns.FacetGrid(data=samples_small,
                      row="params.data.seed",
                      col="params.data.K")
    g.map(sns.histplot, "sample")
    plt.show()

    # TODO map this pointplot over all the comparisons we made using FacetGrid
    # diffs_ = diffs.set_index(index)[names].stack()
    # diffs_ = diffs_.rename("p").reset_index().rename(
    #     columns={"level_2": "event"})
    # diffs_["DX,K"] = list(
    #     zip(diffs_["params.data.DX"], diffs_["params.data.K"]))
    # fig, ax = plt.subplots(1, layout="constrained", figsize=(10, 10))
    # plot = sns.pointplot(data=diffs_, x="event", y="p", hue="DX,K", ax=ax)
    # plot.figure.savefig("plots/eval/nonnegative-K-DX1.pdf")
    # plt.show()

    import IPython
    IPython.embed(banner1="")
    import sys
    sys.exit(1)
    # consider running `globals().update(locals())` in the shell to fix not being
    # able to put scopes around variables


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
