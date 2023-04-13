import os

import arviz as az
import click
import cmpbayes
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_palette("colorblind")


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

@click.option("--tracking-uri", type=str, default="mlruns")
def eval(tracking_uri):
    mlflow.set_tracking_uri(tracking_uri)

    df = mlflow.search_runs(experiment_names=["runmany"])
    df = df[df.status == "FINISHED"]

    def get_results(label):

        def _get_results(uri):
            return np.load(uri + f"/results.{label}.npz", allow_pickle=True)

        return _get_results

    def plt_group(df):
        n_runs = len(df)
        scores_pool_ubr = []
        scores_pool_csr = []
        # fig, ax = plt.subplots(n_runs, 2, layout="constrained")
        for i in range(n_runs):
            run = df.iloc[i]
            results_ubr = get_results("ubr")(run["artifact_uri"])
            results_csr = get_results("csr")(run["artifact_uri"])
            scores_ubr = results_ubr["scores"]
            scores_csr = results_csr["scores"]
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
        plt.show()

    df = df.set_index("run_id")

    def get_scores(label):
        return df["artifact_uri"].apply(
            get_results(label)).apply(lambda data: data["scores"])

    df["scores_ubr"] = get_scores("ubr")
    df["scores_csr"] = get_scores("csr")

    df["scores_ubr_median"] = df["scores_ubr"].apply(np.median)
    df["scores_csr_median"] = df["scores_csr"].apply(np.median)
    df["scores_ubr_mean"] = df["scores_ubr"].apply(np.mean)
    df["scores_csr_mean"] = df["scores_csr"].apply(np.mean)

    ## TODO Right now we pool over all data sets and runs. Possibly should
    ## rather pool over each data set individuall? But then that should not make
    ## a difference for these statistics (not for mean but maybe for median?!),
    ## should it?

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
    plt.show()

    model = cmpbayes.NonNegative(df["metrics.mse.test.ubr"].to_numpy(),
                                 df["metrics.mse.test.csr"].to_numpy()).fit(
                                     random_seed=1337, num_samples=10000)

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
