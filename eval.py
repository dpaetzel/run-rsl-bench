import click
import matplotlib.pyplot as plt
import mlflow
import numpy as np


@click.group()
def cli():
    pass


@cli.command()
@click.option("--tracking-uri", type=str, default="mlruns")
def eval(tracking_uri):
    mlflow.set_tracking_uri(tracking_uri)

    # df = mlflow.search_runs(experiment_names=["test"])
    df = mlflow.search_runs(experiment_names=["runmany"])

    def get_results(uri):
        return np.load(uri + "/results.npz", allow_pickle=True)


    def plt_group(df):
        n_runs = len(df)
        scores_pool = []
        scores2_pool = []
        # fig, ax = plt.subplots(n_runs, 2, layout="constrained")
        for i in range(n_runs):
            run = df.iloc[i]
            results = get_results(run["artifact_uri"])
            scores = results["scores"]
            scores2 = results["scores2"]
            for score in scores:
                scores_pool.append(score)
            for score in scores2:
                scores2_pool.append(score)

        #     ax[i, 0].hist(scores, bins=50)
        #     ax[i, 1].hist(scores2, bins=50)

        fig, ax = plt.subplots(1, layout="constrained")
        ax.hist(scores_pool,
                bins=100,
                cumulative=True,
                histtype="step",
                density=True,
                label="no compact")
        ax.hist(scores2_pool,
                bins=100,
                cumulative=True,
                histtype="step",
                density=True,
                label="exp compact")
        ax.legend()
        plt.show()

    groups = df.groupby("params.data.fname").apply(plt_group)

    # consider running `globals().update(locals())` in the shell to fix not being
    # able to put scopes around variables


if __name__ == "__main__":
    cli()
