# Submit run.py jobs to Slurm.
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
import time

import click
import mlflow
import slurm

defaults = dict(n_iter=100000, timeout=10)


def forall_npzs(path, code, seed=None):
    """
    Parameters
    ----------
    path : str
        Path to an NPZ file or a directory where NPZ files reside in.
    code : callable
        Will be called as `code(path, seed)` if `path` is a file and `code(fpath)` for
        each NPZ file with path `fpath` under `path` if `path` is a directory.
    seed : int or None
        Seed to be used for initializing the algorithms' RNGs.
    """
    if os.path.isfile(path):
        code(path, seed)
    elif os.path.isdir(path):
        for fname in os.listdir(path):
            fpath = os.path.join(path, fname)
            if fname.endswith(".npz") and os.path.isfile(fpath):
                code(fpath, seed)
                time.sleep(0.5)


@click.group()
@click.pass_context
def cli(ctx):
    ctx.ensure_object(dict)

    dir_job = slurm.get_dir_job()
    dir_results = slurm.get_dir_results(dir_job)

    tracking_uri = f"{dir_results}/mlruns"

    ctx.obj["dir_job"] = dir_job
    ctx.obj["dir_results"] = dir_results
    ctx.obj["tracking_uri"] = tracking_uri


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
    type=click.IntRange(min=0),
    show_default=True,
    default=None,
    help="Seed for initializing RNGs",
)
@click.option("--experiment-name", type=str, default="optparams")
@click.option("--node", type=str, default="oc-compute03")
@click.option(
    "-o",
    "--slurm-options",
    type=str,
    default=None,
    show_default=True,
    help=("Override Slurm options (for now, see file source for defaults)"),
)
@click.argument("PATH")
@click.pass_context
def optparams(ctx, timeout, seed, experiment_name, node, slurm_options, path):
    """
    If PATH is an NPZ file, then submit a single job for that file. If PATH is a
    directory, look at its immediate contents (i.e. non-recursively) and submit
    a job for each NPZ file found.
    """
    if slurm_options is not None:
        raise NotImplementedError("Has to be implemented")

    dir_job = ctx.obj["dir_job"]
    dir_results = ctx.obj["dir_results"]
    tracking_uri = ctx.obj["tracking_uri"]

    print(f"Initializing mlflow experiment at tracking URI {tracking_uri} …")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.create_experiment(experiment_name)

    def submit_npz(npzfile, seed):
        command = (
            # Note that we keep `{job_dir}` to be inserted by `submit`.
            f'python {{dir_job}}/scripts/many-algorithms/run.py "{npzfile}" optparams '
            f"--tracking-uri={tracking_uri} "
            f"--experiment-name={experiment_name} "
            "--run-name=${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}} "
            f"--timeout={timeout} "
            f"{'' if seed is None else f'--seed={seed}'} "
        )
        slurm.submit(
            command,
            experiment_name,
            node=node,
            # Optuna cannot parallelize properly to multiple CPUs (due to GIL)
            # but many sklearn estimators actually can. Also, we don't want to
            # run too many jobs at the same time since we may run into IO
            # issues.
            n_cpus=4,
            mem_per_cpu="1G",
            dir_job=dir_job,
            dir_results=dir_results,
        )

    forall_npzs(path, submit_npz, seed=seed)


if __name__ == "__main__":
    cli()
