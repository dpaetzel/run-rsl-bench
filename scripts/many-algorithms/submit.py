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

defaults = dict(n_iter=100000, n_threads=4, timeout=10)


@click.group()
def cli():
    pass


@cli.command()
@click.option("--n-threads",
              default=defaults["n_threads"],
              type=int,
              show_default=True,
              help="Number of threads to use while fitting XCSF")
@click.option(
    "-t",
    "--timeout",
    default=defaults["timeout"],
    type=int,
    show_default=True,
    help="Compute budget (in seconds) for the hyperparameter optimization")
@click.option("--experiment-name", type=str, default="optparams")
@click.option("--node", type=str, default="oc-compute03")
@click.option("-o",
              "--slurm-options",
              type=str,
              default=None,
              show_default=True,
              help=("Override Slurm options "
                    "(for now, see file source for defaults)"))
@click.argument("PATH")
def optparams(n_threads, timeout, experiment_name, node, slurm_options, path):
    """
    If PATH is an NPZ file, then submit a single job for that file. If PATH is a
    directory, look at its immediate contents (i.e. non-recursively) and submit
    a job for each NPZ file found.
    """
    if slurm_options is not None:
        raise NotImplementedError("Has to be implemented")

    dir_job = slurm.get_dir_job()
    dir_results = slurm.get_dir_results(dir_job)

    tracking_uri = f"{dir_results}/mlruns"

    def submit_npz(npzfile):
        command = (
            # Note that we keep `{job_dir}` to be inserted by `submit`.
            f'python {{dir_job}}/scripts/many-algorithms/run.py optparams "{npzfile}" '
            f'--tracking-uri={tracking_uri} '
            f'--experiment-name={experiment_name} '
            '--run-name=${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}} '
            f'--n-threads={n_threads} '
            f'--timeout={timeout}')
        slurm.submit(command,
                     experiment_name,
                     node=node,
                     n_cpus=n_threads,
                     mem_per_cpu="1G",
                     dir_job=dir_job,
                     dir_results=dir_results)

    print(f"Initializing mlflow experiment at tracking URI {tracking_uri} …")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.create_experiment(experiment_name)

    if os.path.isfile(path):
        submit_npz(path)
    elif os.path.isdir(path):
        for fname in os.listdir(path):
            fpath = os.path.join(path, fname)
            if fname.endswith('.npz') and os.path.isfile(fpath):
                submit_npz(fpath)
                time.sleep(0.5)


if __name__ == "__main__":
    cli()
