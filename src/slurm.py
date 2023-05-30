# Utilities to interact with Slurm.
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
import pathlib
import shutil
import tempfile
from datetime import datetime
from subprocess import PIPE, Popen


def get_dir_job():
    return os.getcwd()


def get_dir_results(job_dir):
    datetime_ = datetime.now().isoformat()
    dir_results = f"{job_dir}/results/{datetime_}"
    return dir_results


def init_dir_results(dir_results):
    os.makedirs(f"{dir_results}/output", exist_ok=True)
    os.makedirs(f"{dir_results}/jobs", exist_ok=True)
    return dir_results


def submit(command,
           experiment_name,
           n_cpus=4,
           node="oc-compute03",
           mem_per_cpu="1G",
           dir_job=None,
           dir_results=None):
    """
    Parameters
    ----------
    command : str
        A format string that may use the fields `{dir_job}` and `{dir_results}`
        (just look at the code).
    """
    if dir_job is None:
        dir_job = get_dir_job()

    if dir_results is None:
        dir_results = get_dir_results(dir_job)
    init_dir_results(dir_results)

    sbatch = "\n".join([
        f'#!/usr/bin/env bash',  #
        # Default Slurm settings.
        f'#SBATCH --nodelist={node}',
        f'#SBATCH --cpus-per-task={n_cpus}',
        f'#SBATCH --time=1-00:00:00',
        f'#SBATCH --mem-per-cpu={mem_per_cpu}',
        f'#SBATCH --partition=cpu-prio',
        f'#SBATCH --output="{dir_results}/output/output-%A-%a.txt"',
        # Always use srun within sbatch.
        # https://stackoverflow.com/a/53640511/6936216
        f"srun bash -c 'echo Running on $(hostname)'",
        (
            # Don't export environment variables but run Nix in a clean
            # environment or else there will likely be problems with GLIBC
            # versions.
            f'srun --export=NONE '
            f'/run/current-system/sw/bin/nix develop "{dir_job}" --command '
            f'{command.format(**dict(dir_job=dir_job, dir_results=dir_results))}\n'
        )
    ])
    print(sbatch)
    print()

    tmp = tempfile.NamedTemporaryFile()
    with open(tmp.name, "w+") as f:
        f.write(sbatch)
    print(f"Wrote sbatch to {tmp.name}.")
    print()

    p = Popen(["sbatch", f"{tmp.name}"], stdout=PIPE, stdin=PIPE, stderr=PIPE)
    output = p.communicate()
    stdout = output[0].decode("utf-8")
    stderr = output[1].decode("utf-8")
    print(f"stdout:\n{stdout}\n")
    print(f"stderr:\n{stderr}\n")
    jobid = int(stdout.replace("Submitted batch job ", ""))
    print(f"Job ID: {jobid}")
    print()

    sbatch_dir = f"{dir_results}/jobs"
    os.makedirs(sbatch_dir, exist_ok=True)
    tmppath = pathlib.Path(tmp.name)
    fname = pathlib.Path(sbatch_dir, f"{jobid}.sbatch")
    shutil.copy(tmppath, fname)
    print(f"Renamed {tmp.name} to {fname}")
