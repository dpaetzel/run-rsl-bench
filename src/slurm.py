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


def get_dirs():
    job_dir = os.getcwd()
    datetime_ = datetime.now().isoformat()
    results_dir = f"{job_dir}/results/{datetime_}"
    os.makedirs(f"{results_dir}/output", exist_ok=True)
    os.makedirs(f"{results_dir}/jobs", exist_ok=True)
    return job_dir, results_dir


def submit(command, experiment_name, node="oc-compute03", mem="2G"):
    """
    Parameters
    ----------
    command : str
        A format string that may use the fields `{job_dir}` and `{results_dir}`
        (just look at the code).
    """

    job_dir, results_dir = get_dirs()

    sbatch = "\n".join([
        f'#!/usr/bin/env bash',  #
        # Default Slurm settings.
        f'#SBATCH --nodelist={node}',
        f'#SBATCH --time=1-00:00:00',
        f'#SBATCH --mem={mem}',
        f'#SBATCH --partition=cpu-prio',
        f'#SBATCH --output="{results_dir}/output/output-%A-%a.txt"',
        # Always use srun within sbatch.
        # https://stackoverflow.com/a/53640511/6936216
        f"srun bash -c 'echo Running on $(hostname)'",
        (
            # Don't export environment variables but run Nix in a clean
            # environment or else there will likely be problems with GLIBC
            # versions.
            f'srun --export=NONE '
            f'/run/current-system/sw/bin/nix develop "{job_dir}" --command '
            f'{command.format(**dict(job_dir=job_dir, results_dir=results_dir))}\n'
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

    sbatch_dir = f"{results_dir}/jobs"
    os.makedirs(sbatch_dir, exist_ok=True)
    tmppath = pathlib.Path(tmp.name)
    fname = pathlib.Path(sbatch_dir, f"{jobid}.sbatch")
    shutil.copy(tmppath, fname)
    print(f"Renamed {tmp.name} to {fname}")
