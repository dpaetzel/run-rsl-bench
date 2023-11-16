# Experiment runner and evaluation script


# Building the environment


Build (and quickly test) the environment so that it is not built by all the jobs
at the same time after submitting many jobs:


```
srun -w oc-compute03 nix develop --command hostname
srun -w oc-compute03 nix develop --command python -c 'import mlflow; import numpy; import optuna'
```


# Performing hyperparameter search without Slurm


```
PYTHONPATH=src:$PYTHONPATH python scripts/2024-evostar/submit-baremetal.py optparams ../RSLModels.jl/2023-10-20T17-01-56.409-task-selection/
```


# Submitting hyperparameter search to Slurm


To submit hyperparameter search to the Slurm cluster with the default number of
threads/CPUs (check the code, should be 4 but I might have changed that) and an
optimization time of 1 h per algorithm per learning task you should run
something like:


```
nix develop .#devShell.submit --command python scripts/2024-evostar/submit.py optparams --timeout=1800 ../RSLModels.jl/2023-11-10T19-01-45.606-task-selection
```


# Submitting repeated runs of the best-performing hyperparameter configurations to Slurm


To submit repeated runs of the best-performing hyperparameter configurations the
Slurm cluster with the default number of threads/CPUs (check the code, should be
4 but I might have changed that), you should run something like (replacing the
tuning URI with the path to the mlruns folder created by a prior tuning run):


```
nix develop .#devShell.submit --command python scripts/2024-evostar/submit.py runbest ../RSLModels.jl/2023-11-10T19-01-45.606-task-selection --tuning-uri results/2023-11-10T19:41:30.234918-evostar-optparams --n-reps=30 --seed-start=0
```
