# Experiment runner and evaluation script for the 2023 ECXAI paper


Used for the paper *Towards Principled Synthetic Benchmarks for Explainable Rule
Set Learning Algorithms* presented at the *Evolutionary Computing and
Explainable Artificial Intelligence* (ECXAI) workshop taking place as part of
the 2023 GECCO conference.


# Running experiments


Assuming that the data lies in `$DATA/` and is split into a single initial
dataset `$DATA/….npz` and the remaining datasets which lie in a subdirectory
`$DATA/rest/`.  Adjust paths as necessary.


First, enter a development shell at the root of this repository.

```
nix develop
```


Then, run a single set without parallelization (due to mlflow not behaving well
if many things try to initialize an experiment at once):

```
export DATA=~/data-10N
export exp_name="runmany-with-proper-score"
export PYTHONPATH="src:$PYTHONPATH"
python scripts/2023-ecxai/run.py runmany --experiment-name="$exp_name" --startseed=0 --endseed=19 "$DATA"/rsl-K5-DX1-N300-0.npz
```


Finally, run the remaining sets in parallel (use `-print0` and `-0` in order to
cope with spaces in file names). We make XCSF parallelize internally by a factor
of 8 via `--n-threads` so we set the number of jobs to `1/8 = 12.5%` of the
number of cores):

```
find "$DATA"/rest -name '*.npz' -print0 | parallel -0 --jobs 12.5% --progress --eta python run.py runmany --n-threads=8 --experiment-name="$exp_name" --startseed=0 --endseed=19 '{}'
```


# Generating tables/figures


## Generating data set summary statistics (Table 1 of the ECXAI paper)


Assuming that the data set NPZ files lie *flatly* (i.e. no subdirectories) in
`$DATA/`.


First, enter a development shell at the root of this repository.

```
nix develop
```

Then, to generate Table 1 of the paper (a summary statistics of the used data
sets):


```
export PYTHONPATH="src:$PYTHONPATH"
python scripts/2023-ecxai/eval.py datasets $DATA/
```


## Generating a point plot comparing test MSEs of XCSF with UBR and CSR matching function encodings (Figure 1 of the ECXAI paper)


Assuming that when performing the runs (see above) the mlflow tracking URL was
`$tracking_uri` and the experiment name `$exp_name`:

```
export PYTHONPATH="src:$PYTHONPATH"
python scripts/2023-ecxai/eval.py eval --exp-name="$exp_name" --tracking-uri="$tracking_uri" mses-per-task
```
