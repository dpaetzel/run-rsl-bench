# Experiment runner and evaluation scripts


… for TODO.


# Running experiments


Assuming that the data lies in `$DATA/` and is split into a single initial
dataset `$DATA/….npz` and the remaining datasets which lie in `$DATA/rest/`.
Adjust paths as necessary.


## Without parallelization


Works, but does not parallelize.

```bash
nix develop --command python run.py runmany --experiment-name=runmany --startseed=0 --endseed=19 "$DATA"/….npz
for npz in "$DATA"/*; do
    echo "$npz"
    nix develop --command python run.py runmany --experiment-name=runmany --startseed=0 --endseed=19 "$npz"
done
```


## With parallelization


Instead: First, enter development shell.

```
nix develop
```


Then, run a single set without parallelization (due to mlflow not behaving well
if many things try to initialize an experiment at once):

```
export DATA=~/data-10N
python run.py runmany --experiment-name=runmany --startseed=0 --endseed=19 "$DATA"/rsl-K5-DX1-N300-0.npz
```

Finally, run the remaining sets in parallel (use `-print0` and `-0` in order to
cope with spaces in file names):


```
find "$DATA"/rest -name '*.npz' -print0 | parallel -0 python run.py runmany --experiment-name=runmany --startseed=0 --endseed=19 '{}'
```
