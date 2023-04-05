#!/usr/bin/env fish


for npz in ~/Projekte/RSL-Benchmarking/rsl-benchmark/data/*
    echo "$npz"
    nix develop --command python run.py runmany --experiment-name=runmany "$npz"
end
