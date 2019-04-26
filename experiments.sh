#!/bin/bash -x

iterations=15
models="ibm1 ibm2 jump"

# TODO: ensure logs not just show last result

for model in $models; do
    python main.py --iterations=$iterations --model=$model --sampling_method=uniform
done

for model in $models; do
    python main.py --iterations=$iterations --model=$model --sampling_method=uniform --lower
done

for model in $models; do
    python main.py --iterations=$iterations --model=$model --sampling_method=uniform --probabilities=ibm1-uniform.pkl
done

# TODO: ensure ibm2 doesn't override random/seed

for seed in "1 2 3"; do
    for model in "ibm2 jump"; do
        python main.py --iterations=$iterations --model=$model --sampling_method=random --seed=$seed
    done
done
