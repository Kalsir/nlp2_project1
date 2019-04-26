#!/bin/bash -x

iterations=15
models="ibm1 ibm2 jump"

for model in $models; do
    python main.py --iterations=$iterations --model=$model --sampling_method=uniform
done

for model in $models; do
    python main.py --iterations=$iterations --model=$model --sampling_method=uniform --lower
done

for seed in "1 2 3"; do
    for model in "ibm2 jump"; do
        python main.py --iterations=$iterations --model=$model --sampling_method=random --seed=$seed
    done
done

# # hard to automate selection of 'best' ibm1 model to start from, so doing this manually instead...
# for model in $models; do
#     python main.py --iterations=$iterations --model=$model --sampling_method=uniform --probabilities=ibm1-uniform.pkl
# done
