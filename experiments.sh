#!/bin/bash -x

iterations=15
models="jump"

for model in $models; do
    python main.py --iterations=$iterations --model=$model --sampling_method=uniform
done

for seed in "1 2 3"; do
    for model in "jump"; do
        python main.py --iterations=$iterations --model=$model --sampling_method=random --seed=$seed
    done
done

