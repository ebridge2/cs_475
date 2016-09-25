#!/bin/bash

algorithm=(perceptron averaged_perceptron)
options=(nlp easy hard bio speech finance vision)

for opt in "${options[@]}"; do
    for algo in "${algorithm[@]}"; do
        python code/classify.py --mode train --algorithm $algo --model-file ${opt}.perceptron.model --data ${opt}.train

        python code/classify.py --mode test --model-file ${opt}.perceptron.model --data ${opt}.dev --predictions-file ${opt}.dev.predictions

        acc="$(python compute_accuracy.py ${opt}.dev ${opt}.dev.predictions)"

        echo "${opt} | $algo | $acc"

    done
done
