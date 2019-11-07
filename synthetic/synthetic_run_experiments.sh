#!/bin/bash
CELL=jn

N=10
N_STEPS_NEWT=5000
N_STEPS_MM_NEWT=1000
N_STEPS_MM=5
SPARSITY_ALPHA=85e-2

for ((seed=2101; seed<=2120; seed+=1))
do
 for distribution_beta in 0.05 0.2 0.5 1.0
 do
    VARS='num_features='${N}',seed='${seed}',num_steps_newton='${N_STEPS_NEWT}$',num_steps_mm_newton='${N_STEPS_MM_NEWT}',num_steps_mm='${N_STEPS_MM}',sparsity_alpha='${SPARSITY_ALPHA}',beta='${distribution_beta}''
    echo "$VARS"
    command="python -m elliptical-losses.synthetic.synthetic_cov --num_features='${N}' --seed='${seed}' --vars='${VARS}' --num_steps_newton='${N_STEPS_NEWT}' --num_steps_mm='${N_STEPS_MM}' --sparsity_alpha='${SPARSITY_ALPHA}' --beta='${distribution_beta}'"
    eval "$command"
  done
done
