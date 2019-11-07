#!/bin/bash

NOBSERVED=105
NSTOCKS=120

for ((seed1=270; seed1<=330; seed1+=1))
do
  for ((seed2=200; seed2<=230; seed2+=1))
  do
    command="python -m elliptical-losses.hugestock.hugestock --seed1='${seed1}' --seed2='${seed2}' --num_observed='${NOBSERVED}' --num_stocks='${NSTOCKS}'"
    eval "$command"
  done
done
