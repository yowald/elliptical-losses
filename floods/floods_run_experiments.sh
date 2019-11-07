#!/bin/bash

for ((seed=0; seed<=100; seed+=1))
do
  for structure_type in 'full' 'time' 'time-space' 'time-space-3nn'
  do
    command="python -m elliptical-losses.floods.floods --seed='${seed}' --structure_type='${structure_type}'"
    eval "$command"
  done
done
