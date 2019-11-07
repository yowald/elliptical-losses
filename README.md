# elliptical-losses
Code accompanying the paper "Globally Optimal Learning for Structured Elliptical Losses", published at NeurIPS 2019

## How to use the code
To run the synthetic experiments, please download or clone the repository and run: </br>
./elliptical-losses/synthetic/synthetic_run_experiments.sh </br>
Use this from the folder where the 'elliptical-losses' package is at.

To run experiments on river discharge and stocks data, replace 'synthetic' with 'floods' or 'hugestock' in the above command.

Once these scripts are run, figures can be reproduced using:</br>
python ./elliptic_losses/synthetic/make_syn_gauss_fig.py</br>
replace 'gauss' with 'gg_0_5' and 'gg_0_2' for the other plots of results on synthetic data.</br>
A .pdf file with the figure should then be found at:</br>
./elliptic_losses/synthetic/results/synthetic_gaussian_data.pdf

To create figures for river discharge and stocks data, use the 'make_..._fig.py' scripts in the appropriate directories (similarly to the above description).

### Important notes
The river discharge and stocks experiments can be quite heavy on compute resources, but lighter versions can be run muich more easily.</br>
Resulting curves will be less smooth, but the general behavior of the alogrithms should be maintained.

To reduce the runtime of the experiments you may try:

### Requirements
This code should be run with python 3 (tested with version 3.6.2, but should work with others)</br>
The following packages are used in the code: numpy, scipy, tensorflow, pickle, absl
