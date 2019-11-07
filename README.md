# elliptical-losses
Code accompanying the paper "Globally Optimal Learning for Structured Elliptical Losses", published at NeurIPS 2019

## How to use the code
To run the synthetic experiments, please download or clone the repository and run: </br>
*./elliptical-losses/synthetic/synthetic_run_experiments.sh* </br>
Use this from the folder where the 'elliptical-losses' package is at.

To run experiments on river discharge and stocks data, replace 'synthetic' with 'floods' or 'hugestock' in the above command.

Once these scripts are run, figures can be reproduced using:</br>
*python ./elliptic_losses/synthetic/make_syn_gauss_fig.py*</br>
replace 'gauss' with 'gg_0_5' and 'gg_0_2' for the other plots of results on synthetic data.</br>
A .pdf file with the figure should then be found at:</br>
*./elliptic_losses/synthetic/results/synthetic_gaussian_data.pdf*

To create figures for river discharge and stocks data, use the 'make_..._fig.py' scripts in the appropriate directories (similarly to the above description).

### Important notes
The river discharge and stocks experiments can be quite heavy on compute resources, but lighter versions can be run muich more easily.</br>
Resulting curves will be less smooth, but the general behavior of the alogrithms should be maintained.

To reduce the runtime of the experiments you may try:
* Reduce the number of seeds in the different run_experiments.sh scripts.</br>
In stocks experiments, the outer loop of the script repeats the experiment for different subsets of stocks (chooses randomly 105 stocks as observed and 15 as targets), while the inner loop shuffles the training data and repeats the experiment for the current subset of stocks. </br>
Hence reducing seeds in the inner loop too much may result in very jumpy curves, while reducing seeds in the outer loop too much can affect the relative positions of curves with respect to each other. Fixing the seed in the outer loop, and setting the repetitions in the inner loop to ~10, should give a good approximation for the behavior on the fixed subset of stocks.
* Reduce the value of flags *num_steps_newton*, *num_steps_mm_newton*, *num_steps_mm*. The first two flags can be cut down by a factor of ~5, and still maintain reasonable results. Running times will be be much lower for the structured methods.

### Requirements
This code should be run with python 3 (tested with version 3.6.2, but should work with others)</br>
The following packages are used in the code: numpy, scipy, tensorflow, pickle, absl

# Acknowledgements
We thank Guy Shalev for preparing the river discahrge dataset. Discharge levels were downloaded from the website of the United States Geological Survey (USGS), while rainfall measurements are available from the Global Satellite Mapping of Precipitation (GSMaP) product [1]. </br>
We also thank Elad Mezuman and Amir Globerson for helping with early iterations of the paper and code.

[1] T.  Ushio,  K.  Okamoto,  T.  Iguchi,  N.  Takahashi,  K.  Iwanami,  K.  Aonashi,  S.  Shige,H. Hashizume, T. Kubota, and T. Inoue. The global satellite mapping of precipitation (GSMaP)project.Aqua (AMSR-E), 2004
