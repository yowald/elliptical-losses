import numpy as np
import matplotlib as mpl
mpl.use('pgf')
import pickle
import os

def make_results_table():
  base_dir = './elliptical-losses/hugestock/results/'
  m_trains = [15, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600,
              650, 700, 750, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500,
              1600, 1700, 1800]

  seeds1 = np.arange(270, 330)
  seeds2 = np.arange(200, 230)

  structures = ['full', 'glasso']
  algs = ['gmrf', 'tyler', 'gen_gauss_0_5']
  results_table = np.zeros((len(algs), len(structures), len(seeds1),
                            len(seeds2), len(m_trains)))

  num_observed = 105
  for (k, m) in enumerate(m_trains):
    dir_depth1 = os.path.join(base_dir, '%d_%d'%(num_observed, m))
    for (l, seed1) in enumerate(seeds1):
      for (h, seed2) in enumerate(seeds2):
        dir_depth2 = os.path.join(dir_depth1, '%d_%d'%(seed1, seed2))
        for (j, s) in enumerate(structures):
          dir_depth3 = os.path.join(dir_depth2, '%s'%(s))
          for (i, alg) in enumerate(algs):
            fname = os.path.join(dir_depth3, '%s_mse.npy'%(alg))
            if os.path.isfile(fname):
              with open(fname, 'rb') as fp:
                cur_err = np.load(fp, encoding='latin1')
                results_table[i, j, l, h, k] = cur_err
  return results_table

def figsize(scale):
    fig_width_pt = 397.48499                        # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "times",
    "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 28,               # LaTeX default is 10pt font.
    "font.size": 28,
    "legend.framealpha": 0.2,# Make the legend/label fonts a little smaller
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "mathtext.fontset": 'custom',
    "mathtext.sf": 'sans',
    "figure.figsize": figsize(3),     # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
        ]
    }
mpl.rcParams.update(pgf_with_latex)

import matplotlib.pyplot as plt

# I make my own newfig and savefig functions
def newfig(width):
    plt.clf()
    fig = plt.figure(figsize=figsize(width))
    ax = fig.add_subplot(111)
    return fig, ax

def savefig(filename):
  plt.savefig('{}.pgf'.format(filename))
  plt.savefig('{}.pdf'.format(filename))


font_size = 24
line_width = 5

fmt_gmrf = 'v-'
fmt_tyler = '>-'
fmt_laplace = '^-'
fmt_gmrf2 = 's-'
fmt_tyler2 = 'o-'
fmt_laplace2 = 'p-'

mec = 'black'
mew = 2.
ms = 10.

plt.rc('text', usetex=True)


color_gmrf2 = '#FDAE84'
color_tyler2 = '#8BCBC8'
color_laplace2 = '#FF6347'
color_gmrf = '#A26F54'
color_tyler = '#3C2E3D'
color_laplace = '#742D21'

results_table = make_results_table()
alg_inds = {'gmrf': 0, 'tyler': 1, 'gen_gauss_0_5': 2}
structure_inds = {'full': 0, 'glasso': 1}
Ms = [15] + [v for v in np.arange(50, 801, 50)] + [v for v in np.arange(900, 1801, 100)]

num_samples = results_table.shape[0]
num_permutation_seeds = results_table.shape[1]
num_shuffling_seeds = results_table.shape[2]

# get minmum mse for each shuffling of the stocks, we will divide other
# mses by the minimum to get a measure of error that acts similarly for
# different shufflings of the stocks.
# for this purpose, we will exclude results with training sets smaller than 150
# since they contain nans.
results_for_min_error = results_table[:,:,:,:,3:]
results_averaged = np.mean(results_for_min_error, axis=3, keepdims=True)

min_mses_per_seed = np.min(results_averaged, axis=0, keepdims=True)
min_mses_per_seed = np.min(results_averaged, axis=1, keepdims=True)
min_mses_per_seed = np.min(min_mses_per_seed, axis=4, keepdims=True)

results_averaged /= min_mses_per_seed
results_table /= min_mses_per_seed

num_samples = num_permutation_seeds*num_shuffling_seeds

mse_gmrf_full = np.mean(results_table[alg_inds['gmrf'],
                                      structure_inds['full'],
                                      :, :, :],
                        axis=(0,1))
std_gmrf_full = np.std(results_table[alg_inds['gmrf'],
                                     structure_inds['full'],
                                     :, :, :],
                       axis=(0,1))

mse_tyler_full = np.mean(results_table[alg_inds['tyler'],
                                       structure_inds['full'],
                                       :, :, :],
                         axis=(0,1))
std_tyler_full = np.std(results_table[alg_inds['tyler'],
                                      structure_inds['full'],
                                      :, :, :],
                        axis=(0,1))

mse_laplace_full = np.mean(results_table[alg_inds['gen_gauss_0_5'],
                                         structure_inds['full'],
                                         :, :, :],
                           axis=(0,1))
std_laplace_full = np.std(results_table[alg_inds['gen_gauss_0_5'],
                                        structure_inds['full'],
                                        :, :, :],
                          axis=(0,1))

mse_gmrf_structured = np.mean(results_table[alg_inds['gmrf'],
                                            structure_inds['glasso'],
                                            :, :, :],
                              axis=(0,1))
std_gmrf_structured = np.std(results_table[alg_inds['gmrf'],
                                           structure_inds['glasso'],
                                           :, :, :],
                             axis=(0,1))

mse_tyler_structured = np.mean(results_table[alg_inds['tyler'],
                                             structure_inds['glasso'],
                                             :, :, :],
                               axis=(0,1))
std_tyler_structured = np.std(results_table[alg_inds['tyler'],
                                            structure_inds['glasso'],
                                            :, :, :],
                              axis=(0,1))

mse_laplace_structured = np.mean(results_table[alg_inds['gen_gauss_0_5'],
                                               structure_inds['glasso'],
                                               :, :, :],
                                 axis=(0,1))
std_laplace_structured = np.std(results_table[alg_inds['gen_gauss_0_5'],
                                              structure_inds['glasso'],
                                              :, :, :],
                                axis=(0,1))

fig, ax  = newfig(2.3)

line_gmrf_full = ax.errorbar(Ms[15:], mse_gmrf_full[15:],
                             yerr=std_gmrf_full[15:]/(0. + np.sqrt(num_samples)),
                             fmt=fmt_gmrf, color=color_gmrf, label='GMRF full',
                             linewidth=line_width, mec=mec, mew=mew, ms=ms)
line_laplace_full = ax.errorbar(Ms[10:], mse_laplace_full[10:],
                                yerr=std_laplace_full[10:]/(0. + np.sqrt(num_samples)),
                                fmt=fmt_laplace, color=color_laplace,
                                label='Laplace full', linewidth=line_width,
                                mec=mec, mew=mew, ms=ms)
line_tyler_full = ax.errorbar(Ms[10:], mse_tyler_full[10:],
                              yerr=std_tyler_full[10:]/(0. + np.sqrt(num_samples)),
                              fmt=fmt_tyler, color=color_tyler,
                              label='Tyler full', linewidth=line_width,
                              mec=mec, mew=mew, ms=ms)
line_gmrf = ax.errorbar(Ms[:], mse_gmrf_structured,
                        yerr=std_gmrf_structured/(0. + np.sqrt(num_samples)),
                        fmt=fmt_gmrf2, color=color_gmrf2,
                        label='GMRF structured', linewidth=line_width,
                        mec=mec, mew=mew, ms=ms)
line_laplace = ax.errorbar(Ms[:], mse_laplace_structured,
                           yerr=std_laplace_structured/(0. + np.sqrt(num_samples)),
                           fmt=fmt_laplace2, color=color_laplace2,
                           label='Laplace structured', linewidth=line_width,
                           mec=mec, mew=mew, ms=ms)
line_tyler = ax.errorbar(Ms[3:], mse_tyler_structured[3:],
                         yerr=std_tyler_structured[3:]/(0. + np.sqrt(num_samples)),
                         fmt=fmt_tyler2, color=color_tyler2,
                         label='Tyler structured', linewidth=line_width,
                         mec=mec, mew=mew, ms=ms)

ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_frame_on(True)
ax.set_xlabel('Training samples',fontsize=font_size)
ax.set_ylabel('MSE/Best MSE',fontsize=font_size)
ax.legend(fontsize=font_size-2, loc='best')
plt.ylim((1.1, 1.8))
plt.xlim((Ms[2]-10,Ms[-1]))
savefig('./elliptical-losses/hugestock/results/hugestock_mse_from_best')
