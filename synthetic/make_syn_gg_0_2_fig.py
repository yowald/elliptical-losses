import numpy as np
import matplotlib as mpl
mpl.use('pgf')
import pickle
import os

def make_results_table():
  base_dir = './elliptic_losses/synthetic/results/'
  m_trains = [30, 40, 50, 60, 70, 80, 100, 150, 250, 500, 850]

  seeds = np.arange(2101, 2121)
  num_features = 10
  sampler_type = 'mggd_beta_0.20'
  sparsity_alpha = 0.85
  algs = ['gmrf', 'tyler']

  results_table = np.zeros((len(algs), len(m_trains), len(seeds)))

  for (j, m) in enumerate(m_trains):
    dir_depth1 = os.path.join(base_dir, '%d_%d'%(num_features, m))
    for (k, s) in enumerate(seeds):
      dir_depth2 = os.path.join(dir_depth1, '%d'%(s))
      dir_depth3 = os.path.join(dir_depth2, '%s'%(sampler_type))
      dir_depth4 = os.path.join(dir_depth3, '%s'%(sparsity_alpha))
      for (i, alg) in enumerate(algs):
        fname = os.path.join(dir_depth4, '%s_err.npy'%(alg))
        if os.path.isfile(fname):
          with open(fname, 'rb') as fp:
            cur_err = np.load(fp, encoding='latin1')
            results_table[i, j, k] = cur_err
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
fmt_tyler = 'o-'
# fmt_multit = 's-'
# fmt_gen_gauss_0_5 = 'p-'

mec = 'black'
mew = 2.
ms = 10.

plt.rc('text', usetex=True)

color_gmrf = '#FDAE84'
color_tyler = '#8BCBC8'
# color_gen_gauss_0_5 = '#3C2E3D'
# color_multit = '#B37C57'

results_table = make_results_table()

Ms = [30, 40, 50, 60, 70, 80, 100, 150, 250, 500, 850]
ms_start_ind = 3

alg_indices = {'gmrf': 0, 'tyler': 1}
num_samples = results_table.shape[2]


mse_tyler = np.mean(results_table[alg_indices['tyler'], :, :], axis=1)
std_tyler = np.std(results_table[alg_indices['tyler'], :, :], axis=1)

mse_gmrf = np.mean(results_table[alg_indices['gmrf'], :, :], axis=1)
std_gmrf = np.std(results_table[alg_indices['gmrf'], :, :], axis=1)

fig, ax  = newfig(2.3)

line_tyler = ax.errorbar(Ms[ms_start_ind:], mse_tyler[ms_start_ind:], yerr=std_tyler[ms_start_ind:]/(0. + np.sqrt(num_samples)), fmt=fmt_tyler, color=color_tyler, label='Tyler', linewidth=line_width, mec=mec, mew=mew, ms=ms)
line_gmrf = ax.errorbar(Ms[ms_start_ind:], mse_gmrf[ms_start_ind:], yerr=std_gmrf[ms_start_ind:]/(0. + np.sqrt(num_samples)), fmt=fmt_gmrf, color=color_gmrf, label='GMRF', linewidth=line_width, mec=mec, mew=mew, ms=ms)

ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_frame_on(True)
ax.set_xlabel('Training samples',fontsize=font_size)
ax.set_ylabel('$d(\\mathbf{w} - \\mathbf{w}^*)$',fontsize=font_size)
ax.legend(fontsize=font_size-2, loc='best')
plt.ylim((0.05, 0.25))
plt.xlim((60,850))
savefig('./elliptic_losses/synthetic/results/synthetic_mggd_0_2_data')
