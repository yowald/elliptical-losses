import numpy as np
import matplotlib as mpl
mpl.use('pgf')
import pickle
import os

def make_results_table():
    base_dir = './elliptical-losses/floods/results/'
    structures = ['full', 'time', 'time-space', 'time-space-3nn']

    m_trains = np.arange(200,410,30)
    seeds = np.arange(0,100)

    algs = ['gmrf', 'tyler', 'gen_gauss_0_5']
    results_table = np.zeros((len(algs), len(structures), len(m_trains), len(seeds)))

    for (j, s) in enumerate(structures):
      for (k, m) in enumerate(m_trains):
        dir_depth1 = base_dir + '%s_%d'%(s, m) + '/'
        for (l,seed) in enumerate(seeds):
          dir_depth2 = os.path.join(dir_depth1, '%d'%(seed))
          for (i, alg) in enumerate(algs):
            fname = os.path.join(dir_depth2, '%s_mse.npy'%(alg))
            if os.path.isfile(fname):
              with open(fname, 'rb') as fp:
                cur_err = np.load(fp, encoding='latin1')
                # cur_err = np.load(fp, )
                results_table[i, j, k, l] = cur_err
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
    "font.weight": 'bold',
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

# make our own newfig and savefig functions
def newfig(width):
    plt.clf()
    fig = plt.figure(figsize=figsize(width))
    ax = fig.add_subplot(111)
    return fig, ax

def savefig(filename):
  plt.savefig('{}.pgf'.format(filename))
  plt.savefig('{}.pdf'.format(filename))

font_size = 28
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
results_table[np.isnan(results_table)] = 0
Ms = np.arange(40,401,10)
structure_indices = {'full': 0, 'time': 1, 'time-space': 2, 'time-space-3nn': 3}

alg_indices = {'gmrf': 0, 'tyler': 1, 'laplace': 2}
num_samples = results_table.shape[-1]
y_upper_bound_val = 0.22

cur_results = results_table[alg_indices['gmrf'], structure_indices['full'],
                            :, :]
for i in range(cur_results.shape[0]):
    if np.all(cur_results[i,:] > 0) and np.mean(cur_results[i,:])<=y_upper_bound_val:
        start_ind_gmrf_full_test = i
        break
mse_gmrf_full_test = np.mean(cur_results, axis=1)
std_gmrf_full_test = np.std(cur_results, axis=1)

cur_results = results_table[alg_indices['laplace'], structure_indices['full'],
                            :, :]
for i in range(cur_results.shape[0]):
    if np.all(cur_results[i,:] > 0) and np.mean(cur_results[i,:])<=y_upper_bound_val:
        start_ind_laplace_full_test = i
        break
mse_laplace_full_test = np.mean(cur_results, axis=1)
std_laplace_full_test = np.std(cur_results, axis=1)

cur_results = results_table[alg_indices['tyler'], structure_indices['full'],
                            :, :]
for i in range(cur_results.shape[0]):
    if np.all(cur_results[i,:] > 0) and np.mean(cur_results[i,:])<=y_upper_bound_val:
        start_ind_tyler_full_test = i
        break
mse_tyler_full_test = np.mean(cur_results, axis=1)
std_tyler_full_test = np.std(cur_results, axis=1)

cur_results = results_table[alg_indices['gmrf'],
                            structure_indices['time-space-3nn'], :, :]
for i in range(cur_results.shape[0]):
    if np.all(cur_results[i,:] > 0):
        start_ind_gmrf_structured_test = i
        break
mse_gmrf_structured_test = np.mean(cur_results, axis=1)
std_gmrf_structured_test = np.std(cur_results, axis=1)

cur_results = results_table[alg_indices['laplace'],
                            structure_indices['time-space-3nn'], :, :]
for i in range(cur_results.shape[0]):
    if np.all(cur_results[i,:] > 0):
        start_ind_laplace_structured_test = i
        break
mse_laplace_structured_test = np.mean(cur_results, axis=1)
std_laplace_structured_test = np.std(cur_results, axis=1)

cur_results = results_table[alg_indices['tyler'],
                            structure_indices['time-space-3nn'], :, :]
for i in range(cur_results.shape[0]):
    if np.all(cur_results[i,:] > 0):
        start_ind_tyler_structured_test = i
        break
mse_tyler_structured_test = np.mean(cur_results, axis=1)
std_tyler_structured_test = np.std(cur_results, axis=1)

fig, ax  = newfig(2.3)

line_gmrf_full = ax.errorbar(Ms[start_ind_gmrf_full_test:],
                             mse_gmrf_full_test[start_ind_gmrf_full_test:],
                             yerr=std_gmrf_full_test[start_ind_gmrf_full_test:]/(0. + np.sqrt(num_samples)),
                             fmt=fmt_gmrf, color=color_gmrf, label='GMRF full',
                             linewidth=line_width, mec=mec, mew=mew, ms=ms)
line_laplace_full = ax.errorbar(Ms[start_ind_laplace_full_test:],
                                mse_laplace_full_test[start_ind_laplace_full_test:],
                                yerr=std_laplace_full_test[start_ind_laplace_full_test:]/(0. + np.sqrt(num_samples)),
                                fmt=fmt_laplace, color=color_laplace,
                                label='Laplace full', linewidth=line_width,
                                mec=mec, mew=mew, ms=ms)
line_tyler_full = ax.errorbar(Ms[start_ind_tyler_full_test:],
                              mse_tyler_full_test[start_ind_tyler_full_test:],
                              yerr=std_tyler_full_test[start_ind_tyler_full_test:]/(0. + np.sqrt(num_samples)),
                              fmt=fmt_tyler, color=color_tyler,
                              label='Tyler full', linewidth=line_width,
                              mec=mec, mew=mew, ms=ms)

line_gmrf_structured = ax.errorbar(Ms[start_ind_gmrf_structured_test:],
                                   mse_gmrf_structured_test[start_ind_gmrf_structured_test:],
                                   yerr=std_gmrf_structured_test[start_ind_gmrf_structured_test:]/(0. + np.sqrt(num_samples)),
                                   fmt=fmt_gmrf2, color=color_gmrf2,
                                   label='GMRF structured',
                                   linewidth=line_width, mec=mec, mew=mew, ms=ms)
line_laplace_structured = ax.errorbar(Ms[start_ind_laplace_structured_test:],
                                      mse_laplace_structured_test[start_ind_laplace_structured_test:],
                                      yerr=std_laplace_structured_test[start_ind_laplace_structured_test:]/(0. + np.sqrt(num_samples)),
                                      fmt=fmt_laplace2, color=color_laplace2,
                                      label='Laplace structured',
                                      linewidth=line_width, mec=mec, mew=mew, ms=ms)
line_tyler_structured = ax.errorbar(Ms[start_ind_tyler_structured_test:],
                                    mse_tyler_structured_test[start_ind_tyler_structured_test:],
                                    yerr=std_tyler_structured_test[start_ind_tyler_structured_test:]/(0. + np.sqrt(num_samples)),
                                    fmt=fmt_tyler2, color=color_tyler2,
                                    label='Tyler structured',
                                    linewidth=line_width, mec=mec, mew=mew, ms=ms)

ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_frame_on(True)
ax.set_xlabel('Training samples',fontsize=font_size)
ax.set_ylabel('NMSE',fontsize=font_size)
ax.legend(fontsize=font_size-2, loc='best')
# plt.ylim((0.05, 0.22))
plt.xlim((30,400))
savefig('./elliptical-losses/floods/results/floods_figure')
