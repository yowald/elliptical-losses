"""Functions to generate synthetic data and run experiment.

flags control number of variables, sparsity parameter, seed etc.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from absl import app
# from absl import flags

import os
import sys
import numpy as np
import scipy as sp
from scipy.linalg import cho_factor
from scipy.linalg import LinAlgError
from sklearn.datasets import make_sparse_spd_matrix
import tensorflow as tf
from ..PositiveScalarSamplerFactory import PositiveScalarSamplerFactory
from ..structured_optimizers import GMRFOptimizer
from ..structured_optimizers import LossFunctionFactory
from ..structured_optimizers import structured_elliptical_maximum_likelihood

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('num_features', 10, '')
tf.app.flags.DEFINE_integer('seed', 1, '')
tf.app.flags.DEFINE_integer('num_steps_newton', 75000,
                            """Number of steps for newton optimizer.""")
tf.app.flags.DEFINE_integer('num_steps_mm_newton', 1000,
                            """Number of steps or newton in MM algorithm.""")
tf.app.flags.DEFINE_integer('num_steps_mm', 100,
                            """Number of steps for MM algorithm.""")
tf.app.flags.DEFINE_boolean('delete_checkpoint', False,
                            """Delete existing checkpoint and start fresh.""")
tf.app.flags.DEFINE_boolean('delete_existing', False,
                            """Delete existing checkpoint and start fresh.""")
tf.app.flags.DEFINE_float('beta', 0.5,
                          """shape for generalized gaussian data creation.""")
tf.app.flags.DEFINE_float('nu', 3.,
                          'degrees of freedom for multivariate-t'
                          'data creation.')
tf.app.flags.DEFINE_float('learning_rate', 0.05,
                          """Train Validation fraction.""")
tf.app.flags.DEFINE_boolean('standardize_data', True,
                            """If True, divides data by standard deviation.""")
tf.app.flags.DEFINE_float('outliers_std', 10., '')
tf.app.flags.DEFINE_float('outliers_samples_prob', 0.05, '')
tf.app.flags.DEFINE_float('sparsity_alpha', 0.85, '')
tf.app.flags.DEFINE_string('sampler_type', 'mggd',
                           """scalar sampler type to use for data generation""")
tf.app.flags.DEFINE_string('save_dir',
                           './elliptical-losses/synthetic/results/',
                           'Directory where to write event logs '
                           'and checkpoint.')


def is_pos_def(matrix):
  return np.all(np.linalg.eigvals(matrix) > 0)


def get_sparse_high_correlations(dim=25, seed=1, rep_num=1000,
                                 sparsity_alpha=0.9):
  """Gets sparse inverse covariance matrix.

  The method draw a few matrices and returns te one where the average
  correlation between variables is the highest.

  Args:
    dim: the dimension of the matrix to be returned.
    seed: seed for reproducibility.
    rep_num: number of matrices to draw and choose from.
    sparsity_alpha: sparsity parameter. see details of make_sparse_spd_matrix.

  Returns:
    A sparse inverse covariance matrix.
  """
  np.random.seed(seed)
  max_mean = 0
  for _ in range(rep_num):
    candidate_matrix = make_sparse_spd_matrix(dim, alpha=sparsity_alpha,
                                              smallest_coef=.4, largest_coef=.7)
    candidate_correlations = np.linalg.inv(candidate_matrix)
    diag_part = np.sqrt(np.expand_dims(np.diag(candidate_correlations), axis=0))
    candidate_correlations /= diag_part
    candidate_correlations /= diag_part.transpose()
    cur_mean = np.tril(np.abs(candidate_correlations)).mean()
    if max_mean < cur_mean:
      best_candidate = candidate_matrix
      max_mean = cur_mean
  return best_candidate


def get_edge_indices_from_matrix(matrix, miss_probability=0.0):
  """Gets a list of indices where the entries in the given matrix are non-zero.

  Each index is a list of two integers [i,j] such that matrix[i,j]!=0.

  Args:
    matrix: the matrix to get the edges of.
    miss_probability: float in the range [0., 1.], edges will be omitted from
      the least with this probability.

  Returns:
    A list of indices (or edges so to speak).
  """
  [n, _] = matrix.shape
  edge_indices_triu = []
  edge_indices_tril = []
  for i in range(n-1):
    for j in range(i+1, n):
      if (np.abs(matrix[i, j]) > 0 and np.random.rand() > miss_probability):
        edge_indices_triu.append([i, j])
        edge_indices_tril.append([j, i])
  edge_indices = np.array(edge_indices_triu + edge_indices_tril)
  return edge_indices


def check_pd(matrix, lower=True):
  """Checks if matrix is positive definite.

  Args:
    matrix: input to check positive definiteness of.
    lower: If True gets the lower triangular part of the Cholesky decomposition.

  Returns:
    If matrix is positive definite returns True and its Cholesky decomposition,
    otherwise returns False and None.
  """
  try:
    return True, np.tril(cho_factor(matrix, lower=lower)[0])
  except LinAlgError as err:
    if 'not positive definite' in str(err):
      return False, None


def get_elliptic_data(scalar_sampler, n, m_train, seed=1, sparsity_alpha=0.9):
  """Generates data from an elliptic distribution.

  Args:
    scalar_sampler: a function that receives an integer m, and draws m positive
      scalars from some distribution. the distribution defines the type of
      elliptic distribution we are using.
      See Frahm 04. https://kups.ub.uni-koeln.de/1319/
    n: number of variables in the elliptic distribution.
    m_train: number of training examples to draw from distribution.
    seed: seed for the random number generator, for reproducibility purposes.
    sparsity_alpha: sparsity parameter. see details of make_sparse_spd_matrix.

  Returns:
    Training data, and the inverse covariance matrix it was generates with.

  Raises:
    Exception: if there was a problem with generating a covariance matrix, such
    that the resulting matrix was not positive definite.
  """
  np.random.seed(seed)
  num_samples = m_train
  inverse_cov = get_sparse_high_correlations(n, seed,
                                             sparsity_alpha=sparsity_alpha)
  inverse_cov = np.float32(inverse_cov)
  covariance = np.linalg.inv(inverse_cov)
  if not check_pd(covariance):
    raise Exception('covariance matrix is not Positive Definite')

  spherical_uniform = np.random.randn(n, num_samples)
  spherical_uniform /= np.linalg.norm(spherical_uniform, axis=0)

  scaling_params = scalar_sampler(num_samples)
  train_data = np.multiply(scaling_params.T,
                           sp.linalg.sqrtm(covariance).dot(spherical_uniform))

  return train_data, inverse_cov


def get_losses_dictionary(features_dimension):
  """Creates a dictionary with all the losses to test, and their gradients.

  Args:
    features_dimension: the dimension of the inverse covariance matrix we are
      estimating.

  Returns:
    A dictionary where the keys are the names of the losses to estimate and the
    values are tuples of (loss, grad) where loss is the loss function and grad
    is its gradient.
  """
  loss_factory = LossFunctionFactory()
  loss_dict = {
      'tyler': loss_factory.tylers_estimator({'d': features_dimension}),
      'gen_gauss_0_2': loss_factory.generalized_gaussian({
          'm': (features_dimension)**((0.2-1)/0.2),
          'beta': 0.2
      }),
      'gen_gauss_0_5': loss_factory.generalized_gaussian({
          'm': (features_dimension)**((0.5-1)/0.5),
          'beta': 0.5
      }),
      'multivariate_t': loss_factory.multivariate_t({
          'nu': 3.,
          'd': features_dimension
      })
  }
  return loss_dict


def get_distance_from_ground_truth(ground_truth_matrix, estimation, std=None):
  """Calculates an normalized distance of estimation and ground truth matrix.

  Args:
    ground_truth_matrix: the true inverse covariance matrix we are estimating.
    estimation: the estimation of the matrix.
    std: if not None, it is the standard deviation of each feature in the
      training data. This is used to restore the original sclaes of the features
      before measuring the distance between matrices.
  Returns:
    the normalized frobenius distance (i.e. froebnius distance divided by
    frobenius norm of ground_truth_matrix) between normalized versions of
    estimation and ground_truth_matrix. normaliztion is done by dividing
    estimation by its trace and multiplying by that of ground_truth_matrix.
  """
  if std is not None:
    diag_of_stds = np.linalg.inv(np.diag(std))
    estimation = diag_of_stds.dot(estimation).dot(diag_of_stds)
  estimation *= (np.trace(ground_truth_matrix)/np.trace(estimation))
  distance_between_normalized = np.linalg.norm(estimation - ground_truth_matrix)
  return distance_between_normalized/np.linalg.norm(ground_truth_matrix)


def run_experiment(data_train, edge_indices_with_diag, inverse_covariance,
                   seed, sampler_type, sampler_param, sparsity_alpha,
                   num_steps_newton, num_steps_mm_newton, num_steps_mm,
                   standardize_data=True):
  """Runs a single experiment comparing all losses on generated data.

  Args:
    data_train: the generated data to run on.
    edge_indices_with_diag: list of edges to use for the graphical structure.
      An edge is itself a list of two integers in the range [0..num_features-1].
      Should include self edges (i.e. [i,i]) for digonal elements of the inverse
      covariance.
    inverse_covariance: the ground truth inverse covariance matrix used to
      generate the data.
    seed: the seed used in generation of the data, for logging purposes.
    sampler_type: the type of sampler used to generate the data (see
      PositiveScalarSamplerFactory)
    sampler_param: parameter for the scalar sampler (shape for mggd and degrees
      of freedom for t-distribution)
    sparsity_alpha: sparsity parameter. see details of make_sparse_spd_matrix.
    num_steps_newton: maximum number of steps for newton optimizer in structured
      gmrfs.
    num_steps_mm_newton: maximum number of steps for inner loop newton optimizer
      in minimization majorization of structured robust mrfs.
    num_steps_mm: maximum number of minimization majorization steps in robust
      mrfs.
    standardize_data: if True, divides training data by standard deviations
      before passing to structured optimizers.
  """
  [num_features, m_train] = data_train.shape
  tf.logging.info('==== seed={}, m_train={},'.format(seed, m_train))

  # Create directory to save results.
  full_dir = os.path.join(FLAGS.save_dir, '%d_%d' %
                          (num_features, m_train))
  full_dir = os.path.join(full_dir, '%d' % (seed))
  if sampler_type == 'mggd':
    full_dir = os.path.join(full_dir,
                            '%s_beta_%0.2f' % (sampler_type, sampler_param))
  elif sampler_type == 'multivariate_t':
    full_dir = os.path.join(full_dir,
                            '%s_nu_%0.2f' % (sampler_type, sampler_param))
  full_dir = os.path.join(full_dir, '%0.2f' % (sparsity_alpha))
  if tf.gfile.Exists(full_dir):
    if FLAGS.delete_existing:
      tf.gfile.DeleteRecursively(full_dir)
  tf.gfile.MakeDirs(full_dir)

  # Standardize data and keep stds
  std_val = None
  if standardize_data:
    std_val = np.std(data_train, axis=1)
    data_train_ = data_train/np.std(data_train, axis=1, keepdims=True)
  else:
    data_train_ = data_train

  # Sample Covariance
  sample_cov = data_train.dot(data_train.T)/m_train
  inverse_sample_cov = np.linalg.pinv(sample_cov)

  sample_cov_err = get_distance_from_ground_truth(inverse_covariance,
                                                  inverse_sample_cov,
                                                  std=None)
  # Save results for sample covariance estimator.
  fname = os.path.join(full_dir, '%s.npy' % 'sample_cov_err')
  print('fname', fname)
  with tf.gfile.Open(fname, 'w') as fp:
    print(sample_cov_err)
    np.save(fp, sample_cov_err)

  # Gaussian MRF
  gmrf_optimizer = GMRFOptimizer(num_features, edge_indices_with_diag)
  estimate_gmrf, _ = (
      gmrf_optimizer.alt_newton_coord_descent(data_train_,
                                              max_iter=num_steps_newton))
  gmrf_err = get_distance_from_ground_truth(inverse_covariance, estimate_gmrf,
                                            std=std_val)
  fname = os.path.join(full_dir, '%s.npy' % 'gmrf_err')
  print('fname', fname)
  with tf.gfile.Open(fname, 'w') as fp:
    print(gmrf_err)
    np.save(fp, gmrf_err)

  n_steps_newt = num_steps_mm_newton
  loss_dict = get_losses_dictionary(num_features)
  for estimator_name, (loss, loss_grad) in loss_dict.items():
    estimate_cur, _ = (
        structured_elliptical_maximum_likelihood(data_train_, loss, loss_grad,
                                                 edge_indices_with_diag,
                                                 initial_value=None,
                                                 max_iters=num_steps_mm,
                                                 newton_num_steps=n_steps_newt))
    cur_err = get_distance_from_ground_truth(inverse_covariance, estimate_cur,
                                             std=std_val)
    fname = os.path.join(full_dir, '%s.npy' % (estimator_name+'_err'))
    print('fname', fname)
    with tf.gfile.Open(fname, 'w') as fp:
      print(cur_err)
      np.save(fp, cur_err)


def main(argv):
  del argv  # Unused.
  tf.logging.set_verbosity(tf.logging.INFO)
  seed = FLAGS.seed
  num_features = FLAGS.num_features
  num_steps_newton = FLAGS.num_steps_newton
  num_steps_mm_newton = FLAGS.num_steps_mm_newton
  num_steps_mm = FLAGS.num_steps_mm
  sparsity_alpha = FLAGS.sparsity_alpha
  sampler_type = FLAGS.sampler_type
  standardize_data = FLAGS.standardize_data
  beta = FLAGS.beta
  nu = FLAGS.nu
  # Get the scalar sampler for generating elliptic data
  scalar_sampler_factory = PositiveScalarSamplerFactory()
  if sampler_type == 'mggd':
    assert(beta <= 1 and beta > 0)
    sampler_param = beta
    gen_gauss_sampler_params = {'shape': beta, 'dim': num_features}
    scalar_sampler = \
        scalar_sampler_factory.generalized_gaussian(gen_gauss_sampler_params)
  elif sampler_type == 'multivariate_t':
    assert nu > 2
    sampler_param = nu
    multi_t_sampler_params = {'nu': nu, 'dim': num_features}
    scalar_sampler = \
        scalar_sampler_factory.multivariate_t(multi_t_sampler_params)
  else:
    raise ValueError('Unrecognized sampler type')

  # Create training data and ground truth parameters.
  m_train_max = 1500
  np.random.seed(seed)
  data_train, inverse_cov = get_elliptic_data(scalar_sampler, num_features,
                                              m_train_max, seed=seed,
                                              sparsity_alpha=sparsity_alpha)
  edge_indices = get_edge_indices_from_matrix(inverse_cov)
  edge_indices = np.concatenate([edge_indices,
                                 [[i, i] for i in range(num_features)]])

  m_trains = [30, 40, 50, 60, 70, 80, 100, 150, 250, 500, 850]
  for m in m_trains:
    np.random.seed(seed)
    train_inds = np.random.permutation(m_train_max)[:m]
    data_train_cur = data_train[:, train_inds]
    print('==== n={}, seed={}, m_train={}, sparsity_alpha={}'
          ', distribution_beta={}'.format(num_features, seed, m, sparsity_alpha,
                                          beta))
    run_experiment(data_train_cur, edge_indices, inverse_cov, seed,
                   sampler_type, sampler_param, sparsity_alpha,
                   num_steps_newton, num_steps_mm_newton, num_steps_mm,
                   standardize_data=standardize_data)


if __name__ == '__main__':
  tf.app.run(main)
