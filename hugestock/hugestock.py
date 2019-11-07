"""Functions to load stocks data and run experiments
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import datetime
import os
from absl import app
from absl import flags
import numpy as np
import pickle
import tensorflow as tf

from ..structured_optimizers import GMRFOptimizer
from ..structured_optimizers import LossFunctionFactory
from ..structured_optimizers import non_structured_elliptical_maximum_likelihood
from ..structured_optimizers import structured_elliptical_maximum_likelihood
from ..util import fit_least_squares_scalar_multiple
from ..util import get_edge_indices_from_adjmat_symmetric
from ..util import get_regressor_from_inverse_covariance

FLAGS = flags.FLAGS
tf.app.flags.DEFINE_string('save_dir',
                           './elliptical-losses/hugestock/results/',
                           'Directory where to write event logs '
                           'and checkpoint.')
tf.app.flags.DEFINE_integer('m_train', 879, '')
tf.app.flags.DEFINE_integer('num_observed', 105, '')
tf.app.flags.DEFINE_integer('num_stocks', 120, '')
tf.app.flags.DEFINE_integer('seed1', 1, '')
tf.app.flags.DEFINE_integer('seed2', 2, '')
tf.app.flags.DEFINE_boolean('delete_existing', False, 'Delete existing  '
                            'directory if exits.')
tf.app.flags.DEFINE_integer('num_steps_newton', 75000,
                            """Number of steps for newton optimizer.""")
tf.app.flags.DEFINE_integer('num_steps_mm_newton', 7000,
                            """Number of steps or newton in MM algorithm.""")
tf.app.flags.DEFINE_integer('num_steps_mm', 10,
                            """Number of steps for MM algorithm.""")


def load_data_and_structure(seed1, num_stocks, num_observed):
  """Loads data and structure for stocks data.

  Args:
    seed1: seed to set the observed and hidden stocks. for reproducibility
      purposes
    num_stocks: total number of stocks to consider in our data.
    num_observed: how many of the 'num_stocks' stocks should be observed.
  Returns:
    observed_train: training data of observed stocks. array of shape
      [num_features, num_instances].
    hidden_train: training data of hidden stocks (i.e. the labels). this is
      an array of shape [num_labels, num_instances]
    observed_test: test data of observed stocks.
    hidden_test: test data of hidden stocks
    edge_indices: list of edges to use for the graphical structure.
      An edge is itself a list of two integers in the range [0..num_stocks-1].
      Includes self edges (i.e. [i,i]) for digonal elements of the inverse
      covariance.
    stds_train: standard deviations of observed and hidden stocks in the
      training data (stds of observed stocks are first, last ones are of the
      hidden stocks).
  """

  # loads the stock values
  data_path = './elliptical-losses/hugestock/data/stocks_04_15_data.npy'
  with open(data_path, 'rb') as fp:
    all_data = np.load(fp, encoding='latin1')
    fp.close()

  # loads the dates of corresponding records in the data table
  data_path = './elliptical-losses/hugestock/data/stocks_04_15_dates.npy'
  with open(data_path, 'rb') as fp:
    dates_list = np.load(fp, encoding='bytes')
    fp.close()


  # loads the result of running glasso on the training data.
  # this means we don't learn structures for specific divisions of stocks, but
  # we don't have to run glasso for each trail separately.
  data_path = './elliptical-losses/hugestock/data/glasso_precision.npy'
  with open(data_path, 'rb') as fp:
      glasso_structure = np.load(fp, encoding='latin1')
      fp.close()

  # permute stocks to decide on a random subset of stocks we will be running
  # on. just to keep running times a little bit lower...
  num_stocks_in_dataset = all_data.shape[1]
  np.random.seed(seed1)
  perm_stocks = np.random.permutation(np.arange(num_stocks_in_dataset))
  all_data_permuted = all_data[:, perm_stocks]
  all_data_permuted = all_data_permuted[:, :num_stocks]
  # load list of edges in structure
  glasso_structure = glasso_structure[[[i] for i in perm_stocks], perm_stocks]
  glasso_structure = glasso_structure[:num_stocks, :num_stocks]
  adjmat = (glasso_structure != 0)
  edge_indices = get_edge_indices_from_adjmat_symmetric(adjmat)
  edge_indices = np.vstack([edge_indices, [[i, i] for i in range(num_stocks)]])

  # divide to train and test data, exclude the approximate dates of the
  # financial crisis from the training data.
  crisis_start = datetime.datetime.strptime('2007-07-01', '%Y-%m-%d')
  crisis_end = datetime.datetime.strptime('2009-07-01', '%Y-%m-%d')
  test_start = datetime.datetime.strptime('2011-07-01', '%Y-%m-%d')

  train_inds = []
  test_inds = []
  for d in dates_list:
    # if (d < crisis_start) or (d > crisis_end and d < test_start):
    if d < test_start:
      train_inds += [True]
    else:
      train_inds += [False]
    if d >= test_start:
      test_inds += [True]
    else:
      test_inds += [False]
  data_train = all_data_permuted[train_inds, :]
  data_test = all_data_permuted[test_inds, :]

  # standardize data.
  means_train = np.mean(data_train, axis=0, keepdims=True)
  stds_train = np.std(data_train, axis=0, keepdims=True)

  data_train -= means_train
  data_train /= stds_train

  data_test -= means_train
  data_test /= stds_train
  # divide to observed stocks (features) and hidden ones (labels)
  observed_train = data_train[:, :num_observed]
  hidden_train = data_train[:, num_observed:]

  observed_test = data_test[:, :num_observed]
  hidden_test = data_test[:, num_observed:]

  return (observed_train, hidden_train, observed_test, hidden_test,
          edge_indices, stds_train)


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
      'gen_gauss_0_5': loss_factory.generalized_gaussian({
          'm': (features_dimension)**(-1),
          'beta': 0.5
      })
  }
  return loss_dict


def get_mse(observed_test, hidden_test, stds, regressor):
  """Returns the mse of a regressor on the test data.

  Args:
    observed_test: features of test data.
    hidden_test: labels of test data.
    stds: standard deviation of all stocks on the training data, first items
      are for observed stocks, last ones are for hidden ones.
    regressor: the linear regressor to get the error for.

  Returns: the mse of the regressor, multiuplied elementwise by the standrad
    deviation. this is done to measure the mse in scales similar to the
    original data, before standardization.
  """
  num_observed = observed_test.shape[1]
  std_hidden = stds[:, num_observed:]
  errs = (hidden_test.T-regressor.dot(observed_test.T))*std_hidden.T
  return np.mean(errs**2)


def gmrf(train_data, edge_indices, stds, observed_test, hidden_test,
         num_steps_newton):
  """Runs the structured square loss method.

  Args:
    train_data: array of size [num_examples, num_stocks].
    edge_indices: list of edges to use for the graphical structure.
      An edge is itself a list of two integers in the range [0..num_stocks-1].
      Includes self edges (i.e. [i,i]) for diagonal elements of the inverse
      covariance.
    stds: standard deviation of all stocks on the training data, first items
      are for observed stocks, last ones are for hidden ones.
    observed_test: array of size [num_test_examples, num_observed_stocks].
    hidden_test: array of size [num_test_examples, num_hidden_stocks].
    num_steps_newton: maximum number of coordinate descent iterations to perform
      in the newton optimizer.

  Returns:
    the mse of the learned regressor over the test set.
  """
  num_observed = observed_test.shape[1]
  num_hidden = hidden_test.shape[1]
  gmrf_optimizer = GMRFOptimizer(num_observed+num_hidden, edge_indices)
  # run the optimizer.
  inverse_covariance, _ = (
      gmrf_optimizer.alt_newton_coord_descent(train_data.T,
                                              max_iter=num_steps_newton))
  # construcut regressor from inverse covariance.
  regressor = get_regressor_from_inverse_covariance(inverse_covariance,
                                                    num_observed)
  # get predictions on training data.
  observed_train = train_data[:, :num_observed]
  hidden_train = train_data[:, num_observed:]
  # small touch - fit a multiplicative scalar to the regressor that minimizes
  # the squared error.
  least_squares_scaling = fit_least_squares_scalar_multiple(regressor,
                                                            observed_train,
                                                            hidden_train)
  return get_mse(observed_test, hidden_test, stds,
                 least_squares_scaling*regressor)


def sample_covariance(train_data, stds, observed_test, hidden_test):
  """Runs the unstructured square loss method.

  This is equivalent to using the sample covariance, hence the name of the
  method.

  Args:
    train_data: array of size [num_examples, num_stocks].
    stds: standard deviation of all stocks on the training data, first items
      are for observed stocks, last ones are for hidden ones.
    observed_test: array of size [num_test_examples, num_observed_stocks].
    hidden_test: array of size [num_test_examples, num_hidden_stocks].

  Returns:
    the mse of the learned regressor over the test set.
  """
  m = train_data.shape[0]
  num_observed = observed_test.shape[1]
  # calculate sample covariance matrix.
  covariance = train_data.T.dot(train_data)/m
  # construcut regressor from covariance matrix.
  sigma_yx = covariance[num_observed:, :num_observed]
  sigma_xx_inv = np.linalg.inv(covariance[:num_observed, :num_observed])
  regressor = sigma_yx.dot(sigma_xx_inv)
  # get predictions on training data.
  observed_train = train_data[:, :num_observed]
  hidden_train = train_data[:, num_observed:]
  # small touch - fit a multiplicative scalar to the regressor that minimizes
  # the squared error.
  least_squares_scaling = fit_least_squares_scalar_multiple(regressor,
                                                            observed_train,
                                                            hidden_train)
  return get_mse(observed_test, hidden_test, stds,
                 least_squares_scaling*regressor)


def robust_estimator(loss, loss_grad, train_data, edge_indices, stds,
                     observed_test, hidden_test, num_steps_mm_newton,
                     num_steps_mm):
  """Runs a structured method with a robust loss.

  Args:
    loss: the robust loss function to use.
    loss_grad: the gradient function of the robust loss.
    train_data: array of size [num_examples, num_stocks].
    edge_indices: list of edges to use for the graphical structure.
      An edge is itself a list of two integers in the range [0..num_stocks-1].
      Includes self edges (i.e. [i,i]) for digonal elements of the inverse
      covariance.
    stds: standard deviation of all stocks on the training data, first items
      are for observed stocks, last ones are for hidden ones.
    observed_test: array of size [num_test_examples, num_observed_stocks].
    hidden_test: array of size [num_test_examples, num_hidden_stocks].
    num_steps_mm_newton: maximum number of coordinate descent iterations to
      perform in the newton optimizer within each minimizaiton majorrization
      iteration.
    num_steps_mm: maximum number of minimization majorization iteartions to run.

  Returns:
    the mse of the learned regressor over the test set.
  """
  num_observed = observed_test.shape[1]
  inverse_covariance, _ = (
      structured_elliptical_maximum_likelihood(train_data.T, loss, loss_grad,
                                               edge_indices, initial_value=None,
                                               max_iters=num_steps_mm,
                                               newton_num_steps=num_steps_mm_newton))
  regressor = get_regressor_from_inverse_covariance(inverse_covariance,
                                                    num_observed)
  observed_train = train_data[:, :num_observed]
  hidden_train = train_data[:, num_observed:]
  # small touch - fit a multiplicative scalar to the regressor that minimizes
  # the squared error.
  least_squares_scaling = fit_least_squares_scalar_multiple(regressor,
                                                            observed_train,
                                                            hidden_train)
  return get_mse(observed_test, hidden_test, stds,
                 least_squares_scaling*regressor)


def robust_unstructured_estimator(loss, loss_grad, train_data, stds,
                                  observed_test, hidden_test, num_steps_mm):
  """Runs an unstructured method with a robust loss.

  We just perform minimization majorization using the sample covariance.

  Args:
    loss: the robust loss function to use.
    loss_grad: the gradient function of the robust loss.
    train_data: array of size [num_examples, num_stocks].
    stds: standard deviation of all stocks on the training data, first items
      are for observed stocks, last ones are for hidden ones.
    observed_test: array of size [num_test_examples, num_observed_stocks].
    hidden_test: array of size [num_test_examples, num_hidden_stocks].
    num_steps_mm: maximum number of minimization majorization iteartions to run.

  Returns:
    the mse of the learned regressor over the test set.
  """
  num_observed = observed_test.shape[1]
  inverse_covariance, _ = (
      non_structured_elliptical_maximum_likelihood(train_data.T, loss,
                                                   loss_grad,
                                                   max_iters=num_steps_mm))
  regressor = get_regressor_from_inverse_covariance(inverse_covariance,
                                                    num_observed)
  observed_train = train_data[:, :num_observed]
  hidden_train = train_data[:, num_observed:]
  # small touch - fit a multiplicative scalar to the regressor that minimizes
  # the squared error.
  least_squares_scaling = fit_least_squares_scalar_multiple(regressor,
                                                            observed_train,
                                                            hidden_train)
  return get_mse(observed_test, hidden_test, stds,
                 least_squares_scaling*regressor)


def run_experiment(seed1, seed2, observed_train, hidden_train, observed_test,
                   hidden_test, edge_indices, stds, num_steps_newton,
                   num_steps_mm_newton, num_steps_mm):
  """Runs a single test with all the methods and saves the results.

  Args:
    seed1: seed to use for the random division of stocks into observed and
      hidden ones. for reproducibility purposes.
    seed2: seed to use for the random shuffle of training data. for
      reproducibility purposes.
    observed_train: array of size [num_train_examples, num_observed_sotcks],
      holds training data for observed sotcks.
    hidden_train: array of size [num_train_examples, num_hidden_sotcks],
      holds training data for hidden sotcks.
    observed_test: array of size [num_test_examples, num_observed_sotcks],
      holds test data for observed sotcks.
    hidden_test: array of size [num_test_examples, num_hidden_sotcks],
      holds test data for hidden sotcks.
    edge_indices: list of edges to use for the graphical structure.
      An edge is itself a list of two integers in the range [0..num_stocks-1].
      Includes self edges (i.e. [i,i]) for digonal elements of the inverse
      covariance.
    stds: standard deviation of all stocks on the training data, first items
      are for observed stocks, last ones are for hidden ones.
    num_steps_newton: maximum number of coordinate descent iterations to perform
      in the newton optimizer for non-robust methods.
    num_steps_mm_newton: maximum number of coordinate descent iterations to
      perform in the newton optimizer within each minimizaiton majorrization
      iteration.
    num_steps_mm: maximum number of minimization majorization iteartions to run.
  """

  [m_train, num_observed] = observed_train.shape
  [_, num_hidden] = hidden_train.shape
  print('==== seed1={}, seed2={}, nx={}, m_train={}, '.format(seed1, seed2,
                                                              num_observed,
                                                              m_train))
  # prepare directory to save results.
  full_dir = os.path.join(FLAGS.save_dir, '%d_%d'%(num_observed, m_train))
  full_dir = os.path.join(full_dir, '%d_%d'%(seed1, seed2))

  if tf.gfile.Exists(full_dir):
    if FLAGS.delete_existing:
      tf.gfile.DeleteRecursively(full_dir)
  tf.gfile.MakeDirs(full_dir)

  # run square-loss (i.e. gaussian) methods and save results.
  data_train = np.column_stack([observed_train, hidden_train])
  robust_losses_dict = get_losses_dictionary(num_hidden + num_observed)

  # run all the unstructured estimators
  cur_dir = os.path.join(full_dir, 'full')
  if tf.gfile.Exists(cur_dir):
    if FLAGS.delete_existing:
      tf.gfile.DeleteRecursively(cur_dir)
  tf.gfile.MakeDirs(cur_dir)
  # gaussian
  gauss_unstruct_mse = (
      sample_covariance(data_train, stds, observed_test, hidden_test))
  fname = os.path.join(cur_dir, '%s.npy' % 'gmrf_mse')
  print('fname', fname)
  with tf.gfile.Open(fname, 'w') as fp:
    print(gauss_unstruct_mse)
    np.save(fp, gauss_unstruct_mse)
  # robust
  for estimator_name, (loss, loss_grad) in robust_losses_dict.items():
    cur_err = robust_unstructured_estimator(loss, loss_grad, data_train, stds,
                                            observed_test, hidden_test,
                                            num_steps_mm)
    fname = os.path.join(cur_dir, '%s.npy' % (estimator_name+'_mse'))
    print('fname', fname)
    with tf.gfile.Open(fname, 'w') as fp:
      print(cur_err)
      np.save(fp, cur_err)

  # run all the structured estimators
  cur_dir = os.path.join(full_dir, 'glasso')
  if tf.gfile.Exists(cur_dir):
    if FLAGS.delete_existing:
      tf.gfile.DeleteRecursively(cur_dir)
  tf.gfile.MakeDirs(cur_dir)
  # gaussian
  gmrf_mse = gmrf(data_train, edge_indices, stds, observed_test, hidden_test,
                  num_steps_newton)

  fname = os.path.join(cur_dir, '%s.npy' % 'gmrf_mse')
  print('fname', fname)
  with tf.gfile.Open(fname, 'w') as fp:
    print(gmrf_mse)
    np.save(fp, gmrf_mse)
  # robust
  for estimator_name, (loss, loss_grad) in robust_losses_dict.items():
    cur_err = robust_estimator(loss, loss_grad, data_train, edge_indices, stds,
                               observed_test, hidden_test, num_steps_mm_newton,
                               num_steps_mm)
    fname = os.path.join(cur_dir, '%s.npy' % (estimator_name+'_mse'))
    print('fname', fname)
    with tf.gfile.Open(fname, 'w') as fp:
      print(cur_err)
      np.save(fp, cur_err)


def main(argv):
  del argv  # Unused.
  tf.logging.set_verbosity(tf.logging.INFO)
  print(tf.__version__)
  seed1 = FLAGS.seed1
  seed2 = FLAGS.seed2
  num_steps_newton = FLAGS.num_steps_newton
  num_steps_mm_newton = FLAGS.num_steps_mm_newton
  num_steps_mm = FLAGS.num_steps_mm
  num_observed = FLAGS.num_observed
  num_stocks = FLAGS.num_stocks

  [
      observed_train, hidden_train, observed_test, hidden_test, edge_indices,
      sigma
  ] = (
      load_data_and_structure(seed1, num_stocks, num_observed))
  np.random.seed(seed2)
  perm_samples = np.random.permutation(observed_train.shape[0])
  m_trains = [
      15, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700,
      750, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800
  ]
  for m_train in m_trains:
    run_experiment(seed1, seed2, observed_train[perm_samples[:m_train], :],
                   hidden_train[perm_samples[:m_train], :], observed_test,
                   hidden_test, edge_indices, sigma, num_steps_newton,
                   num_steps_mm_newton, num_steps_mm)

if __name__ == '__main__':
  app.run(main)
