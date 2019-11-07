"""Functions to load floods data and run experiments
"""

from absl import app
from absl import flags

import os

import numpy as np
import tensorflow as tf
import pickle
from scipy.linalg import toeplitz as toeplitz


# change structured optimizers path to suitable path
from ..structured_optimizers import GMRFOptimizer
from ..structured_optimizers import LossFunctionFactory
from ..structured_optimizers import non_structured_elliptical_maximum_likelihood
from ..structured_optimizers import structured_elliptical_maximum_likelihood
from ..util import get_edge_indices_from_adjmat_symmetric
from ..util import fit_least_squares_scalar_multiple
from ..util import standardize_data
from ..util import get_regressor_from_inverse_covariance
import datetime

FLAGS = flags.FLAGS
tf.app.flags.DEFINE_string('save_dir',
                           './elliptical-losses/floods/results/',
                           'Directory where to write event logs '
                           'and checkpoint.')
tf.app.flags.DEFINE_integer('seed', 105, '')
tf.app.flags.DEFINE_boolean('delete_existing', False, 'Delete existing  '
                            'directory if exits.')
tf.app.flags.DEFINE_integer('num_steps_newton', 5000,
                            """Number of steps for newton optimizer.""")
tf.app.flags.DEFINE_integer('num_steps_mm_newton', 1000,
                            """Number of steps or newton in MM algorithm.""")
tf.app.flags.DEFINE_integer('num_steps_mm', 5,
                            """Number of steps for MM algorithm.""")

tf.app.flags.DEFINE_string('structure_type', 'full',
                           'type of structure to use. possible values are:'
                           'full - fit entire convariance matrix'
                           'time - use each sites features to predict its'
                           'discharge, and no other features.'
                           'time-space - use an additional neighboring site'
                           'time-space-3nn - use features of two neighboring'
                           'sites')


def get_edge_lists(num_sites=34, num_features_per_site=2,
                   num_labels_per_site=3):
  """Gets a dictionary with edge indices for structures used in this dataset.

  The key is the name of the structure, values are numpy array of shape
  [num_edges, 2], where each row is the indices of an edge in the graphical
  structure.

  Args:
    num_sites: the number of sites we have in the dataset.
    num_features_per_site: the number of features per site that we use to
      predict the labels.
    num_labels_per_site: the number of forward time steps for which we predict
      discharge values.

  Returns:
    edge_lists_dictionary: dictionary of edge lists for all the structures we
      consider in the task.
  """

  edge_lists_dictionary = {}
  # Each site has 2 features (discharge and precipitation at time t) and 3
  # labels (discharge at times t+1, t+2, t+3). The following builds an adjacency
  # matrix between these variables.
  # The chosen structure connects precipitation to all other variables and
  # uses a temporal chain structure for the discharge variables.
  intra_site_adjacency = np.zeros(num_features_per_site + num_labels_per_site)
  intra_site_adjacency[0] = 1.
  intra_site_adjacency[1] = 1.
  intra_site_adjacency = toeplitz(intra_site_adjacency)
  intra_site_adjacency[0, :] = 1.
  intra_site_adjacency[:, 0] = 1.

  num_variables = num_sites*(num_features_per_site + num_labels_per_site)
  diag_edges_list = [[i, i] for i in range(num_variables)]

  # First type of structure - each site just uses its features to predict 3
  # steps forward.

  # Duplicate the intra-site adjacency using a Kronecker product to create the
  # full adjacency matrix
  adjacency_matrix_temporal = np.kron(intra_site_adjacency, np.eye(num_sites))
  edge_lists_dictionary['time'] = get_edge_indices_from_adjmat_symmetric(
      adjacency_matrix_temporal)
  edge_lists_dictionary['time'] = np.vstack(
      [edge_lists_dictionary['time'], diag_edges_list])

  # Second type of structure - each site also uses features of its adjacent
  # sites (the sites' order in the data somewhat reflects physical proximity).
  site_adjacency_one_neighbor = np.zeros(num_sites)
  site_adjacency_one_neighbor[:2] = 1
  adjacency_matrix_one_nei = np.kron(intra_site_adjacency,
                                     toeplitz(site_adjacency_one_neighbor))
  edge_lists_dictionary['time-space'] = get_edge_indices_from_adjmat_symmetric(
      adjacency_matrix_one_nei)
  edge_lists_dictionary['time-space'] = np.vstack(
      [edge_lists_dictionary['time-space'], diag_edges_list])

  # Third type of structure - use two neighboring sites
  site_adjacency_three_neighbors = np.zeros(num_sites)
  site_adjacency_three_neighbors[:4] = 1
  adjacency_matrix_three_nei = np.kron(intra_site_adjacency,
                                       toeplitz(site_adjacency_three_neighbors))
  edge_lists_dictionary[
      'time-space-3nn'] = get_edge_indices_from_adjmat_symmetric(
          adjacency_matrix_three_nei)
  edge_lists_dictionary['time-space-3nn'] = np.vstack(
      [edge_lists_dictionary['time-space-3nn'], diag_edges_list])

  return edge_lists_dictionary


def load_data_and_structure(structure_type):
  """Loads data and structure for floods data.

  Args:
    structure_type: the type of structure to get edge indices for. see the
      'structure_type' flag for options of structures.

  Returns:
    features: array of shape [num_examples, num_features] holding all the
      features in the data.
    labels: array of shape [num_examples, num_labels] holding all the labels in
      the data.
    edge_indices: list of edges of a graphical structure.
      An edge is itself a list of two integers in the range
      [0..num_features+num_labels-1]. Structures include self edges (i.e. [i,i])
      for diagonal elements of the inverse covariance.
  """
  base_dir = './elliptical-losses/floods/data/'
  with open(base_dir+'floods_data.pickle', 'rb') as data_file:
    discharge_dict = pickle.load(data_file, encoding='latin1')
    data_file.close()
  # data_file = tf.gfile.Open(base_dir+'floods_data.pickle', 'r')

  discharge_array = discharge_dict['discharge'].T
  precipitation_array = discharge_dict['precipitation'].T

  features = np.concatenate((precipitation_array[:-3], discharge_array[:-3]),
                            axis=1)
  labels = np.concatenate((discharge_array[1:-2], discharge_array[2:-1],
                           discharge_array[3:]), axis=1)

  edges_dict = get_edge_lists()
  edge_indices = None if structure_type == 'full' else edges_dict[structure_type]

  return features, labels, edge_indices


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


def get_nmse(features_test, labels_test, regressor, features_mean, labels_mean,
             features_std=None, labels_std=None):
  """Returns the nmse of a regressor on the test data.

  Args:
    features_test: features of test data.
    labels_test: labels of test data.
    regressor: the linear regressor to get the error for. regressor is trained
      to give predictions on standardized data.
    features_mean: mean of all features on the training data.
    labels_mean: mean of all labels on the training data.
    features_std: standard deviation of all features on the training data.
    labels_std: standard deviation of all labels on the training data.

  Returns: the normalized mse of the regressor on the test data.
  """
  if features_std is not None or labels_std is not None:
    standardized_features = (features_test.T - features_mean.T)/features_std.T
    standardized_predictions = regressor.dot(standardized_features)
    raw_predicions = standardized_predictions*labels_std.T + labels_mean.T
    errs = labels_test.T - raw_predicions
  else:
    centered_features = features_test.T - features_mean.T
    centered_predictions = regressor.dot(centered_features)
    errs = labels_test.T - (centered_predictions + labels_mean.T)
  return np.mean(errs**2)/np.mean(labels_test**2)


def gmrf(train_data, edge_indices, features_test, labels_test,
         num_steps_newton):
  """Runs the structured square loss method.

  Args:
    train_data: array of size [num_examples, num_features+num_labels].
    edge_indices: list of edges to use for the graphical structure.
      An edge is itself a list of two integers in the range
      [0..num_features+num_labels-1]. Includes self edges (i.e. [i,i]) for
      diagonal elements of the inverse covariance.
    features_test: array of size [num_test_examples, num_features].
    labels_test: array of size [num_test_examples, num_labels].
    num_steps_newton: maximum number of coordinate descent iterations to perform
      in the newton optimizer.

  Returns:
    the nmse of the learned regressor over the test set.
  """
  num_features = features_test.shape[1]
  num_labels = labels_test.shape[1]
  gmrf_optimizer = GMRFOptimizer(num_features + num_labels, edge_indices)
  standard_train_data, means, stds = standardize_data(train_data)

  inverse_covariance, _ = gmrf_optimizer.alt_newton_coord_descent(
      standard_train_data.T, max_iter=num_steps_newton)
  regressor = get_regressor_from_inverse_covariance(inverse_covariance,
                                                    num_features)
  features_train = standard_train_data[:, :num_features]
  labels_train = standard_train_data[:, num_features:]

  least_squares_scaling = fit_least_squares_scalar_multiple(regressor,
                                                            features_train,
                                                            labels_train)

  return get_nmse(features_test, labels_test, least_squares_scaling*regressor,
                  means[:, :num_features], means[:, num_features:],
                  features_std=stds[:, :num_features],
                  labels_std=stds[:, num_features:])


def sample_covariance(train_data, features_test, labels_test):
  """Runs the unstructured square loss method.

  This is equivalent to using the sample covariance, hence the name of the
  method.

  Args:
    train_data: array of size [num_examples, num_features + num_labels].
    features_test: array of size [num_test_examples, num_features].
    labels_test: array of size [num_test_examples, num_labels].

  Returns:
    the mse of the learned regressor over the test set.
  """
  m = train_data.shape[0]
  num_features = features_test.shape[1]
  standard_train_data, means, stds = standardize_data(train_data)
  # calculate sample covariance matrix.
  covariance = standard_train_data.T.dot(standard_train_data)/m
  # construcut regressor from covariance matrix.
  sigma_yx = covariance[num_features:, :num_features]
  sigma_xx_inv = np.linalg.inv(covariance[:num_features, :num_features])
  regressor = sigma_yx.dot(sigma_xx_inv)
  # get predictions on training data.
  features_train = standard_train_data[:, :num_features]
  labels_train = standard_train_data[:, num_features:]
  # small touch - fit a multiplicative scalar to the regressor that minimizes
  # the squared error.
  least_squares_scaling = fit_least_squares_scalar_multiple(regressor,
                                                            features_train,
                                                            labels_train)
  return get_nmse(features_test, labels_test, least_squares_scaling*regressor,
                  means[:, :num_features], means[:, num_features:],
                  features_std=stds[:, :num_features],
                  labels_std=stds[:, num_features:])


def robust_estimator(loss, loss_grad, train_data, edge_indices, features_test,
                     labels_test, num_steps_mm_newton, num_steps_mm):
  """Runs a structured method with a robust loss.

  Args:
    loss: the robust loss function to use.
    loss_grad: the gradient function of the robust loss.
    train_data: array of size [num_examples, num_features + num_labels].
    edge_indices: list of edges to use for the graphical structure.
      An edge is itself a list of two integers in the range
      [0..num_features + num_labels - 1]. Includes self edges (i.e. [i,i]) for
      diagonal elements of the inverse covariance.
    features_test: array of size [num_test_examples, num_features].
    labels_test: array of size [num_test_examples, num_labels].
    num_steps_mm_newton: maximum number of coordinate descent iterations to
      perform in the newton optimizer within each minimizaiton majorrization
      iteration.
    num_steps_mm: maximum number of minimization majorization iteartions to run.

  Returns:
    the mse of the learned regressor over the test set.
  """
  num_features = features_test.shape[1]
  standard_train_data, means, stds = standardize_data(train_data)
  inverse_covariance, _ = (
      structured_elliptical_maximum_likelihood(standard_train_data.T, loss,
                                               loss_grad, edge_indices,
                                               initial_value=None,
                                               max_iters=num_steps_mm,
                                               newton_num_steps=num_steps_mm_newton))
  regressor = get_regressor_from_inverse_covariance(inverse_covariance,
                                                    num_features)
  features_train = standard_train_data[:, :num_features]
  labels_train = standard_train_data[:, num_features:]
  # small touch - fit a multiplicative scalar to the regressor that minimizes
  # the squared error.
  least_squares_scaling = fit_least_squares_scalar_multiple(regressor,
                                                            features_train,
                                                            labels_train)
  return get_nmse(features_test, labels_test, least_squares_scaling*regressor,
                  means[:, :num_features], means[:, num_features:],
                  features_std=stds[:, :num_features],
                  labels_std=stds[:, num_features:])


def robust_unstructured_estimator(loss, loss_grad, train_data,
                                  features_test, labels_test, num_steps_mm):
  """Runs an unstructured method with a robust loss.

  We just perform minimization majorization using the sample covariance.

  Args:
    loss: the robust loss function to use.
    loss_grad: the gradient function of the robust loss.
    train_data: array of size [num_examples, num_features + num_labels].
    features_test: array of size [num_test_examples, num_features].
    labels_test: array of size [num_test_examples, num_labels].
    num_steps_mm: maximum number of minimization majorization iteartions to run.

  Returns:
    the mse of the learned regressor over the test set.
  """
  num_features = features_test.shape[1]
  standard_train_data, means, stds = standardize_data(train_data)
  inverse_covariance, _ = (
      non_structured_elliptical_maximum_likelihood(standard_train_data.T, loss,
                                                   loss_grad,
                                                   max_iters=num_steps_mm))
  regressor = get_regressor_from_inverse_covariance(inverse_covariance,
                                                    num_features)
  features_train = standard_train_data[:, :num_features]
  labels_train = standard_train_data[:, num_features:]
  # small touch - fit a multiplicative scalar to the regressor that minimizes
  # the squared error.
  least_squares_scaling = fit_least_squares_scalar_multiple(regressor,
                                                            features_train,
                                                            labels_train)
  return get_nmse(features_test, labels_test, least_squares_scaling*regressor,
                  means[:, :num_features], means[:, num_features:],
                  features_std=stds[:, :num_features],
                  labels_std=stds[:, num_features:])


def run_experiment(seed, features_train, labels_train, features_test,
                   labels_test, edge_indices, num_steps_newton,
                   num_steps_mm_newton, num_steps_mm, structure_type):
  """Runs a single test with all the methods and saves the results.

  Args:
    seed: seed we used for the shuffle of data. for reproducibility purposes.
    features_train: array of size [num_train_examples, num_features],
      holds training features.
    labels_train: array of size [num_train_examples, num_labels],
      holds training labels.
    features_test: array of size [num_test_examples, num_features],
      holds test features.
    labels_test: array of size [num_test_examples, num_labels],
      holds test labels.
    edge_indices: list of edges to use for the graphical structure.
      An edge is itself a list of two integers in the range
      [0..num_features + num_labels - 1]. Includes self edges (i.e. [i,i]) for
      digonal elements of the inverse covariance.
    num_steps_newton: maximum number of coordinate descent iterations to perform
      in the newton optimizer for non-robust methods.
    num_steps_mm_newton: maximum number of coordinate descent iterations to
      perform in the newton optimizer within each minimizaiton majorrization
      iteration.
    num_steps_mm: maximum number of minimization majorization iteartions to run.
    structure_type: the name of the structure type we use in this experiment.
  """
  print('==== seed={}, nx={}, m_train={},'
        ' structure={}'.format(seed,
                               features_train.shape[1],
                               features_train.shape[0],
                               structure_type))

  [m_train, num_features] = features_train.shape
  [_, num_labels] = labels_train.shape

  full_dir = os.path.join(FLAGS.save_dir, '%s_%d' % (structure_type, m_train))
  full_dir = os.path.join(full_dir, '%d' % (seed))

  if tf.gfile.Exists(full_dir):
    if FLAGS.delete_existing:
      tf.gfile.DeleteRecursively(full_dir)
  tf.gfile.MakeDirs(full_dir)

  train_data = np.column_stack([features_train, labels_train])
  robust_losses_dict = get_losses_dictionary(num_features + num_labels)
  # unstructured methods use sample covariance based algorithms.
  # structured methods use newton coordinate descent.
  if structure_type == 'full':
    # run gaussian unstructured
    gauss_unstruct_mse = sample_covariance(train_data, features_test,
                                           labels_test)
    fname = os.path.join(full_dir, '%s.npy' % 'gmrf_mse')
    print('fname', fname)
    with tf.gfile.Open(fname, 'w') as fp:
      print(gauss_unstruct_mse)
      np.save(fp, gauss_unstruct_mse)
    # run robust unstructured methods
    for estimator_name, (loss, loss_grad) in robust_losses_dict.items():
      cur_err = robust_unstructured_estimator(loss, loss_grad, train_data,
                                              features_test, labels_test,
                                              num_steps_mm)
      fname = os.path.join(full_dir,
                           '%s.npy' % (estimator_name + '_mse'))
      print('fname', fname)
      with tf.gfile.Open(fname, 'w') as fp:
        print(cur_err)
        np.save(fp, cur_err)
  else:
    # run square-loss (i.e. gaussian) methods and save results.
    gmrf_mse = gmrf(train_data, edge_indices, features_test, labels_test,
                    num_steps_newton)
    fname = os.path.join(full_dir, '%s.npy' % 'gmrf_mse')
    print('fname', fname)
    with tf.gfile.Open(fname, 'w') as fp:
      print(gmrf_mse)
      np.save(fp, gmrf_mse)
    for estimator_name, (loss, loss_grad) in robust_losses_dict.items():
      cur_err = robust_estimator(loss, loss_grad, train_data, edge_indices,
                                 features_test, labels_test,
                                 num_steps_mm_newton, num_steps_mm)
      fname = os.path.join(full_dir,
                           '%s.npy' % (estimator_name+'_mse'))
      print('fname', fname)
      with tf.gfile.Open(fname, 'w') as fp:
        print(cur_err)
        np.save(fp, cur_err)


def main(argv):
  del argv  # Unused.
  tf.logging.set_verbosity(tf.logging.INFO)
  print(tf.__version__)
  seed = FLAGS.seed
  num_steps_newton = FLAGS.num_steps_newton
  num_steps_mm_newton = FLAGS.num_steps_mm_newton
  num_steps_mm = FLAGS.num_steps_mm
  structure_type = FLAGS.structure_type

  [features, labels, edge_indices] = load_data_and_structure(structure_type)
  np.random.seed(seed)
  perm_samples = np.random.permutation(features.shape[0])

  num_samples_train_list = np.arange(40, 410, 10)
  for m_train in num_samples_train_list:
    run_experiment(seed, features[perm_samples[:m_train], :],
                   labels[perm_samples[:m_train], :],
                   features[perm_samples[m_train:], :],
                   labels[perm_samples[m_train:], :],
                   edge_indices,
                   num_steps_newton,
                   num_steps_mm_newton,
                   num_steps_mm,
                   structure_type)

if __name__ == '__main__':
  app.run(main)
