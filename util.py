"""utilities for covariance estimation experiments.
"""

import numpy as np


def get_edge_indices_from_adjmat(adjacency_matrix):
  """Gets a list of indices that are non-zero in adjacency_matrix.

  Args:
    adjacency_matrix: a 2-dim numpy array to get the edges from.

  Returns:
    A list with all the indices [i, j] where adjacency_matrix[i, j] is non-zero.
    Each item in the list is a list of two integers.
  """
  [nx, ny] = adjacency_matrix.shape
  edge_indices = []
  for i in range(nx):
    for j in range(ny):
      if np.abs(adjacency_matrix[i, j]) > 0:
        edge_indices.append([i, j])
  edge_indices = np.array(edge_indices)
  return edge_indices


def get_edge_indices_from_adjmat_symmetric(adjacency_matrix):
  """Gets a list of indices that are non-zero in symmetric adjacency_matrix.

  Args:
    adjacency_matrix: a 2-dim numpy array to get the edges from.

  Returns:
    A list with all the indices [i, j] and [j,i] where adjacency_matrix[i, j]
    is non-zero and j>i. Each item in the list is a list of two integers.
  """
  assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1]
  n = adjacency_matrix.shape[0]
  edge_indices = []
  for i in range(n):
    for j in np.arange(i+1, n):
      if np.abs(adjacency_matrix[i, j]) > 0:
        edge_indices.append([i, j])
  edge_indices = np.array(edge_indices)
  edge_indices = np.vstack([edge_indices, np.fliplr(edge_indices)])
  return edge_indices


def fit_least_squares_scalar_multiple(regressor, features_train, labels_train):
  """Fits a multiplicative scalar to a linear regressor, to minimize l2 loss.

  This slightly improves regression results.

  Args:
    regressor: a linear regressor from features to labels. array of shape
      [num_labels, num_features].
    features_train: the features in the training set. array of shape
      [num_examples, num_features].
    labels_train: the labels in the training set. array of shape
      [num_examples, num_labels].

  Returns:
    a multiplicative scalar 'c' that minimizes -
    || labels_train - c*regressor@features_train ||_F
  """
  predictions_train = regressor.dot(features_train.T)
  inner_prods_with_labels = np.trace(labels_train.T.dot(predictions_train.T))
  norms_of_predictions = np.trace(predictions_train.dot(predictions_train.T))
  least_squares_scaling = inner_prods_with_labels/norms_of_predictions
  return least_squares_scaling


def get_regressor_from_inverse_covariance(inverse_covariance, num_features):
  """Gets an optimal linear regressor from an inverse covariance matrix.

  Assuming the number of features is num_features and number of labels is
  num_labels, the first num_features rows/colums of inverse_covariance
  should correspond to the features. While the last num_labels rows/columns
  correspond to labels.

  Args:
    inverse_covariance: symmetric positive definite matrix of shape
      [num_features+num_labels, num_features+num_labels], holding an estimate
      inverse covariance matrix of the features and labels.
    num_features: number of features to perform regression from. it is assumed
      that we would like to get a regressor from the variables that correspond
      to the first num_features rows/columns of inverse_covariance, to the
      variables corresponding to the other rows/columns.

  Returns:
    a linear regressor from features to labels.
  """

  gamma_yy = inverse_covariance[num_features:, num_features:]
  gamma_yx = inverse_covariance[num_features:, :num_features]
  regressor = -np.linalg.inv(gamma_yy).dot(gamma_yx)

  return regressor


def standardize_data(data):
  """Reduces mean from data and divides by its standard deviation.

  Args:
    data: array of size [num_example, num_features].
  Returns:
    standardized_data: input data after standardization.
    means: the means of the input data, that were reduced to create the
      standardized version.
    stds: the standard deviation of each feature of the input data, that were
      divided by to create the standardized version.
  """

  means = np.mean(data, axis=0, keepdims=True)
  stds = np.std(data, axis=0, keepdims=True)
  standardized_data = (data - means)/stds

  return standardized_data, means, stds
