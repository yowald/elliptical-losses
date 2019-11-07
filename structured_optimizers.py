"""Minimization-majorization algorithms for robust maximum likelihood problems.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.linalg import cho_factor
from scipy.linalg import cho_solve
from scipy.linalg import LinAlgError
import tensorflow as tf


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


def chol_inv(cho_part, lower=True):
  """Given a matrix's Cholesky decomposition, returns its inverse.

  Args:
    cho_part: Cholesky decomposition of the matrix to invert, as given
      by cho_solve.

    lower: True if the given Cholesky factor is the lower triangular part,
      False if it is the upper part.

  Returns:
    Inverse of a matrix whose Cholesky deocmposition is cho_part.
  """
  return cho_solve((cho_part, lower), np.eye(cho_part.shape[0]))


def inv(matrix):
  """Inversion of a SPD matrix using Cholesky decomposition.

  Args:
    matrix: matrix to invert.

  Returns:
    inverted matrix.
  """
  return chol_inv(check_pd(matrix)[1])


def log(x):
  """Override the numpy log function to return -inf for non-positive values.

  Args:
    x: argument for log.

  Returns:
    logarithm of x.
  """
  return np.log(x) if x > 0 else -np.inf


class LossFunctionFactory(object):
  """Creates loss and gradient function of elliptical losses.
  """

  def __init__(self):
    self.params = None

  def gaussian_mle(self):
    """Returns loss and gradient for likelihood of a multivariate gaussian.
    """
    g = lambda z: z
    grad = lambda z: 1
    return g, grad

  def hubers_loss(self, params):
    """Returns loss and gradient for a Huber style function.

    Args:
      params: Dictionary with loss parameters. Items are 'delta': location
        parameter to switch loss from quadratic to linear, 'd': dimension of the
        problem.
    """
    def g(z):
      if np.abs(z) < params['delta']:
        return np.sqrt(params['d'])*z
      else:
        return np.sqrt(params['d'])*np.sqrt(2*params['delta']*np.abs(z)
                                            - params['delta']**2)
    def grad(z):
      if np.abs(z) < params['delta']:
        return np.sqrt(params['d'])*np.sign(z)
      else:
        return np.sqrt(params['d'])*(
            (1./np.sqrt(2*np.abs(z)*params['delta'] - params['delta']**2))
            *params['delta']*np.sign(z))
    return g, grad

  def tylers_estimator(self, params):
    """Returns loss and gradient for maximum likelihood of an angular distribution.

    Args:
      params: Dictionary with loss parameters. Items are 'd': dimension of the
        problem.
    """
    g = lambda z: params['d']*np.log(z)
    grad = lambda z: params['d']/z
    return g, grad

  def generalized_gaussian(self, params):
    """Returns loss and gradient for maximum likelihood of a generalized gaussian distribution.

    Args:
      params: Dictionary with loss parameters. Items are 'beta': shape parameter
        ,'m': scale parameter. See https://arxiv.org/pdf/1302.6498.pdf for
        definition.
    """
    def g(z):
      return np.float_power(z, params['beta'])/(params['m']**params['beta'])

    def grad(z):
      return (params['beta']*(np.float_power(z, (params['beta']-1)))
              /(params['m']**params['beta']))
    return g, grad

  def multivariate_t(self, params):
    """Returns loss and gradient for maximum likelihood of a multivariate-t distribution.

    Args:
      params: Dictionary with loss parameters. Items are 'nu': degrees of
        freedom, 'd': dimension of the problem.
    """
    def g(z):
      return ((params['nu'] + params['d']))*np.log(1+z/params['nu'])
    def grad(z):
      return (((params['nu'] + params['d'])/(params['nu']))
              *(1./(1+z/params['nu'])))
    return g, grad


class GMRFOptimizer(object):
  """Newton Coordinate Descent optimizer for a Gaussian Markov Random Field.

  Code is adapted from this Github repository: https://github.com/dswah/sgcrfpy
  """

  def __init__(self, d, edge_indices, learning_rate=0.5):
    self.inverse_covariance = np.eye(d)
    self.covariance = np.eye(d)
    self.d = d
    self.sample_covariance = None
    self.edges = edge_indices
    self.learning_rate = learning_rate
    # step size reduction factor for line search
    self.beta = 0.5
    self.slack = 0.05

  def set_inverse_covariance(self, inverse_covariance):
    self.inverse_covariance = inverse_covariance
    self.covariance = np.linalg.inv(inverse_covariance)

  def line_search(self, direction):
    """Backtracking line search to find a direction that keeps us in the PSD cone.

    Args:
      direction: Descent direction found by coordinate descent.
    Returns:
      next_point: New point found by line search.
      alpha: The step size taken by line search.
    """
    # returns cholesky decomposition of Lambda and the learning rate
    alpha = self.learning_rate
    while True:
      new_point = self.inverse_covariance + alpha * direction
      if not np.isfinite(new_point).all():
        alpha = alpha * self.beta
        continue
      pd, next_point = check_pd(new_point)
      if pd and self.check_descent(direction, alpha):
        # step is positive definite and we have sufficient descent
        break
      alpha = alpha * self.beta
    return next_point, alpha

  def check_descent(self, direction, alpha):
    """Checks if the given direction and step size give sufficient descent.

    Args:
      direction: the descent direction to take.
      alpha: step size.
    Returns:
      Whether the step gives sufficient descent or not.
    """
    grad_inverse_covariance = self.sample_covariance - self.covariance

    direction_similarity = np.trace(np.dot(grad_inverse_covariance,
                                           direction))

    nll_a = self.neg_log_likelihood_wrt_inverse(self.inverse_covariance +
                                                alpha * direction)
    nll_b = self.neg_log_likelihood_wrt_inverse(self.inverse_covariance) + (
        alpha * self.slack*direction_similarity)
    return nll_a <= nll_b and np.isfinite(nll_a)

  def neg_log_likelihood(self):
    # compute the negative log-likelihood of the GMRF for current estimate
    if self.sample_covariance is None:
      return None
    else:
      return (np.trace(self.sample_covariance.dot(self.inverse_covariance))
              -log(np.linalg.det(self.inverse_covariance)))

  def neg_log_likelihood_wrt_inverse(self, cand_inverse_covariance):
    # compute the negative log-likelihood of the GMRF for some candidate matrix
    return -log(np.linalg.det(cand_inverse_covariance)) + (
        np.trace(np.dot(self.sample_covariance, cand_inverse_covariance)))

  def descent_direction_inverse_covariance(self):
    """Gets descent direction for the inverse covariance matrix.

    Returns:
      The descent direction.
    """
    delta = np.zeros_like(self.inverse_covariance)
    log_det_grad = np.zeros_like(self.inverse_covariance)

    sigma = self.covariance

    for i, j in np.random.permutation(np.array(self.edges)):
      # Solves minimization of the objective w.r.t the i,j'th element of the
      # inverse covariance.
      if i > j:
        continue

      if i == j:
        a = sigma[i, i] ** 2
      else:
        a = sigma[i, j] ** 2 + sigma[i, i] * sigma[j, j]

      b = self.sample_covariance[i, j] - (
          sigma[i, j] - np.dot(sigma[i, :], log_det_grad[:, j]))

      # delta holds the update to each coordinate, due to the cooridinate
      # descent operation on it.
      if i == j:
        u = -b/a
        delta[i, i] += u
        log_det_grad[i, :] += u * sigma[i, :]
      else:
        u = -b/a
        delta[j, i] += u
        delta[i, j] += u
        log_det_grad[j, :] += u * sigma[i, :]
        log_det_grad[i, :] += u * sigma[j, :]

    return delta

  def reset_inverse_covariance_estimates(self):
    self.inverse_covariance = np.eye(self.d)
    self.covariance = np.eye(self.d)

  def alt_newton_coord_descent(self, features, max_iter=200,
                               convergence_tolerance=1e-5,
                               initialize_to_sample_covariance=False):
    """Solves the maximum likelihood problem with Newton coordinate descent.

    Args:
      features: matrix of shape [num_features, num_examples] with the data to
        fit a covariance matrix to.
      max_iter: maximum number of coordinate descent iterations to run.
      convergence_tolerance: threshold to determine convergance of the
        algorithm. If the (drop in objective)/(current objective) is smaller
        than convergence_tolerance then we'll say the algorithm converged.
      initialize_to_sample_covariance: if True initializes the estimate of the
        inverse covariance to the inverse of the sample covariance.
    Returns:
      The estimate of the inverse covariance matrix and whether the algorithm
      converged up to desired tolerance or not.
    """
    m = features.shape[1]
    self.sample_covariance = features.dot(features.T) / m
    self.nll = []
    self.lrs = []

    if initialize_to_sample_covariance:
      self.set_inverse_covariance(np.linalg.inv(self.sample_covariance))

    converged_up_to_tolerance = False
    for t in range(max_iter):
      self.nll.append(self.neg_log_likelihood())
      # solve D_lambda via coordinate descent
      descent_direction = self.descent_direction_inverse_covariance()
      if not np.isfinite(descent_direction).all():
        # add a small multiple of identity matrix if matrix is ill-defined.
        eps = 1e-04
        self.covariance = np.linalg.inv(self.inverse_covariance
                                        + eps*np.eye(self.d))
        descent_direction = self.descent_direction_inverse_covariance()
        if not np.isfinite(descent_direction).all():
          tf.logging.info('Newton optimization failed due to overflow.')
          return self.inverse_covariance.copy(), converged_up_to_tolerance

      # line search for best step size
      learning_rate = self.learning_rate
      new_estimate, learning_rate = self.line_search(descent_direction)
      self.lrs.append(learning_rate)
      self.inverse_covariance = self.inverse_covariance.copy() + (
          learning_rate * descent_direction)

      # update variable params
      # use chol decomp from the backtracking
      self.covariance = chol_inv(new_estimate)
      if not np.isfinite(self.covariance).all():
        eps = 1e-04
        self.covariance = np.linalg.inv(self.inverse_covariance
                                        + eps*np.eye(self.d))
        if not np.isfinite(self.covariance).all():
          tf.logging.info('Newton optimization failed due to overflow.')
          return self.inverse_covariance.copy(), converged_up_to_tolerance

      nll_at_new_point = self.neg_log_likelihood()
      descent_made = self.nll[-1] - self.neg_log_likelihood()
      if (np.abs(descent_made)/np.abs(nll_at_new_point) < convergence_tolerance
          and t > 0):
        converged_up_to_tolerance = True
        break
    return self.inverse_covariance.copy(), converged_up_to_tolerance


# Functions for Minimization-Majorization


def scale_dataset(features, inverse_covariance, loss_grad):
  """Scale each example in the dataset according the loss' linear approximation.

  More accurately, scales each example z by:
  sqrt(psi(z.T*inverse_covariance*z))

  Args:
    features: numpy array of shape [num_features, num_examples] holding the
      dataset.
    inverse_covariance: the estimated inverse covariance matrix of the data.
    loss_grad: gradient function of the robust loss used in the minimization
      majorization algorithm.

  Returns:
    The scaled dataset.
  """
  z_vec = np.diag(features.T.dot(inverse_covariance).dot(features))
  scaling_factors = np.array([np.sqrt(loss_grad(z)) for z in z_vec], ndmin=2)
  scaled_features = np.multiply(features, scaling_factors)
  return scaled_features


def elliptical_objective(features, inverse_covariance, loss):
  """Calculate the negative log-likelihood objective of a robust MRF.

  Args:
    features: numpy array of shape [num_features, num_examples] holding the
      dataset.
    inverse_covariance: inverse covariance matrix to calculate the objective
      for.
    loss: the robust loss function to use.

  Returns:
    The negative log-likelihood of the robust MRF at the given point.
  """
  z_vec = np.diag(features.T.dot(inverse_covariance).dot(features))
  mean_g_of_z = np.mean([loss(z) for z in z_vec])
  return mean_g_of_z - log(np.linalg.det(inverse_covariance))


def structured_elliptical_maximum_likelihood(features, loss, loss_grad,
                                             edge_indices, initial_value=None,
                                             max_iters=7, tolerance=1e-4,
                                             newton_num_steps=750):
  """Solves a robust maximum likelihood problem with a graphical structure.

  Args:
    features: numpy array of shape [num_features, num_examples] holding the
      dataset.
    loss: the robust loss function to use.
    loss_grad: gradient function of the robust loss to use.
    edge_indices: list of edges to use for the graphical structure. An edge is
      itself a list of two integers in the range [0..num_features-1]. Should
      include self edges (i.e. [i,i]) for digonal elements of the inverse
      covariance.
    initial_value: array of size [num_features, num_features] holding an initial
      estimate for the inverse covariance. If None then we initialize to the
      identity matrix.
    max_iters: maximum number of iterations of minimization majorization to run.
    tolerance: threshold to determine convergance of the algorithm. If the
      (drop in objective)/(current objective) is smaller than
      convergence_tolerance then we'll say the algorithm converged.
    newton_num_steps: maximum number of steps for the newton algorithm in each
      iteration of the inner loop of the minimization-majorization algorithm.

  Returns:
    the estimate of the inverse covariance and whether the algorithm converged
    or not.
  """
  [d, _] = features.shape
  if initial_value is None:
    inverse_covariance = np.eye(d)
  else:
    inverse_covariance = initial_value

  scaled_features = scale_dataset(features, inverse_covariance, loss_grad)
  gmrf_optimizer = GMRFOptimizer(d, edge_indices)
  gmrf_optimizer.set_inverse_covariance(inverse_covariance)
  converged_up_to_tolerance = False

  inverse_covariance, _ = (
      gmrf_optimizer.alt_newton_coord_descent(scaled_features,
                                              max_iter=newton_num_steps))
  prev_objective = elliptical_objective(features, inverse_covariance, loss)
  for _ in np.arange(max_iters-1):
    scaled_features = scale_dataset(features, inverse_covariance, loss_grad)
    inverse_covariance, _ = (
        gmrf_optimizer.alt_newton_coord_descent(scaled_features,
                                                max_iter=newton_num_steps))
    cur_objective = elliptical_objective(features, inverse_covariance, loss)
    drop_ratio = np.abs(cur_objective-prev_objective)/np.abs(prev_objective)
    if drop_ratio < tolerance:
      converged_up_to_tolerance = True
      return inverse_covariance, converged_up_to_tolerance
    else:
      prev_objective = cur_objective
  return inverse_covariance, converged_up_to_tolerance


def non_structured_elliptical_maximum_likelihood(features, loss, loss_grad,
                                                 max_iters=5, tolerance=1e-4):
  """Solves a robust maximum likelihood problem with no structure.

  This uses minimization-majorization on the sample covariance matrix.

  Args:
    features: numpy array of shape [num_features, num_examples] holding the
      dataset.
    loss: the robust loss function to use.
    loss_grad: gradient function of the robust loss to use.
    max_iters: maximum number of iterations of minimization majorization to run.
    tolerance: threshold to determine convergance of the algorithm. If the
      (drop in objective)/(current objective) is smaller than
      convergence_tolerance then we'll say the algorithm converged.

  Returns:
    the estimate of the inverse covariance and whether the algorithm converged
    or not.
  """
  [d, m] = features.shape
  sample_covariance = features.dot(features.T)/m
  inverse_covariance = np.linalg.inv(sample_covariance)
  eps = 1e-4
  prev_objective = elliptical_objective(features,
                                        inverse_covariance+eps*np.eye(d), loss)
  converged_up_to_tolerance = False
  for _ in range(max_iters):
    scaled_features = scale_dataset(features, inverse_covariance, loss_grad)
    sample_covariance = scaled_features.dot(scaled_features.T)/m
    inverse_covariance = np.linalg.inv(sample_covariance)
    cur_objective = elliptical_objective(features,
                                         inverse_covariance+eps*np.eye(d), loss)
    drop_ratio = np.abs(cur_objective-prev_objective)/np.abs(prev_objective)
    if drop_ratio < tolerance:
      converged_up_to_tolerance = True
      return inverse_covariance, converged_up_to_tolerance
    else:
      prev_objective = cur_objective
  return inverse_covariance, converged_up_to_tolerance
