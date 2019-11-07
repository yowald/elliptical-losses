"""samplers for scale variables of elliptical distributions.
"""

import numpy as np


class PositiveScalarSamplerFactory():

  """ corresponds to the samples creating a multivariate Gaussian distribution """
  def root_chi_square(self, params):
    sampler = lambda m: np.sqrt(np.random.chisquare(params['df'], size=(m, 1)))
    return sampler

  def exponential(self, params):
    sampler = lambda m: np.random.exponential(params['scale'], size=(m, 1))
    return sampler

  def laplace(self, params):
    sampler = lambda m: np.sqrt(np.random.laplace(scale=params['scale'],
                                                  size=(m,1))**2)
    return sampler

  def multivariate_t(self, params):
    d = params['dim']
    nu = params['nu']
    sampler = lambda m: np.sqrt(d*np.random.f(d, nu, m))
    return sampler

  def generalized_gaussian(self, params):
    """Returns a sampler for a 2*beta'th root of a Gamma distribution.

    When incorporated as the scalar sampler of an Elliptical distribution, this
    creates the Generalized Gausian distribution, from:
    Pascal et al. - Parameter Estimation For Multivariate Generalized Gaussian
    Distributions. IEEE trans on SP 2017.

    Args:
      params: Dictionary with required parameters for the sampler. Here this is
      the shape of a Gamma distribution and the dimension of the corresponding
      multivariate distribution. Key names should be 'shape' and 'dim'.

    Returns:
      sampler - a scalar sampler of a Gamma distribution.
    """
    beta = params['shape']
    p = params['dim']
    sampler = lambda m: np.power(np.random.gamma(p/(2*beta), scale=2., size=(m, 1)),
                                 1./(2*beta))
    return sampler
