# Copyright 2021 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Utilities for training energy-based models."""
import jax
import jax.numpy as jnp
import numpy as np

from flax import struct
from functools import partial


@struct.dataclass
class ReplayBuffer(object):
  """Replay buffer for sampling."""
  buffer_size: int
  dims: int
  data: any

  def add(self, samples):
    num_samples = len(samples)
    new_data = jnp.concatenate((samples, self.data[:-num_samples]))
    return self.replace(buffer_size=self.buffer_size,
                        dims=self.dims,
                        data=new_data)

  def sample(self, rng, n, p=0.95):
    """Generates a set of samples. With probability p, each sample
    will come from the replay buffer. With probability (1-p) each
    sample will be sampled from a uniform distribution.
    """
    buf_mask = jax.random.bernoulli(rng, p=p, shape=(n,))[:, jnp.newaxis]
    rand_mask = 1 - buf_mask
    idx = jax.random.choice(rng, self.buffer_size, shape=(n,), replace=False)
    buf = self.data[idx]
    rand = jax.random.uniform(rng, shape=(n, self.dims))
    return jnp.where(rand_mask, rand, buf)


def vgrad(f, x):
  """Computes gradients for a vector-valued function.
    
  >>> vgrad(lambda x: 3*x**2, jnp.ones((1,)))
  DeviceArray([6.], dtype=float32)
  """
  y, vjp_fn = jax.vjp(f, x)
  return vjp_fn(jnp.ones(y.shape))[0]


def create_noise_schedule(sigma_begin=1,
                          sigma_end=1e-2,
                          L=10,
                          schedule='geometric'):
  """Creates a noise schedule.

  Args:
    sigma_begin: Starting variance.
    sigma_end: Ending variance.
    L: Number of values in the noise schedule.
    schedule: Type of schedule.
  """
  if schedule == 'geometric':
    sigmas = jnp.exp(jnp.linspace(jnp.log(sigma_begin), jnp.log(sigma_end), L))
  elif schedule == 'linear':
    sigmas = jnp.linspace(sigma_begin, sigma_end, L)
  elif schedule == 'fibonacci':
    sigmas = [1e-6, 2e-6]
    for i in range(L - 2):
      sigmas.append(sigmas[-1] + sigmas[-2])
    sigmas = jnp.array(sigmas)
  else:
    raise ValueError(f'Unsupported schedule: {schedule}')

  return sigmas


@partial(jax.jit, static_argnums=(
    4,
    5,
    6,
    7,
))
def annealed_langevin_dynamics(rng,
                               model,
                               sigmas,
                               init,
                               epsilon,
                               T,
                               denoise,
                               infill=False,
                               infill_samples=None,
                               infill_masks=None,
                               K=1):
  """Annealed Langevin dynamics sampling from Song et al.
  
  Args:
    rng: Random number generator key.
    model: Score network.
    sigmas: Noise schedule.
    init: Initial state for Langevin dynamics (usually uniform noise).
    epsilon: Step size coefficient.
    T: Number of steps per noise level.
    denoise: Apply an additional denoising step to final samples.
    infill: Infill partially complete samples.
    infill_samples: Partially complete samples to infill.
    infill_masks: Binary mask for infilling partially complete samples.
        A zero indicates an element that must be infilled by Langevin dynamics.
  
  Returns:
    state: Final state sampled from Langevin dynamics.
    collection: Array of state at each step of sampling with shape 
        (num_sigmas * T + 1 + int(denoise), :).
    ld_metrics: Metrics collected for each noise level with shape (num_sigmas, T).
  """
  if not infill:
    infill_samples = jnp.zeros(init.shape)
    infill_masks = jnp.zeros(init.shape)

  collection_steps = 100
  start = init * (1 - infill_masks) + infill_samples * infill_masks
  images = np.zeros((collection_steps + 1 + int(denoise), *init.shape))
  images = jax.ops.index_update(images, jax.ops.index[0, :], start)
  collection_idx = jnp.linspace(1,
                                len(sigmas) * T,
                                collection_steps).astype(jnp.int32)

  def langevin_step(params, i):
    state, rng, sigma_i, alpha, collection = params
    rng, step_rng, infill_rng = jax.random.split(rng, num=3)
    sigma = sigmas[sigma_i]

    y = infill_samples + sigma * jax.random.normal(key=infill_rng,
                                                   shape=infill_samples.shape)

    grad = model(state, sigma)
    noise = jnp.sqrt(2 * alpha) * jax.random.normal(key=step_rng,
                                                    shape=state.shape)
    next_state = state + alpha * grad + noise  # gradient ascent

    # Apply infilling mask
    next_state = next_state * (1 - infill_masks) + y * infill_masks

    # Collect samples
    image_idx = sigma_i * T + i + 1
    idx_mask = jnp.in1d(collection_idx, image_idx)
    idx = jnp.sum(jnp.arange(len(collection_idx)) * idx_mask) + 1
    collection = jax.lax.cond(idx_mask.any(),
                              lambda op: jax.ops.index_update(
                                  collection, jax.ops.index[op, :], next_state),
                              lambda op: collection,
                              operand=idx)

    # Collect metrics
    grad_norm = jnp.sqrt(jnp.sum(jnp.square(grad), axis=1) + 1e-10).mean()
    noise_norm = jnp.sqrt(jnp.sum(jnp.square(noise), axis=1) + 1e-10).mean()
    step_norm = jnp.sqrt(jnp.sum(jnp.square(alpha * grad), axis=1) +
                         1e-10).mean()
    metrics = grad_norm, step_norm, alpha, noise_norm

    next_params = (next_state, rng, sigma_i, alpha, collection)
    return next_params, metrics

  def sample_with_sigma(params, sigma_i):
    state, rng, collection = params
    sigma = sigmas[sigma_i]
    alpha = epsilon * (sigma / sigmas[-1])**2

    ld_params = (state, rng, sigma_i, alpha, collection)
    next_ld_state, metrics = jax.lax.scan(langevin_step, ld_params,
                                          jnp.arange(T))
    next_state, rng, sigma_i, alpha, collection = next_ld_state

    next_params = (next_state, rng, collection)
    return next_params, metrics

  assert len(sigmas) >= 2
  init_params = (init, rng, images)
  ld_state, ld_metrics = jax.lax.scan(sample_with_sigma, init_params,
                                      jnp.arange(len(sigmas)))
  state, rng, collection = ld_state

  # Additional denoising step.
  if denoise:
    state = state + sigmas[-1]**2 * model(state, sigmas[-1])
    collection = jax.ops.index_update(collection, jax.ops.index[-1, :], state)

  return state, collection, jnp.stack(ld_metrics)


@partial(jax.jit, static_argnums=(
    4,
    5,
    6,
    7,
))
def consistent_langevin_dynamics(rng,
                                 model,
                                 sigmas,
                                 init,
                                 epsilon,
                                 T,
                                 denoise=True,
                                 infill=False,
                                 infill_samples=None,
                                 infill_masks=None,
                                 K=1):
  """Consistent annealed Langevin dynamics sampling from Jolicoeur-Martineau et al.
  
  Args:
    rng: Random number generator key.
    model: Score network.
    sigmas: Noise schedule.
    init: Initial state.
    epsilon: Step size coefficient.
    T: Number of steps per noise level.
  
  Returns:
    state: Final state sampled from Langevin dynamics.
    ld_metrics: Metrics collected for each noise level with shape (num_sigmas, T).
  """
  if infill:
    raise NotImplementedError

  def langevin_step(params, i):
    state, rng = params
    rng, step_rng = jax.random.split(rng)

    sigma = sigmas[i]
    next_sigma = jnp.where(i < len(sigmas) - 1, sigmas[i + 1], 0.)

    alpha = epsilon * (sigma / sigmas[-1])**2
    grad = model(state, sigma)
    noise = beta * next_sigma * jax.random.normal(key=step_rng,
                                                  shape=state.shape)
    next_state = state + alpha * grad + noise

    # Collect metrics
    grad_norm = jnp.sqrt(jnp.sum(jnp.square(grad), axis=1) + 1e-10).mean()
    noise_norm = jnp.sqrt(jnp.sum(jnp.square(noise), axis=1) + 1e-10).mean()
    step_norm = jnp.sqrt(jnp.sum(jnp.square(alpha * grad), axis=1) +
                         1e-10).mean()
    metrics = grad_norm, step_norm, alpha, noise_norm

    next_params = (next_state, rng)
    return next_params, metrics

  assert len(sigmas) >= 2
  gamma = sigmas[0] / sigmas[1]
  beta = jnp.sqrt(1 - (1 - epsilon / (sigmas[-1]**2))**2)

  init_params = (init, rng)
  ld_state, ld_metrics = jax.lax.scan(langevin_step, init_params,
                                      jnp.arange(len(sigmas)))
  state, rng = ld_state

  if denoise:
    state = state + sigmas[-1]**2 * model(state, sigmas[-1])

  ld_metrics = jnp.stack(ld_metrics)
  ld_metrics = jnp.expand_dims(ld_metrics, axis=2)
  return state, ld_metrics


@partial(jax.jit, static_argnums=(
    4,
    5,
    6,
    7,
))
def diffusion_dynamics(rng,
                       model,
                       betas,
                       init,
                       epsilon,
                       T,
                       denoise,
                       infill=False,
                       infill_samples=None,
                       infill_masks=None,
                       K=1):
  """Diffusion dynamics (reverse process decoder).
  
  Args:
    rng: Random number generator key.
    model: Diffusion probabilistic network.
    betas: Noise schedule.
    init: Initial state for Langevin dynamics (usually Gaussian noise).
    epsilon: Null parameter.
    T: Null parameter.
    denoise: Null parameter used in other methods to find EDS.
    infill: Infill partially complete samples.
    infill_samples: Partially complete samples to infill.
    infill_masks: Binary mask for infilling partially complete samples.
        A zero indicates an element that must be infilled by Langevin dynamics.
  
  Returns:
    state: Final state sampled from Langevin dynamics.
    collection: Array of state at each step of sampling with shape 
        (num_sigmas * T + 1 + int(denoise), :).
    ld_metrics: Metrics collected for each noise level with shape (num_sigmas, T).
  """
  if not infill:
    infill_samples = jnp.zeros(init.shape)
    infill_masks = jnp.zeros(init.shape)

  alphas = 1 - betas
  alphas_prod = jnp.cumprod(alphas)
  alphas_prod_prev = jnp.concatenate([jnp.ones((1,)), alphas_prod[:-1]])
  assert alphas.shape == alphas_prod.shape == alphas_prod_prev.shape

  collection_steps = 40
  start = init * (1 - infill_masks) + infill_samples * infill_masks
  images = np.zeros((collection_steps + 1, *init.shape))
  collection = jax.ops.index_update(images, jax.ops.index[0, :], start)
  collection_idx = jnp.linspace(1, len(betas),
                                collection_steps).astype(jnp.int32)

  def sample_with_beta(params, t):
    state, rng, collection = params
    rng, key = jax.random.split(rng)

    # Noise schedule constants
    beta = betas[t]
    alpha = alphas[t]
    alpha_prod = alphas_prod[t]
    alpha_prod_prev = alphas_prod_prev[t]

    # Constants for posterior q(x_t|x_0)
    sqrt_reciprocal_alpha_prod = jnp.sqrt(1 / alpha_prod)
    sqrt_alpha_prod_m1 = jnp.sqrt(1 - alpha_prod) * sqrt_reciprocal_alpha_prod

    # Create infilling template
    rng, infill_noise_rng = jax.random.split(rng)
    infill_noise_cond = t > 0
    infill_noise = jax.random.normal(key=infill_noise_rng,
                                     shape=infill_samples.shape)
    noisy_y = jnp.sqrt(alpha_prod) * infill_samples + jnp.sqrt(
        1 - alpha_prod) * infill_noise
    y = infill_noise_cond * noisy_y + (1 - infill_noise_cond) * infill_samples

    # Constants for posterior q(x_t-1|x_t, x_0)
    posterior_mu1 = beta * jnp.sqrt(alpha_prod_prev) / (1 - alpha_prod)
    posterior_mu2 = (1 - alpha_prod_prev) * jnp.sqrt(alpha) / (1 - alpha_prod)

    # Clipped variance (must be non-zero)
    posterior_var = beta * (1 - alpha_prod_prev) / (1 - alpha_prod)
    posterior_var_clipped = jnp.maximum(posterior_var, 1e-20)
    posterior_log_var = jnp.log(posterior_var_clipped)

    # Noise
    rng, noise_rng = jax.random.split(rng)
    noise_cond = t > 0
    noise = jax.random.normal(key=noise_rng, shape=state.shape)
    noise = noise_cond * noise + (1 - noise_cond) * jnp.zeros(state.shape)
    noise = noise * jnp.exp(0.5 * posterior_log_var)

    # Reverse process (reconstruction)
    noise_condition_vec = jnp.sqrt(alpha_prod) * jnp.ones((noise.shape[0], 1))
    noise_condition_vec = noise_condition_vec.reshape(
        init.shape[0], *([1] * len(init.shape[1:])))
    eps_recon = model(state, noise_condition_vec)
    state_recon = sqrt_reciprocal_alpha_prod * state - sqrt_alpha_prod_m1 * eps_recon
    state_recon = jnp.clip(state_recon, -1., 1.)
    posterior_mu = posterior_mu1 * state_recon + posterior_mu2 * state
    next_state = posterior_mu + noise

    # Infill
    next_state = next_state * (1 - infill_masks) + y * infill_masks

    # Collect metrics
    step = state - next_state
    grad_norm = jnp.sqrt(jnp.sum(jnp.square(eps_recon), axis=1) + 1e-10).mean()
    noise_norm = jnp.sqrt(jnp.sum(jnp.square(noise), axis=1) + 1e-10).mean()
    step_norm = jnp.sqrt(jnp.sum(jnp.square(step), axis=1) + 1e-10).mean()
    metrics = (grad_norm, step_norm, alpha_prod, noise_norm)

    # Collect samples
    image_idx = len(betas) - t + 1
    idx_mask = jnp.in1d(collection_idx, image_idx)
    idx = jnp.sum(jnp.arange(len(collection_idx)) * idx_mask) + 1
    collection = jax.lax.cond(idx_mask.any(),
                              lambda op: jax.ops.index_update(
                                  collection, jax.ops.index[op, :], next_state),
                              lambda op: collection,
                              operand=idx)

    next_params = (next_state, rng, collection)
    return next_params, metrics

  init_params = (init, rng, collection)
  beta_steps = jnp.arange(len(betas) - 1, -1, -1)
  ld_state, ld_metrics = jax.lax.scan(sample_with_beta, init_params, beta_steps)
  state, rng, collection = ld_state
  ld_metrics = jnp.stack(ld_metrics)
  ld_metrics = jnp.expand_dims(ld_metrics, 2)
  return state, collection, ld_metrics

def diffusion_dynamics_sdedit(rng,
                       model,
                       betas,
                       init,
                       epsilon,
                       T,
                       denoise,
                       infill=False,
                       infill_samples=None,
                       infill_masks=None,
                       K = 1):
  """Diffusion dynamics (reverse process decoder).
  
  Args:
    rng: Random number generator key.
    model: Diffusion probabilistic network.
    betas: Noise schedule.
    init: Initial state for Langevin dynamics (usually Gaussian noise).
    epsilon: Null parameter.
    T: Null parameter.
    denoise: Null parameter used in other methods to find EDS.
    infill: Infill partially complete samples.
    infill_samples: Partially complete samples to infill.
    infill_masks: Binary mask for infilling partially complete samples.
        A zero indicates an element that must be infilled by Langevin dynamics.
  
  Returns:
    state: Final state sampled from Langevin dynamics.
    collection: Array of state at each step of sampling with shape 
        (num_sigmas * T + 1 + int(denoise), :).
    ld_metrics: Metrics collected for each noise level with shape (num_sigmas, T).
  """

  if not infill:
    infill_samples = jnp.zeros(init.shape)
    infill_masks = jnp.zeros(init.shape)

  alphas = 1 - betas
  alphas_prod = jnp.cumprod(alphas)
  alphas_prod_prev = jnp.concatenate([jnp.ones((1,)), alphas_prod[:-1]])
  assert alphas.shape == alphas_prod.shape == alphas_prod_prev.shape

  collection_steps = 40

  x0 = infill_samples
  x = infill_samples
  Omega = 1 - infill_masks

  def sample_with_beta(params, t):
    x, rng, collection = params
    rng, key = jax.random.split(rng)

    # Noise schedule constants
    beta = betas[t]
    alpha = alphas[t]
    alpha_prod = alphas_prod[t]
    alpha_prod_prev = alphas_prod_prev[t]

    # Constants for posterior q(x_t|x_0)
    sqrt_reciprocal_alpha_prod = jnp.sqrt(1 / alpha_prod)
    sqrt_alpha_prod_m1 = jnp.sqrt(1 - alpha_prod) * sqrt_reciprocal_alpha_prod

    # Create infilling template
    rng, infill_noise_rng = jax.random.split(rng)
    infill_noise_cond = t > 0
    z1 = jax.random.normal(key=infill_noise_rng, shape=x0.shape)

    y = jnp.sqrt(alpha_prod) * x0 + jnp.sqrt(1 - alpha_prod) * z1
    #y = infill_noise_cond * noisy_y + (1 - infill_noise_cond) * x0

    # Constants for posterior q(x_t-1|x_t, x_0)
    posterior_mu1 = beta * jnp.sqrt(alpha_prod_prev) / (1 - alpha_prod)
    posterior_mu2 = (1 - alpha_prod_prev) * jnp.sqrt(alpha) / (1 - alpha_prod)

    # Clipped variance (must be non-zero)
    posterior_var = beta * (1 - alpha_prod_prev) / (1 - alpha_prod)
    posterior_var_clipped = jnp.maximum(posterior_var, 1e-20)
    posterior_log_var = jnp.log(posterior_var_clipped)

    # Noise (can this be the same as the noise generated for infill?)
    rng, noise_rng = jax.random.split(rng)
    noise_cond = t > 0
    #z2 = jax.random.normal(key=noise_rng, shape=x.shape)
    #z2 = noise_cond * z2 + (1 - noise_cond) * jnp.zeros(x.shape)
    epsilonz = jnp.exp(0.5 * posterior_log_var) * z1

    # Reverse process (reconstruction)
    noise_condition_vec = jnp.sqrt(alpha_prod) * jnp.ones((epsilonz.shape[0], 1))
    noise_condition_vec = noise_condition_vec.reshape(
        init.shape[0], *([1] * len(init.shape[1:])))

    eps_recon = model(x, noise_condition_vec)

    x_recon = sqrt_reciprocal_alpha_prod * x - sqrt_alpha_prod_m1 * eps_recon
    x_recon = jnp.clip(x_recon, -1., 1.)

    # Infill
    x_next = (1 - Omega) * y  + Omega * (posterior_mu1 * x_recon + posterior_mu2 * x + epsilonz)

    # Collect metrics
    step = x - x_next
    grad_norm = jnp.sqrt(jnp.sum(jnp.square(eps_recon), axis=1) + 1e-10).mean()
    noise_norm = jnp.sqrt(jnp.sum(jnp.square(z1), axis=1) + 1e-10).mean()
    step_norm = jnp.sqrt(jnp.sum(jnp.square(step), axis=1) + 1e-10).mean()
    metrics = (grad_norm, step_norm, alpha_prod, noise_norm)

    # Collect samples
    image_idx = len(betas) - t + 1
    idx_mask = jnp.in1d(collection_idx, image_idx)
    idx = jnp.sum(jnp.arange(len(collection_idx)) * idx_mask) + 1
    collection = jax.lax.cond(idx_mask.any(),
                              lambda op: jax.ops.index_update(
                                  collection, jax.ops.index[op, :], x_next),
                              lambda op: collection,
                              operand=idx)

    next_params = (x_next, rng, collection)
    return next_params, metrics

  K = 3
  for k in range(K):
    # Assume init is sigma2(t_0)z, gaussian noise already generated with sigma^2
    rng, z_rng = jax.random.split(rng)
    z = jax.random.normal(key=z_rng, shape=x0.shape)
    x = jnp.sqrt(alphas_prod[-1]) * (Omega * x + (1 - Omega) * x0) + jnp.sqrt(1 - alphas_prod[-1]) * z # Omega is the editable region

    images = np.zeros((collection_steps + 1, *init.shape))
    collection = jax.ops.index_update(images, jax.ops.index[0, :], x)
    collection_idx = jnp.linspace(1, len(betas),
                                collection_steps).astype(jnp.int32)

    init_params = (init, rng, collection)
    beta_steps = jnp.arange(len(betas) - 1, -1, -1)
    ld_state, ld_metrics = jax.lax.scan(sample_with_beta, init_params, beta_steps)
    x, rng, collection = ld_state
    ld_metrics = jnp.stack(ld_metrics)
    ld_metrics = jnp.expand_dims(ld_metrics, 2)

  return x, collection, ld_metrics # return end values

def collate_sampling_metrics(ld_metrics):
  """Converts Langevin metrics into TensorBoard-readable format.
  
  Args:
    ld_metrics: A tensor with metrics returned by annealed_langevin_dynamics
        sampling procedure.
  """
  num_metrics, num_sigmas, num_steps = ld_metrics.shape
  del num_metrics  # unused
  sampling_metrics = [[] for i in range(num_sigmas)]
  for i in range(num_sigmas):
    grad_norm, step_norm, alpha, noise_norm = ld_metrics[:, i, :]
    for j in range(num_steps):
      metrics = {
          'slope': grad_norm[j],
          'step': step_norm[j],
          'alpha': alpha[j],
          'noise': noise_norm[j]
      }
      sampling_metrics[i].append(metrics)
  return sampling_metrics

def reverse_diffusion_sampler(rng,
  model,
  betas,
  init,
  epsilon,
  T,
  denoise,
  infill=False,
  infill_samples=None,
  infill_masks=None,
  K=1):
  """Reverse Diffusion Sampler (for VESDEs)
  
  Args:
    rng: Random number generator key.
    model: Diffusion probabilistic network.
    betas: Noise schedule.
    init: Initial state for Langevin dynamics (usually Gaussian noise).
    epsilon: Null parameter.
    T: Null parameter.
    denoise: Null parameter used in other methods to find EDS.
    infill: Infill partially complete samples.
    infill_samples: Partially complete samples to infill.
    infill_masks: Binary mask for infilling partially complete samples.
        A zero indicates an element that must be infilled by Langevin dynamics.
  
  Returns:
    state: Final state sampled from Langevin dynamics.
    collection: Array of state at each step of sampling with shape 
        (num_sigmas * T + 1 + int(denoise), :).
    ld_metrics: Metrics collected for each noise level with shape (num_sigmas, T).
  """

  def batch_mul(a, b):
    return jax.vmap(lambda a, b: a * b)(a, b)
  
  if not infill:
    infill_samples = jnp.zeros(init.shape)
    infill_masks = jnp.zeros(init.shape)

  eps = 1e-5 # default set by song
  T = 1 # default set by song
  N = 1000

  # Define timesteps for collection
  collection_steps = 40
  sigma_min = 1e-6
  sigma_max = 10

  start = init * (1 - infill_masks) + infill_samples * infill_masks
  images = np.zeros((collection_steps + 1, *init.shape))
  collection = jax.ops.index_update(images, jax.ops.index[0, :], start)
  t_vec = jnp.linspace(T, eps, N).astype(jnp.int32) # This is betas
  collection_idx = jnp.linspace(1, N, collection_steps).astype(jnp.int32)

  def sample_step(params, i):

    state, rng, collection = params

    t = jnp.tile(t_vec[i], (state.shape[0], 1, 1))

    std = sigma_min * (sigma_max / sigma_min) ** t

    # Reverse process (reconstruction)
    diffusion = std * jnp.sqrt(2 * (jnp.log(sigma_max) - jnp.log(sigma_min)))
    dt = 1 / N
    G = diffusion * jnp.sqrt(dt)

    score = model(state, std) 

    rev_f = - batch_mul(G ** 2, score)
    rev_G = G

    rng, noise_rng = jax.random.split(rng)
    z = jax.random.normal(noise_rng, state.shape)

    mean = state - rev_f
    next_state = mean + batch_mul(rev_G, z)

    # Infill
    next_state = next_state * (1 - infill_masks) + infill_samples * infill_masks

    # Collect metrics
    step = state - next_state
    grad_norm = jnp.sqrt(jnp.sum(jnp.square(score), axis=1) + 1e-10).mean()
    noise_norm = jnp.sqrt(jnp.sum(jnp.square(z), axis=1) + 1e-10).mean()
    step_norm = jnp.sqrt(jnp.sum(jnp.square(step), axis=1) + 1e-10).mean()
    alpha_prod = std[0].squeeze()
    metrics = (grad_norm, step_norm, alpha_prod, noise_norm)

    # Collect samples
    image_idx = N - i + 1
    idx_mask = jnp.in1d(collection_idx, image_idx)
    idx = jnp.sum(jnp.arange(len(collection_idx)) * idx_mask) + 1
    collection = jax.lax.cond(idx_mask.any(),
                              lambda op: jax.ops.index_update(
                                  collection, jax.ops.index[op, :], next_state),
                              lambda op: collection,
                              operand=idx)

    next_params = (next_state, rng, collection)
    return next_params, metrics

  init_params = (init, rng, collection)
  t_steps = jnp.arange(N - 1, -1, -1)
  ld_state, ld_metrics = jax.lax.scan(sample_step, init_params, t_steps)
  state, rng, collection = ld_state
  ld_metrics = jnp.stack(ld_metrics)
  ld_metrics = jnp.expand_dims(ld_metrics, 2)
  return state, collection, ld_metrics
