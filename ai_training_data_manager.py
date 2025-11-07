import os

import numpy

n_cores = 8
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={n_cores}'

import numpy as np
import jax
import jax.numpy as jnp
import h5py

from scattering_simulation import prepare_lens_scattering_solver
import topology_parametrization

jax.config.update('jax_enable_x64', True)


def generate_training_topology_params(r_max, n, rng_key):
    topology = topology_parametrization.FourierExpansion(r_max=r_max, symmetry_type='main_diagonal')
    expansion_norms = np.linalg.norm(topology.primary_basis, axis=-1)
    _, unique_norm_counts = np.unique(expansion_norms, return_counts=True)
    unique_norm_end_indices = np.cumsum(unique_norm_counts) - 1

    i_key, value_key, a00_key = jax.random.split(rng_key, num=3)
    random_end_i = jax.random.randint(i_key, shape=(n,), minval=0, maxval=len(unique_norm_end_indices))
    random_end_i = unique_norm_end_indices[random_end_i]

    params = jax.random.uniform(value_key, shape=(2, n, topology.n_primary_parameters), minval=-1, maxval=1)
    params = params[0] + 1j * params[1]
    keep_mask = np.tile(np.arange(topology.n_primary_parameters), reps=(n, 1))
    keep_mask = keep_mask <= random_end_i[:, None]
    params *= keep_mask
    params = params.at[:, 0].set(np.clip(0.3 * jax.random.normal(a00_key), -1, 1))
    params = np.hstack([
        params[:, 0][:, None].astype(float),
        params[:, 1:].real,
        params[:, 1:].imag
    ])
    return params


def generate_patterns_in_paralel(params, r_max, resolution):
    topology = topology_parametrization.FourierExpansion(r_max=r_max, symmetry_type='main_diagonal')
    patterns = jax.lax.map(lambda arg: topology(arg, n_samples=resolution), params, batch_size=10000)
    return patterns


def simulate_in_parallel(patterns, batch_size, **sim_kwargs):
    n_samples = patterns.shape[0]
    arg_shape = patterns.shape[1:]
    assert n_samples % n_cores == 0
    patterns = patterns.reshape(8, n_samples // 8, *arg_shape)

    simulate_scattering_single, expansion_indices = prepare_lens_scattering_solver(**sim_kwargs)

    def simulate_scattering_map(args):
        return jax.lax.map(simulate_scattering_single, args, batch_size=batch_size)

    scattered_amplitudes = jax.pmap(simulate_scattering_map)(patterns)
    return scattered_amplitudes.reshape(n_samples, -1)


if __name__ == '__main__':
    from time import time
    start = time()

    r_max = 10
    n_samples = 32
    topology_params = generate_training_topology_params(r_max=r_max, n=n_samples, rng_key=jax.random.key(42))
    patterns = generate_patterns_in_paralel(topology_params, r_max=r_max, resolution=128)

    scattered_amplitudes = simulate_in_parallel(
        patterns=patterns,
        batch_size=10,
        wavelength=650,
        period=2000,
        lens_thickness=600,
        substrate_thickness=500,
        approximate_number_of_terms=600,
    )

    print(time() - start)

    with h5py.File(f'red.hdf5', 'a') as f:
        f.create_dataset('topology_params', data=topology_params)
        f.create_dataset('field_amps', data=scattered_amplitudes)
