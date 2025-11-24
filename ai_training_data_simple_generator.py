import numpy as np
import jax
import jax.numpy as jnp
import h5py

from scattering_simulation import prepare_lens_scattering_solver

from time import time

jax.config.update('jax_enable_x64', True)


if __name__ == '__main__':
    # print('Enter start index: ', end='')
    # start_i = int(input())
    # print('Enter end index: ', end='')
    # end_i = int(input())

    func, expansion = prepare_lens_scattering_solver(
        wavelength=450,
        period=2000,
        lens_thickness=600,
        substrate_thickness=500,
        approximate_number_of_terms=600
    )
    sim_f = jax.jit(func)

    # patterns = np.load('wave_maps_4224.npz')['x'][start_i:end_i]
    patterns = np.load('wave_maps_4224.npz')['x']
    print(patterns.shape, len(expansion))
    patterns = jnp.array(patterns).astype(float)
    amps = np.zeros((patterns.shape[0], len(expansion)), dtype=complex)

    for i, pattern in enumerate(patterns):
        start = time()
        amps[i] = sim_f(pattern)
        print(i, time() - start)

    # with h5py.File(f'ai_training_data/temp_batches/green_redmaps_{start_i}_{end_i}.hdf5', 'a') as f:
    with h5py.File(f'ai_training_data/temp_batches/blue_redmaps.hdf5', 'a') as f:
    #     f.create_dataset('patterns', data=patterns)
        f.create_dataset('field_amps', data=amps)
