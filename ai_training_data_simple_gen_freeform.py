import numpy as np
import jax
import jax.numpy as jnp
import h5py

from scattering_simulation import prepare_lens_scattering_solver
import topology_parametrization

from time import time

jax.config.update('jax_enable_x64', True)


if __name__ == '__main__':
    # print('Enter start index: ', end='')
    # start_i = int(input())
    # print('Enter end index: ', end='')
    # end_i = int(input())
    start_i = 0
    end_i = 512
    i_range = np.arange(start_i, end_i)
    n_samples = len(i_range)

    func, expansion = prepare_lens_scattering_solver(
        wavelength=650,
        period=2000,
        lens_thickness=600,
        substrate_thickness=500,
        approximate_number_of_terms=600
    )
    sim_f = jax.jit(func)

    patterns = []
    amps = []

    for i in i_range:
        start = time()

        pattern =
        amp = sim_f(pattern)
        patterns.append(pattern)
        amps.append(amp)

        print(i, time() - start)

    patterns = np.stack(patterns)
    amps = np.stack(amps)

    with h5py.File(f'ai_training_data/temp_batches/red_freeform_{start_i}_{end_i}.hdf5', 'a') as f:
        f.create_dataset('patterns', data=patterns)
        f.create_dataset('field_amps', data=amps)
