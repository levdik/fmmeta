import numpy as np
import jax.numpy as jnp
import jax

from scattering_simulation import prepare_lens_scattering_solver

from time import time

jax.config.update('jax_enable_x64', True)


if __name__ == '__main__':
    patterns = np.load('freeform_training_patterns.npy')
    print(patterns.shape)

    func, expansion = prepare_lens_scattering_solver(
        wavelength=650,
        period=2000,
        lens_thickness=600,
        substrate_thickness=500,
        approximate_number_of_terms=300,
        include_reflection=True
    )
    sim_f = jax.jit(func)

    permutation = jax.random.permutation(jax.random.key(0), patterns.shape[0])
    amps = []

    try:
        for i in range(patterns.shape[0]):
            start = time()
            amps_i = sim_f(patterns[permutation[i]])
            amps_i.block_until_ready()
            amps.append(amps_i)
            print(i, time() - start)
    except KeyboardInterrupt:
        pass

    n_completed = len(amps)
    print('Completed:', n_completed)

    amps = np.array(amps)
    np.savez('ai_training_data/red_freeform.npz', patterns=patterns[permutation[:n_completed]], amps=amps)
