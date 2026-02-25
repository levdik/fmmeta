import numpy as np
import jax
import jax.numpy as jnp
import h5py

from scattering_simulation import prepare_lens_scattering_solver

from time import time

jax.config.update('jax_enable_x64', True)


if __name__ == '__main__':
    patterns = np.load('wave_maps_4224.npz')['x'][:16]
    print(patterns.shape)
    patterns = jnp.array(patterns).astype(float)

    for wl in [650, 550, 450]:
        for th in [400, 450, 500, 550, 600, 650, 700, 750, 800]:
            func, expansion = prepare_lens_scattering_solver(
                wavelength=wl,
                period=2000,
                lens_thickness=th,
                substrate_thickness=500,
                approximate_number_of_terms=600
            )
            sim_f = jax.jit(func)

            for i, pattern in enumerate(patterns):
                amps = sim_f(pattern)
                print(wl, th, i, jnp.linalg.norm(amps))
