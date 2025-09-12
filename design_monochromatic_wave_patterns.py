import jax
import jax.numpy as jnp
import numpy as np
import optax

from fmmax import basis, fmm, fields, scattering
from field_postprocessing import calculate_focusing_efficiency
import wave_pattern_factory

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


def optimize_wavy_monochromatic_lens(
        wavelength, permittivity, period, thickness, focal_length, approx_n_terms, wave_n_max
):
    (
        permittivity_map_to_scattered_amps_func,
        propagating_basis_indices
    ) = wave_pattern_factory.prepare_wave_lens_scattering_simulating_function(
        wavelength=wavelength,
        thickness=thickness,
        period=period,
        approx_n_terms=approx_n_terms,
        propagate_by_distance=focal_length
    )

    (
        full_basis_indices, primary_basis_indices, symmetry_indices
    ) = wave_pattern_factory.generate_wave_permittivity_primary_basis_indices(
        r=wave_n_max, symmetry_type='central'
    )
    n_primary_params = len(primary_basis_indices)
    # init_params = jnp.zeros(shape=(2, n_primary_params))
    init_params = 0.1 * jax.random.uniform(jax.random.key(42), shape=(2, n_primary_params))

    def project_onto_boundaries(x):
        return jnp.clip(x, -1., 1.)

    def params_to_focusing_efficiency(params):
        primary_pattern_amps = params[0] + 1j * params[1]
        pattern_amps = primary_pattern_amps[symmetry_indices]
        permittivity_pattern = wave_pattern_factory.generate_wave_permittivity_pattern(
            amplitudes=pattern_amps,
            basis_indices=full_basis_indices,
            permittivity=permittivity
        )
        focal_plane_scattered_amps = permittivity_map_to_scattered_amps_func(permittivity_pattern)
        focusing_efficiency = calculate_focusing_efficiency(
            focal_plane_scattered_amps, propagating_basis_indices
        )
        transmission_efficiency = jnp.sum(jnp.abs(focal_plane_scattered_amps) ** 2)
        total_efficiency = transmission_efficiency * focusing_efficiency
        return total_efficiency

    def loss_fn(params):
        return -params_to_focusing_efficiency(params)

    loss_value_and_grad_fn = jax.value_and_grad(loss_fn)

    optimizer = optax.adam(1e-2, b1=0.9)
    opt_state = optimizer.init(init_params)

    # @jax.jit
    def step(x, opt_state):
        loss, grad = loss_value_and_grad_fn(x)
        updates, opt_state = optimizer.update(grad, opt_state)
        x = optax.apply_updates(x, updates)
        x = project_onto_boundaries(x)
        return x, opt_state, loss, grad

    x = init_params
    max_eff = 0
    best_x = None

    try:
        for i in range(300):
            new_x, opt_state, loss, grad = step(x, opt_state)
            print(i + 1, -loss, sep='\t')
            x = new_x
            if -loss > max_eff:
                max_eff = -loss
                best_x = x
    except KeyboardInterrupt:
        pass

    print('Max eff:', max_eff)

    primary_pattern_amps = best_x[0] + 1j * best_x[1]
    pattern_amps = primary_pattern_amps[symmetry_indices]
    permittivity_pattern = wave_pattern_factory.generate_wave_permittivity_pattern(
        amplitudes=pattern_amps,
        basis_indices=full_basis_indices,
        permittivity=permittivity
    )
    plt.imshow(permittivity_pattern)
    plt.axis('off')
    plt.show()

    print(repr(primary_pattern_amps))


if __name__ == '__main__':
    jnp.set_printoptions(linewidth=1000)

    optimize_wavy_monochromatic_lens(
        wavelength=650,
        permittivity=4,
        period=2000,
        thickness=500,
        focal_length=4000,
        approx_n_terms=300,
        wave_n_max=8
    )
