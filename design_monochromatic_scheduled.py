import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt

import topology_parametrization
from scattering_simulation import prepare_lens_scattering_solver
from field_postprocessing import calculate_focusing_efficiency
from manufacturing_constraints import too_thin_area
from design_optimizer import run_gradient_ascent

jax.config.update('jax_enable_x64', True)
jnp.set_printoptions(precision=15)
jnp.set_printoptions(linewidth=10000)

if __name__ == '__main__':
    wavelength = 650
    period = 2000
    lens_thickness = 600
    substrate_thickness = 500
    detector_offset = 3500

    min_feature_width = 100
    n_px = 100
    min_width_px = round(min_feature_width * n_px / period)
    min_width_px = (min_width_px // 2) * 2 + 1

    topology = topology_parametrization.GaussianField(n_px, sigma=16, symmetry_type='central')
    simulate_lens_scattering, propagating_basis_indices = prepare_lens_scattering_solver(
        wavelength=wavelength,
        period=period,
        lens_thickness=lens_thickness,
        substrate_thickness=substrate_thickness,
        approximate_number_of_terms=300,
        propagate_by_distance=detector_offset
    )


    def calculate_lens_efficiency(geometrical_parameters, bin_strength):
        pattern = topology(geometrical_parameters, n_px, bin_strength=bin_strength)
        pattern_bin = topology(geometrical_parameters, n_px, bin_strength=1)
        focal_plane_amplitudes = simulate_lens_scattering(pattern)
        focusing_efficiency = calculate_focusing_efficiency(focal_plane_amplitudes, propagating_basis_indices)
        transmission_efficiency = jnp.sum(jnp.abs(focal_plane_amplitudes) ** 2)
        invalid_area = too_thin_area(pattern_bin, min_width_px)
        return focusing_efficiency * transmission_efficiency - 4 * invalid_area


    init_geometrical_parameters = jax.random.uniform(
        jax.random.key(8), (topology.n_geometrical_parameters,), minval=-1, maxval=1)
    # init_geometrical_parameters = jnp.zeros(topology.n_geometrical_parameters)

    init_pattern = topology(init_geometrical_parameters, n_px, bin_strength=0)
    plt.imshow(init_pattern, vmin=0, vmax=1)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    loss_value_and_grad_fn = jax.value_and_grad(lambda x, s: -calculate_lens_efficiency(x, s))
    optimizer = optax.adam(learning_rate=5e-3)
    opt_state = optimizer.init(init_geometrical_parameters)


    @jax.jit
    def step(x, opt_state, progress):
        # bin_strength = jnp.astype(progress > 0.5, float)
        bin_strength = 1
        loss, grad = loss_value_and_grad_fn(x, bin_strength)
        updates, opt_state = optimizer.update(grad, opt_state)
        x = optax.apply_updates(x, updates)
        x = jnp.clip(x, -1, 1)
        return x, opt_state, loss, grad


    x = init_geometrical_parameters
    n_steps = 100
    best_x = None
    max_f = -jnp.inf

    try:
        for i in range(n_steps):
            progress = i / (n_steps - 1)
            new_x, opt_state, loss, grad = step(x, opt_state, progress)

            if -loss > max_f:
                max_f = -loss
                best_x = x

            print(i, -loss, sep='\t')
            x = new_x
    except KeyboardInterrupt:
        pass

    final_x = x
    final_f = -loss
    final_x = best_x
    final_f = max_f
    print('Final eff:', final_f)
    print('Invalid area:', too_thin_area(topology(final_x, n_px), min_width_px))

    plt.imshow(topology(final_x, n_px, bin_strength=1), vmin=0, vmax=1)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
