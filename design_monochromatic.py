import jax
import jax.numpy as jnp
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
    wavelength = 450
    period = 2000
    lens_thickness = 600
    substrate_thickness = 500
    detector_offset = 3500

    min_feature_width = 100
    n_px = 100
    min_width_px = round(min_feature_width * n_px / period)
    min_width_px = (min_width_px // 2) * 2 + 1

    # topology = topology_parametrization.FourierInterpolation(10, symmetry_type='central')
    # topology = topology_parametrization.FourierExpansion(7, 'main_diagonal')
    topology = topology_parametrization.GaussianField(n_px, sigma=32)
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
        return focusing_efficiency * transmission_efficiency - 10 * invalid_area
        # return focusing_efficiency * transmission_efficiency

    init_geometrical_parameters = jax.random.uniform(
        jax.random.key(1), (topology.n_geometrical_parameters,), minval=-0.5, maxval=0.5)

    init_pattern = topology(init_geometrical_parameters, n_px)
    plt.imshow(init_pattern, vmin=0, vmax=1)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    optimized_geometrical_parameters, max_eff = run_gradient_ascent(
        target_function=lambda x: calculate_lens_efficiency(x, 1),
        x_init=init_geometrical_parameters,
        learning_rate=1e-2,
        n_steps=200,
        boundary_projection_function=lambda x: jnp.clip(x, -1, 1),
        except_keyboard_interrupt=True
    )

    print('Max eff:', max_eff)
    print('Invalid area:', too_thin_area(topology(optimized_geometrical_parameters, n_px), min_width_px))
    print(optimized_geometrical_parameters)

    plt.imshow(topology(optimized_geometrical_parameters, n_px, bin_strength=1), vmin=0, vmax=1)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
