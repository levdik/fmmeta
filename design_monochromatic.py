import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import topology_parametrization
from scattering_simulation import prepare_lens_scattering_solver
from field_postprocessing import calculate_focusing_efficiency
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

    grid_size = 10

    topology = lens_topology_parametrization.FourierInterpolation(
        grid_size=grid_size, symmetry_type='central')
    simulate_lens_scattering, propagating_basis_indices = prepare_lens_scattering_solver(
        wavelength=wavelength,
        period=period,
        lens_thickness=lens_thickness,
        substrate_thickness=substrate_thickness,
        approximate_number_of_terms=300,
        propagate_by_distance=detector_offset
    )


    def calculate_lens_efficiency(geometrical_parameters):
        lens_topology = topology(geometrical_parameters)
        focal_plane_amplitudes = simulate_lens_scattering(lens_topology)
        focusing_efficiency = calculate_focusing_efficiency(focal_plane_amplitudes, propagating_basis_indices)
        transmission_efficiency = jnp.sum(jnp.abs(focal_plane_amplitudes) ** 2)
        return focusing_efficiency * transmission_efficiency

    init_geometrical_parameters = jax.random.uniform(
        jax.random.key(33), (topology.n_geometrical_parameters,), minval=-0.5, maxval=0.5)

    init_pattern = topology(init_geometrical_parameters)
    plt.imshow(init_pattern)
    plt.axis('off')
    plt.show()

    optimized_geometrical_parameters, max_eff = run_gradient_ascent(
        target_function=calculate_lens_efficiency,
        x_init=init_geometrical_parameters,
        learning_rate=1e-2,
        n_steps=100,
        boundary_projection_function=lambda x: jnp.clip(x, -1, 1),
        except_keyboard_interrupt=True
    )

    print('Max eff:', max_eff)
    print(optimized_geometrical_parameters)

    plt.imshow(topology(optimized_geometrical_parameters))
    plt.axis('off')
    plt.show()
