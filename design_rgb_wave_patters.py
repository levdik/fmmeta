import jax
import jax.numpy as jnp
import optax

from field_postprocessing import calculate_focusing_efficiency
import wave_pattern_factory as wf
from field_postprocessing import intensity_map_from_fourier_amplitudes

from field_plotter import plot_amplitude_map

import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')


wavelengths = [450, 550, 650]
relative_focal_points = ((0.75, 0.75), ((0.25, 0.75), (0.75, 0.25)), (0.25, 0.25))


def optimize_wavy_rgb_lens(
        permittivity, period, thickness, focal_length, approx_n_terms, wave_n_max,
        init_params=None, n_steps=100, learning_rate=0.01
):
    common_func_prep_kwargs = {
        'thickness': thickness,
        'period': period,
        'approx_n_terms': approx_n_terms,
        'propagate_by_distance': focal_length,
    }
    red_sim_func, red_basis_indices = wf.prepare_wave_lens_scattering_simulating_function(
        wavelength=wavelengths[2], **common_func_prep_kwargs)
    # green_sim_func, green_basis_indices = wf.prepare_wave_lens_scattering_simulating_function(
    #     wavelength=wavelengths[1], **common_func_prep_kwargs)
    blue_sim_func, blue_basis_indices = wf.prepare_wave_lens_scattering_simulating_function(
        wavelength=wavelengths[0], **common_func_prep_kwargs)

    def red_focusing_efficiency_function(permittivity_map):
        focal_plane_amps = red_sim_func(permittivity_map)
        focusing_efficiency = calculate_focusing_efficiency(
            focal_plane_amps, red_basis_indices, relative_focal_points[2])
        transmission_efficiency = jnp.sum(jnp.abs(focal_plane_amps) ** 2)
        total_efficiency = transmission_efficiency * focusing_efficiency
        return total_efficiency

    # def green_focusing_efficiency_function(permittivity_map):
    #     focal_plane_amps = green_sim_func(permittivity_map)
    #     focusing_efficiency = (
    #             calculate_focusing_efficiency(
    #                 focal_plane_amps, green_basis_indices, relative_focal_points[1][0])
    #             + calculate_focusing_efficiency(
    #         focal_plane_amps, green_basis_indices, relative_focal_points[1][1])
    #     )
    #     transmission_efficiency = jnp.sum(jnp.abs(focal_plane_amps) ** 2)
    #     total_efficiency = transmission_efficiency * focusing_efficiency
    #     return total_efficiency

    def blue_focusing_efficiency_function(permittivity_map):
        focal_plane_amps = blue_sim_func(permittivity_map)
        focusing_efficiency = calculate_focusing_efficiency(
            focal_plane_amps, blue_basis_indices, relative_focal_points[0])
        transmission_efficiency = jnp.sum(jnp.abs(focal_plane_amps) ** 2)
        total_efficiency = transmission_efficiency * focusing_efficiency
        return total_efficiency

    (
        full_basis_indices, primary_basis_indices, symmetry_indices
    ) = wf.generate_wave_permittivity_primary_basis_indices(
        r=wave_n_max, symmetry_type='main_diagonal'
    )
    n_primary_params = len(primary_basis_indices)
    if init_params is None:
        # init_params_re_im = 0.1 * jax.random.uniform(jax.random.key(42), shape=(2, n_primary_params), minval=-1, maxval=1)
        # init_params = init_params_re_im[0] + 1j * init_params_re_im[1]
        init_params = 0.1 * jax.random.uniform(jax.random.key(42), shape=(2, n_primary_params), minval=-1, maxval=1)

    # def project_onto_boundaries(x):
    #     return jnp.where(jnp.abs(x) > 1, x / jnp.abs(x), x)

    def params_to_focusing_efficiency(params):
        pattern_amps = params[0][symmetry_indices] + 1j * params[1][symmetry_indices]
        permittivity_pattern = wf.generate_wave_permittivity_pattern(
            amplitudes=pattern_amps,
            basis_indices=full_basis_indices,
            permittivity=permittivity
        )
        # overall_efficiency = (
        #     red_focusing_efficiency_function(permittivity_pattern)
        #     + green_focusing_efficiency_function(permittivity_pattern)
        #     + blue_focusing_efficiency_function(permittivity_pattern)
        # ) / 3
        overall_efficiency = (
            red_focusing_efficiency_function(permittivity_pattern)
            + blue_focusing_efficiency_function(permittivity_pattern)
        ) / 2
        return overall_efficiency

    def loss_fn(params):
        return -params_to_focusing_efficiency(params)

    loss_value_and_grad_fn = jax.value_and_grad(loss_fn)

    optimizer = optax.adam(learning_rate, b1=0.9)
    opt_state = optimizer.init(init_params)

    @jax.jit
    def step(x, opt_state):
        loss, grad = loss_value_and_grad_fn(x)
        updates, opt_state = optimizer.update(grad, opt_state)
        x = optax.apply_updates(x, updates)
        # x = project_onto_boundaries(x)
        return x, opt_state, loss, grad

    x = init_params
    max_eff = 0
    best_x = None

    try:
        for i in range(n_steps):
            new_x, opt_state, loss, grad = step(x, opt_state)
            print(i + 1, -loss, sep='\t')
            x = new_x
            if -loss > max_eff:
                max_eff = -loss
                best_x = x
    except KeyboardInterrupt:
        pass

    print('Max eff:', max_eff)

    pattern_amps = best_x[0][symmetry_indices] + 1j * best_x[1][symmetry_indices]
    permittivity_pattern = wf.generate_wave_permittivity_pattern(
        amplitudes=pattern_amps,
        basis_indices=full_basis_indices,
        permittivity=permittivity
    )

    red_focal_plane_amps = red_sim_func(permittivity_pattern)
    # green_focal_plane_amps = green_sim_func(permittivity_pattern)
    blue_focal_plane_amps = blue_sim_func(permittivity_pattern)
    # focal_plane_amps = [red_focal_plane_amps, green_focal_plane_amps, blue_focal_plane_amps]
    # basis_indices = [red_basis_indices, green_basis_indices, blue_basis_indices]
    focal_plane_amps = [red_focal_plane_amps, blue_focal_plane_amps]
    basis_indices = [red_basis_indices, blue_basis_indices]

    # fig, ax = plt.subplots(2, 2)
    fig, ax = plt.subplots(1, 3, figsize=(30, 10))
    plt.tight_layout()
    ax = ax.flatten()

    ax[0].imshow(permittivity_pattern, extent=(0, period, 0, period))
    ax[0].set_axis_off()

    for i in range(len(focal_plane_amps)):
        intensity = intensity_map_from_fourier_amplitudes(
            amplitudes=focal_plane_amps[i],
            basis_indices=basis_indices[i]
        )

        plot_amplitude_map(
            fig, ax[i + 1], intensity,
            # wavelength_nm=wavelengths[2 - i],
            wavelength_nm=[wavelengths[2], wavelengths[0]][i],
            map_bounds=[0, period],
            cbar=False
        )
        ax[i + 1].set_axis_off()

    plt.show()

    return best_x


if __name__ == '__main__':
    jnp.set_printoptions(linewidth=1000)

    params = optimize_wavy_rgb_lens(
        permittivity=4,
        period=2000,
        thickness=600,
        focal_length=4000,
        approx_n_terms=300,
        wave_n_max=8,
        n_steps=500,
        learning_rate=0.01
    )

    print(repr(params))

    # params = None
    # old_basis = None
    # new_basis = None
    # print('Start with random initialization')
    #
    # for r_max in [3, 4, 5, 6, 7, 8]:
    #     print('Start with r_max:', r_max)
    #     old_basis = new_basis
    #     _, new_basis, _ = wf.generate_wave_permittivity_primary_basis_indices(r_max, symmetry_type='main_diagonal')
    #
    #     if params is not None:
    #         old_amps = params[0] + 1j * params[1]
    #         new_amps = wf.copy_amps_to_new_basis(old_amps, old_basis, new_basis)
    #         params = jnp.stack([new_amps.real, new_amps.imag])
    #
    #     params = optimize_wavy_rgb_lens(
    #         permittivity=4,
    #         period=2000,
    #         thickness=600,
    #         focal_length=4000,
    #         approx_n_terms=300,
    #         wave_n_max=r_max,
    #         init_params=params,
    #         n_steps=50,
    #         learning_rate=0.01
    #     )
    #
    #     print(repr(params))
