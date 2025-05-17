import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import optax

from fmmax import basis, fmm, fields, scattering

from lens_permittivity_profile_generator import generate_lens_permittivity_map
from field_postprocessing import make_jit_focusing_efficiency_function, intensity_map_from_fourier_amplitudes
from field_plotter import plot_amplitude_map


def generate_monochromatic_lens_symmetry_indices(
        n_lens_subpixels,
        relative_focal_point_position=(0.5, 0.5),
        tolerance_decimals=6
):
    single_coordinate_samples = np.linspace(0, 1, n_lens_subpixels)
    x_mesh, y_mesh = np.meshgrid(single_coordinate_samples, single_coordinate_samples)
    x_distances = np.abs(x_mesh - relative_focal_point_position[0])
    y_distances = np.abs(y_mesh - relative_focal_point_position[1])
    distances = x_distances ** 2 + y_distances ** 2
    unique_values, symmetry_indices = jnp.unique(np.round(distances, tolerance_decimals), return_inverse=True)
    return len(unique_values), symmetry_indices


def prepare_functions_for_optimization(
        wavelength,
        permittivity,
        lens_subpixel_size,
        n_lens_subpixels,
        lens_thickness,
        focal_length,
        approximate_number_of_terms
):
    total_lens_period = n_lens_subpixels * lens_subpixel_size
    n_unique_widths, symmetry_indices = generate_monochromatic_lens_symmetry_indices(n_lens_subpixels)

    primitive_lattice_vectors = basis.LatticeVectors(
        u=total_lens_period * basis.X, v=total_lens_period * basis.Y
    )
    expansion = basis.generate_expansion(
        primitive_lattice_vectors=primitive_lattice_vectors,
        approximate_num_terms=approximate_number_of_terms,
        truncation=basis.Truncation.CIRCULAR,
    )

    basis_indices_norm = np.linalg.norm(expansion.basis_coefficients, axis=-1)
    n_propagating_waves = np.count_nonzero(basis_indices_norm < total_lens_period / wavelength)
    propagating_basis_indices = tuple((int(i), int(j)) for (i, j) in expansion.basis_coefficients[:n_propagating_waves])

    in_plane_wavevector = jnp.array([0., 0.])
    solve_result_ambient = fmm.eigensolve_isotropic_media(
        permittivity=jnp.atleast_2d(1.),
        wavelength=jnp.asarray(wavelength),
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        expansion=expansion,
        formulation=fmm.Formulation.FFT
    )
    inc_fwd_amplitude = jnp.zeros(2 * len(expansion.basis_coefficients))
    zero_mode_index, = jnp.where(np.all(expansion.basis_coefficients == 0, axis=1))
    inc_fwd_amplitude = inc_fwd_amplitude.at[zero_mode_index].set(1.)
    fwd_amplitude = jnp.asarray(inc_fwd_amplitude[..., np.newaxis], dtype=jnp.float32)

    def focal_plane_field_fourier_amplitudes(unique_widths):
        # unique_widths = jnp.concatenate([unique_widths, jnp.zeros(1)])
        widths = unique_widths[symmetry_indices]
        shapes = jnp.stack([widths] + [jnp.zeros_like(widths)] * 3, axis=-1)
        permittivity_pattern = generate_lens_permittivity_map(
            shapes=shapes,
            sub_pixel_size=lens_subpixel_size,
            n_lens_subpixels=n_lens_subpixels,
            permittivity_pillar=permittivity
        )
        solve_result_crystal = fmm.eigensolve_isotropic_media(
            permittivity=permittivity_pattern,
            wavelength=jnp.asarray(wavelength),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            expansion=expansion,
            formulation=fmm.Formulation.FFT
        )
        s_matrices_interior = scattering.stack_s_matrices_interior(
            layer_solve_results=[solve_result_ambient, solve_result_crystal, solve_result_ambient],
            layer_thicknesses=[focal_length, lens_thickness, focal_length]  # type: ignore[arg-type]
        )
        amplitudes_interior = fields.stack_amplitudes_interior(
            s_matrices_interior=s_matrices_interior,
            forward_amplitude_0_start=jnp.zeros_like(fwd_amplitude),
            backward_amplitude_N_end=fwd_amplitude,
        )
        forward_amps, backwards_amps = amplitudes_interior[0]
        focal_plane_amps = fields.propagate_amplitude(
            amplitude=backwards_amps,
            distance=focal_length,  # type: ignore[arg-type]
            layer_solve_result=solve_result_ambient
        )
        (_, ef_y, _), _ = fields.fields_from_wave_amplitudes(
            forward_amplitude=forward_amps,
            backward_amplitude=focal_plane_amps,
            layer_solve_result=solve_result_ambient
        )
        return ef_y.flatten()

    jit_focal_plane_field_fourier_amplitudes = jax.jit(focal_plane_field_fourier_amplitudes)
    jit_focusing_efficiency_from_amplitudes = make_jit_focusing_efficiency_function(
        basis_indices=propagating_basis_indices,
        relative_focal_point=(0.5, 0.5)
    )

    def total_efficiency_from_unique_widths(unique_widths_normalized):
        unique_widths = unique_widths_normalized * lens_subpixel_size
        fourier_amps = jit_focal_plane_field_fourier_amplitudes(unique_widths)
        focusing_efficiency = jit_focusing_efficiency_from_amplitudes(amplitudes=fourier_amps[:n_propagating_waves])
        transmission_efficiency = jnp.sum(jnp.abs(fourier_amps) ** 2)
        return focusing_efficiency * transmission_efficiency

    return (
        focal_plane_field_fourier_amplitudes,
        total_efficiency_from_unique_widths,
        # n_unique_widths - 1,
        n_unique_widths - 1,
        propagating_basis_indices
    )


def run_optimization_and_visualization():
    wavelength = 450
    permittivity = 4
    lens_subpixel_size = 300
    n_lens_subpixels = 7
    lens_thickness = 500
    focal_length = 4000
    approximate_number_of_terms = 300

    (
        focal_plane_field_fourier_amplitudes,
        total_efficiency_from_unique_widths,
        n_unique_widths,
        propagating_basis_indices
    ) = prepare_functions_for_optimization(
        wavelength=wavelength,
        permittivity=permittivity,
        lens_subpixel_size=lens_subpixel_size,
        n_lens_subpixels=n_lens_subpixels,
        lens_thickness=lens_thickness,
        focal_length=focal_length,
        approximate_number_of_terms=approximate_number_of_terms,
    )

    # key = jax.random.key(2)
    # key, subkey = jax.random.split(key)
    # initial_unique_widths = jax.random.uniform(subkey, (n_unique_widths,), minval=0., maxval=1.)
    # initial_unique_widths = jnp.array([360., 300., 220., 235., 180., 85.]) / 400
    # initial_unique_widths = jnp.array([271.2412, 239.99948, 199.95596, 199.9913, 167.50484, 150.41992]) / 400
    # initial_unique_widths = 0.5 * jnp.ones(n_unique_widths)

    # print(len(expansion.basis_coefficients), focusing_efficiency_from_unique_widths(initial_unique_widths))

    # val_and_grad_f = jax.value_and_grad(total_efficiency_from_unique_widths)
    # current_unique_widths = initial_unique_widths
    # learning_rate = 0.05
    # # final_learning_rate = 0.01
    # n_iterations = 100
    # # learning_rate_step = (learning_rate - final_learning_rate) / n_iterations
    # for i in range(n_iterations):
    #     print(current_unique_widths * lens_subpixel_size)
    #     value, grad = val_and_grad_f(current_unique_widths)
    #     print(value)
    #     current_unique_widths += learning_rate * grad
    #     # learning_rate -= learning_rate_step
    #     current_unique_widths = jnp.clip(current_unique_widths, 0., 1.)

    def loss_fn(arg):
        return -total_efficiency_from_unique_widths(arg)

    loss_value_and_grad_fn = jax.value_and_grad(loss_fn)

    learning_rate = 0.01
    # optimizer = optax.sgd(learning_rate)
    optimizer = optax.adam(learning_rate, b1=0.5)

    # pillar_centsrs =

    x_init = 0.5 * jnp.ones(n_unique_widths)

    # key = jax.random.key(1)
    # x_init = jax.random.uniform(key, (n_unique_widths,))

    # x_init = jnp.asarray([266, 268, 190, 199,  130]) / 400

    # [266 268 190 199 130] - optimum so far, eff = 0.7390, obtained by uniform 200 pillars (fixed empty corner)
    # rate 0.05 stabilizes at ~50 steps (slowly drifting though)
    # rate 0.1 goes into borders quickly and stabilizes at eff = 0.72
    # rate 0.01 reaches 0.72 at ~15 steps, and then slowly drifts towards 0.7390, the best so fat
    # probably for bad initial guesses 0.05 is the best, 0.01 is too slow - OR NO, 0.05 SEEMS TO BE OFTEN UNSTABLE
    # actually there seems to be a lot of local flat areas
    # with initial x=all 300, fixed empty corner it drifted towards [361 315 325 239 204] eff=0.6790 (rate 0.01)
    # but didn't converge with 100 steps
    # with initial x=all 300, fixed empty corner and rate 0.05 it drifted into the edge [400 280 377 238 210] eff=0.6597
    # with initial x=all 100, fixed empty corner and rate 0.05 it drifted into the edge [304 209   0 194   0] eff=0.7209

    # GENERAL CONCLUSION ON THE GRADIENT DESCENT
    # quite sensitive to initial conditions
    # easily gets pushed towards corners
    # many flat regions that vary in efficiency (0.66, 0.72, 0.74 - so significantly)
    # it's possible to get a decent resul, but only by tweaking parameters

    # IDEA FOR VISUALIZATION
    # plot loss functions over steps for many different parameters, coloring lines that bumped into corners

    opt_state = optimizer.init(x_init)

    def project(x):
        return jnp.clip(x, 0., 1.)

    @jax.jit
    def step(x, opt_state):
        loss, grad = loss_value_and_grad_fn(x)
        updates, opt_state = optimizer.update(grad, opt_state)
        x = optax.apply_updates(x, updates)
        x = project(x)
        return x, opt_state, loss, grad

    x = x_init
    for i in range(100):
        new_x, opt_state, loss, grad = step(x, opt_state)
        avg_grad_norm = jnp.linalg.norm(grad) / len(x)
        rounded_widths = jnp.round(x * lens_subpixel_size).astype(int)
        print(f"Step {i}: efficiency={-loss:.4f}, widths={rounded_widths}, |grad|={avg_grad_norm}")
        if avg_grad_norm < 0.01:
            print("Stop on gradient norm criteria")
            break
        x = new_x


    # initial_amps = focal_plane_field_fourier_amplitudes(initial_unique_widths * lens_subpixel_size)[:len(propagating_basis_indices)]
    # # initial_eff = jit_focusing_efficiency_from_amplitudes(initial_amps) * jnp.sum(jnp.abs(initial_amps) ** 2)
    # # print(initial_eff)
    # initial_intensity_map = intensity_map_from_fourier_amplitudes(initial_amps, propagating_basis_indices)
    # initial_intensity_map = (initial_intensity_map + initial_intensity_map.T) / 2
    #
    # final_amps = focal_plane_field_fourier_amplitudes(current_unique_widths * lens_subpixel_size)[:len(propagating_basis_indices)]
    # # final_eff = jit_focusing_efficiency_from_amplitudes(final_amps) * jnp.sum(jnp.abs(final_amps) ** 2)
    # # print(final_eff)
    # intensity_map = intensity_map_from_fourier_amplitudes(final_amps, propagating_basis_indices)
    # intensity_map = (intensity_map + intensity_map.T) / 2
    #
    # max_intensity = max(float(jnp.max(intensity_map)), float(jnp.max(initial_intensity_map)))
    # fig, ax = plt.subplots(1, 2)
    # plot_amplitude_map(
    #     fig, ax[0], initial_intensity_map,
    #     wavelength_nm=wavelength, map_bounds=[0, n_lens_subpixels * lens_subpixel_size], vmax=max_intensity)
    # plot_amplitude_map(
    #     fig, ax[1], intensity_map,
    #     wavelength_nm=wavelength, map_bounds=[0, n_lens_subpixels * lens_subpixel_size], vmax=max_intensity)
    # plt.show()


if __name__ == '__main__':
    run_optimization_and_visualization()
