import jax
import jax.numpy as jnp
import numpy as np
import optax

from fmmax import basis, fmm, fields, scattering

from lens_permittivity_profile_generator import generate_lens_permittivity_map
from field_postprocessing import make_jit_focusing_efficiency_function


n_lens_subpixels = 8
symmetry_indices = jnp.array([
    [1, 2, 2, 1, 3, 4, 4, 3],
    [2, 0, 0, 2, 4, 5, 5, 4],
    [2, 0, 0, 2, 4, 5, 5, 4],
    [1, 2, 2, 1, 3, 4, 4, 3],
    [3, 4, 4, 3, 1, 2, 2, 1],
    [4, 5, 5, 4, 2, 0, 0, 2],
    [4, 5, 5, 4, 2, 0, 0, 2],
    [3, 4, 4, 3, 1, 2, 2, 1]
], dtype=int)
n_unique_widths = 6


def prepare_functions_for_optimization(
        wavelength,
        permittivity,
        lens_subpixel_size,
        lens_thickness,
        focal_length,
        approximate_number_of_terms,
        relative_focal_point_position
):
    total_lens_period = n_lens_subpixels * lens_subpixel_size

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
            layer_thicknesses=[0., lens_thickness, 0.]  # type: ignore[arg-type]
        )
        amplitudes_interior = fields.stack_amplitudes_interior(
            s_matrices_interior=s_matrices_interior,
            forward_amplitude_0_start=fwd_amplitude,
            backward_amplitude_N_end=jnp.zeros_like(fwd_amplitude),
        )
        transmitted_forward_amps, _ = amplitudes_interior[-1]
        focal_plane_amps = fields.propagate_amplitude(
            amplitude=transmitted_forward_amps,
            distance=focal_length,  # type: ignore[arg-type]
            layer_solve_result=solve_result_ambient
        )
        (_, ef_y, _), _ = fields.fields_from_wave_amplitudes(
            forward_amplitude=focal_plane_amps,
            backward_amplitude=jnp.zeros_like(focal_plane_amps),
            layer_solve_result=solve_result_ambient
        )
        return ef_y.flatten()

    jit_focal_plane_field_fourier_amplitudes = jax.jit(focal_plane_field_fourier_amplitudes)
    jit_focusing_efficiency_from_amplitudes = make_jit_focusing_efficiency_function(
        basis_indices=propagating_basis_indices,
        relative_focal_point=relative_focal_point_position
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
        propagating_basis_indices
    )


def run_optimization_and_visualize_results(wavelength, lens_subpixel_size, lens_thickness):
    permittivity = 4
    focal_length = 4000
    approximate_number_of_terms = 300
    relative_focal_point_position = (0.25, 0.75)
    total_lens_period = n_lens_subpixels * lens_subpixel_size

    (
        focal_plane_field_fourier_amplitudes,
        total_efficiency_from_unique_widths,
        propagating_basis_indices,
    ) = prepare_functions_for_optimization(
        wavelength=wavelength,
        permittivity=permittivity,
        lens_subpixel_size=lens_subpixel_size,
        lens_thickness=lens_thickness,
        focal_length=focal_length,
        approximate_number_of_terms=approximate_number_of_terms,
        relative_focal_point_position=relative_focal_point_position
    )

    def loss_fn(arg):
        return -total_efficiency_from_unique_widths(arg)

    loss_value_and_grad_fn = jax.value_and_grad(loss_fn)

    learning_rate = 0.01
    optimizer = optax.adam(learning_rate, b1=0.9)

    x_init = 0.5 * jnp.ones(n_unique_widths)
    # from translibs.translib_square_th700_p300 import a as lib_a, transmissions as lib_transmissions
    # lib_a = lib_a[:-1]
    # lib_transmissions = lib_transmissions[:-1]
    # from lens_forward_design import find_best_fit_pillars
    # from phase_profile_manager import generate_target_phase
    # from lens_permittivity_profile_generator import generate_pillar_center_positions
    # pillar_centers = generate_pillar_center_positions(
    #     lens_subpixel_size=lens_subpixel_size, n_lens_subpixels=n_lens_subpixels)
    # target_phases = generate_target_phase(
    #     points=pillar_centers,
    #     focal_points=[
    #         [-total_lens_period/4, total_lens_period/4, focal_length],
    #         [total_lens_period/4, -total_lens_period/4, focal_length]
    #     ],
    #     wavelength=wavelength,
    #     xy_period=total_lens_period
    # )
    # target_phases -= target_phases.reshape(n_lens_subpixels, n_lens_subpixels)[0, 1]
    # # print(target_phases.reshape(n_lens_subpixels, n_lens_subpixels))
    # best_fit_pillar_indices = find_best_fit_pillars(jnp.exp(1j * target_phases), lib_transmissions, n_wavelengths=1)
    # best_fit_pillar_widths = lib_a[best_fit_pillar_indices].reshape(n_lens_subpixels, -1)
    # unique_widths_init = np.zeros(n_unique_widths)
    # for sym_i, a_i in zip(symmetry_indices.flatten(), best_fit_pillar_widths.flatten()):
    #     unique_widths_init[sym_i] = a_i
    # print(unique_widths_init.astype(int).tolist())
    # # return
    # x_init = jnp.array(unique_widths_init / lens_subpixel_size)
    # print(x_init)

    opt_state = optimizer.init(x_init)

    def project_onto_boundaries(x):
        return jnp.clip(x, 0., 1.)

    @jax.jit
    def step(x, opt_state):
        loss, grad = loss_value_and_grad_fn(x)
        updates, opt_state = optimizer.update(grad, opt_state)
        x = optax.apply_updates(x, updates)
        x = project_onto_boundaries(x)
        return x, opt_state, loss, grad

    x = x_init
    max_eff = 0.

    for i in range(100):
        new_x, opt_state, loss, grad = step(x, opt_state)
        avg_grad_norm = jnp.linalg.norm(grad) / len(x)
        rounded_widths = jnp.round(x * lens_subpixel_size).astype(int)
        max_eff = max(-loss, max_eff)
        # print(f"Step {i}: efficiency={-loss:.4f}, widths={rounded_widths}, |grad|={avg_grad_norm}")
        # if avg_grad_norm < 0.001:
        #     print("Stop on gradient norm criteria")
        #     break
        print(i + 1, -loss, sep='\t')
        x = new_x
    print('Max efficiency:', max_eff)
    print(rounded_widths[symmetry_indices])


if __name__ == '__main__':
    jnp.set_printoptions(linewidth=1000)

    run_optimization_and_visualize_results(
        wavelength=650,
        lens_subpixel_size=300,
        lens_thickness=700
    )
