import jax
import jax.numpy as jnp
import numpy as np
import optax

from fmmax import basis, fmm, fields, scattering

from lens_permittivity_profile_generator import generate_lens_permittivity_map
from field_postprocessing import make_jit_focusing_efficiency_function


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
        n_unique_widths - 1,
        propagating_basis_indices
    )


def run_optimization_and_visualize_results():
    wavelength = 650
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

    def loss_fn(arg):
        return -total_efficiency_from_unique_widths(arg)

    loss_value_and_grad_fn = jax.value_and_grad(loss_fn)

    learning_rate = 0.01
    # optimizer = optax.sgd(learning_rate)
    optimizer = optax.adam(learning_rate, b1=0.5)

    x_init = 0.5 * jnp.ones(n_unique_widths)

    # key = jax.random.key(1)
    # x_init = jax.random.uniform(key, (n_unique_widths,))

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
    for i in range(100):
        new_x, opt_state, loss, grad = step(x, opt_state)
        avg_grad_norm = jnp.linalg.norm(grad) / len(x)
        rounded_widths = jnp.round(x * lens_subpixel_size).astype(int)
        print(f"Step {i}: efficiency={-loss:.4f}, widths={rounded_widths}, |grad|={avg_grad_norm}")
        if avg_grad_norm < 0.01:
            print("Stop on gradient norm criteria")
            break
        x = new_x


if __name__ == '__main__':
    run_optimization_and_visualize_results()
