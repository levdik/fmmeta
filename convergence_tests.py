import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from fmmax import basis, fmm, fields, scattering

from lens_permittivity_profile_generator import generate_lens_permittivity_map
from field_postprocessing import calculate_focusing_efficiency


def calculate_lens_efficiency_convergence_data(
        wavelength,
        permittivity,
        lens_subpixel_size,
        n_lens_subpixels,
        lens_thickness,
        focal_length,
        approx_num_terms_range
):
    efficiencies = []
    key = jax.random.key(1)
    widths = jax.random.uniform(key, (n_lens_subpixels, n_lens_subpixels), minval=0., maxval=lens_subpixel_size)

    for approximate_number_of_terms in approx_num_terms_range:
        print(approximate_number_of_terms)
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
        propagating_basis_indices = tuple(
            (int(i), int(j)) for (i, j) in expansion.basis_coefficients[:n_propagating_waves])

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

        def focal_plane_field_fourier_amplitudes():
            shapes = jnp.stack([widths] + [jnp.zeros_like(widths)] * 3, axis=-1)
            permittivity_pattern = generate_lens_permittivity_map(
                shapes=shapes,
                sub_pixel_size=lens_subpixel_size,
                n_lens_subpixels=n_lens_subpixels,
                permittivity_pillar=permittivity,
                n_samples_per_subpixel=100
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

        def focusing_efficiency():
            fourier_amps = focal_plane_field_fourier_amplitudes()
            efficiency = calculate_focusing_efficiency(
                amplitudes=fourier_amps[:n_propagating_waves],
                basis_indices=propagating_basis_indices
            )
            return efficiency

        efficiencies.append(float(focusing_efficiency()))

    return efficiencies


if __name__ == '__main__':
    n_terms_range = list(range(50, 501, 50))
    # n_terms_range = list(range(250, 2001, 250))
    # n_terms_range = list(range(10, 101, 10))
    print(n_terms_range)

    effs = calculate_lens_efficiency_convergence_data(
        wavelength=650,
        permittivity=4,
        lens_subpixel_size=400,
        n_lens_subpixels=6,
        lens_thickness=400,
        focal_length=4000,
        approx_num_terms_range=n_terms_range
    )

    print(effs)

    effs = np.array(effs)
    plt.plot(n_terms_range[1:], np.abs(effs[1:] - effs[:-1]), '-o')
    plt.yscale('log')
    plt.gray()
    plt.show()
