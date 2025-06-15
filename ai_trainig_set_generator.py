import jax
import jax.numpy as jnp
import numpy as np

from fmmax import basis, fmm, fields, scattering
from lens_permittivity_profile_generator import generate_lens_permittivity_map


def prepare_amplitude_generating_function(
        wavelength,
        permittivity,
        lens_subpixel_size,
        n_lens_subpixels,
        lens_thickness,
        approximate_number_of_terms
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

    def trans_ref_fourier_amplitudes(widths):
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
            forward_amplitude_0_start=jnp.zeros_like(fwd_amplitude),
            backward_amplitude_N_end=fwd_amplitude,
        )
        _, trans_amps = amplitudes_interior[0]

        ref_amps, _ = amplitudes_interior[2]
        (_, ref_e_y, _), _ = fields.fields_from_wave_amplitudes(
            forward_amplitude=ref_amps,
            backward_amplitude=jnp.zeros_like(ref_amps),
            layer_solve_result=solve_result_ambient
        )
        (_, trans_e_y, _), _ = fields.fields_from_wave_amplitudes(
            forward_amplitude=trans_amps,
            backward_amplitude=jnp.zeros_like(trans_amps),
            layer_solve_result=solve_result_ambient
        )
        trans_ref_propagating_amplitudes = jnp.concatenate([
            trans_e_y[:n_propagating_waves].flatten(), ref_e_y[:n_propagating_waves].flatten()
        ])
        return trans_ref_propagating_amplitudes

    jit_trans_ref_fourier_amplitudes = jax.jit(trans_ref_fourier_amplitudes)
    return jit_trans_ref_fourier_amplitudes


def generate_and_save_training_set():
    wavelength = 650
    permittivity = 4
    lens_subpixel_size = 300
    n_lens_subpixels = 7
    lens_thickness = 500
    approximate_number_of_terms = 300

    f = prepare_amplitude_generating_function(
        wavelength=wavelength,
        permittivity=permittivity,
        lens_subpixel_size=lens_subpixel_size,
        n_lens_subpixels=n_lens_subpixels,
        lens_thickness=lens_thickness,
        approximate_number_of_terms=approximate_number_of_terms
    )

    # print(f(jnp.zeros((n_lens_subpixels, n_lens_subpixels))))
    # return

    map_f = jax.vmap(f)

    n_training_samples = 100

    widths = lens_subpixel_size * jax.random.uniform(
        jax.random.key(0),
        shape=(n_training_samples, n_lens_subpixels, n_lens_subpixels)
    )
    trans_ref_propagating_amplitudes = map_f(widths)
    jnp.savez('ai_training_data/red_th500.npz', amps=trans_ref_propagating_amplitudes, widths=widths)


def _examine_training_set():
    jnp.set_printoptions(linewidth=1000)

    data = jnp.load('ai_training_data/red_7x7_th500_p300_120k.npz')
    widths = jnp.array(data['widths'])
    amplitudes = jnp.array(data['amps'])
    print(widths.shape, amplitudes.shape)
    print(widths)
    n_modes = amplitudes.shape[1] // 2
    trans_amps = amplitudes[:, :n_modes]
    ref_amps = amplitudes[:, n_modes:]
    print('Uniform amp:', jnp.sqrt(1 / n_modes))
    print(jnp.min(jnp.abs(trans_amps), axis=0))
    print(jnp.max(jnp.abs(trans_amps), axis=0))

    powers = jnp.sum(jnp.abs(amplitudes) ** 2, axis=-1)
    print(jnp.mean(jnp.abs(powers - 1)))


if __name__ == "__main__":
    # generate_and_save_training_set()
    _examine_training_set()
