import jax
import jax.numpy as jnp
import numpy as np

from fmmax import basis, fmm, fields, scattering
import refractiveindex2 as ri

from wave_pattern_factory import generate_wave_permittivity_primary_basis_indices, generate_wave_permittivity_pattern

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

import pickle

SIO2 = ri.RefractiveIndexMaterial(shelf="main", book="SiO2", page="Malitson")
SI3N4 = ri.RefractiveIndexMaterial(shelf="main", book="Si3N4", page="Luke")


def generate_wave_pattern_basis_and_random_amps(rng_key, r_max):
    r_key, pattern_key, a_00_key = jax.random.split(rng_key, num=3)
    r = jax.random.uniform(r_key, minval=1, maxval=10)

    (
        full_basis_indices, primary_basis, symmetry_indices
    ) = generate_wave_permittivity_primary_basis_indices(r_max, symmetry_type='main_diagonal')

    amps_primary = jax.random.uniform(pattern_key, minval=-1, maxval=1, shape=(2, len(primary_basis)))
    amps_primary = amps_primary[0] + 1j * amps_primary[1]
    a_00 = jnp.clip(0.3 * jax.random.normal(a_00_key), -1, 1)
    amps_primary = amps_primary.at[0].set(a_00)

    for i, (n, m) in enumerate(primary_basis):
        if n ** 2 + m ** 2 > r ** 2:
            amps_primary = amps_primary.at[i].set(0)

    return amps_primary, full_basis_indices, primary_basis, symmetry_indices


def show_example_random_permittivity_patterns(shape):
    rng_key = jax.random.key(1)
    fig, ax = plt.subplots(*shape)

    for ax_i in ax.flatten():
        rng_key, rng_subkey = jax.random.split(rng_key)
        (
            amps_primary, full_basis_indices, _, symmetry_indices
        ) = generate_wave_pattern_basis_and_random_amps(rng_subkey, 10)
        amps = amps_primary[symmetry_indices]
        pattern = generate_wave_permittivity_pattern(
            amplitudes=amps,
            basis_indices=full_basis_indices,
            permittivity=2,
            permittivity_ambience=1.,
            resolution=100
        )
        ax_i.imshow(pattern, cmap='gray', vmax=2, vmin=1)
        ax_i.set_axis_off()

    plt.tight_layout()
    plt.show()


def prepare_permittivity_pattern_to_scattered_amps_function(
        wavelength,
        period,
        lens_thickness,
        substrate_thickness,
        permittivity_substrate,
        approximate_number_of_terms
):
    primitive_lattice_vectors = basis.LatticeVectors(
        u=period * basis.X, v=period * basis.Y
    )
    expansion = basis.generate_expansion(
        primitive_lattice_vectors=primitive_lattice_vectors,
        approximate_num_terms=approximate_number_of_terms,
        truncation=basis.Truncation.CIRCULAR,
    )

    basis_indices_norm = np.linalg.norm(expansion.basis_coefficients, axis=-1)
    n_propagating_waves = np.count_nonzero(basis_indices_norm < period / wavelength)

    in_plane_wavevector = jnp.array([0., 0.])
    solve_result_ambient = fmm.eigensolve_isotropic_media(
        permittivity=jnp.atleast_2d(1.),
        wavelength=jnp.asarray(wavelength),
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        expansion=expansion,
        formulation=fmm.Formulation.FFT
    )
    solve_result_substrate = fmm.eigensolve_isotropic_media(
        permittivity=jnp.atleast_2d(permittivity_substrate),
        wavelength=jnp.asarray(wavelength),
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        expansion=expansion,
        formulation=fmm.Formulation.FFT
    )
    inc_fwd_amplitude = jnp.zeros(2 * len(expansion.basis_coefficients))
    zero_mode_index, = jnp.where(jnp.all(expansion.basis_coefficients == 0, axis=1))
    inc_fwd_amplitude = inc_fwd_amplitude.at[zero_mode_index].set(1.)
    fwd_amplitude = jnp.asarray(inc_fwd_amplitude[..., jnp.newaxis], dtype=float)

    def permittivity_pattern_to_scattered_amps_func(permittivity_pattern):
        solve_result_crystal = fmm.eigensolve_isotropic_media(
            permittivity=permittivity_pattern,
            wavelength=jnp.asarray(wavelength),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            expansion=expansion,
            formulation=fmm.Formulation.FFT
        )
        s_matrices_interior = scattering.stack_s_matrices_interior(
            layer_solve_results=[solve_result_ambient, solve_result_substrate, solve_result_crystal, solve_result_ambient],
            layer_thicknesses=[0., substrate_thickness, lens_thickness, 0.]
            # type: ignore[arg-type]
        )
        amplitudes_interior = fields.stack_amplitudes_interior(
            s_matrices_interior=s_matrices_interior,
            forward_amplitude_0_start=jnp.zeros_like(fwd_amplitude),
            backward_amplitude_N_end=fwd_amplitude,
        )
        _, trans_amps = amplitudes_interior[0]
        (_, trans_e_y, _), _ = fields.fields_from_wave_amplitudes(
            forward_amplitude=trans_amps,
            backward_amplitude=jnp.zeros_like(trans_amps),
            layer_solve_result=solve_result_ambient
        )
        propagating_trans_e_y_amps = trans_e_y[:n_propagating_waves].flatten()
        return propagating_trans_e_y_amps

    return permittivity_pattern_to_scattered_amps_func


def prepare_simulate_scattering_and_derivative_func(
        wave_pattern_full_basis_indices,
        amp_symmetry_indices,
        wavelength,
        period,
        lens_permittivity,
        lens_thickness,
        substrate_thickness,
        permittivity_substrate,
        approximate_number_of_terms
):
    permittivity_pattern_to_scattered_amps_func = prepare_permittivity_pattern_to_scattered_amps_function(
        wavelength,
        period,
        lens_thickness,
        substrate_thickness,
        permittivity_substrate,
        approximate_number_of_terms
    )

    def result_func(wave_pattern_primary_amps, project_d1_onto):
        def simulate_scattering(primary_pattern_amps_re_im):
            n_params = len(primary_pattern_amps_re_im) // 2
            primary_pattern_amps = primary_pattern_amps_re_im[:n_params] + 1j * primary_pattern_amps_re_im[n_params:]
            pattern_amps = primary_pattern_amps[amp_symmetry_indices]

            permittivity_pattern = generate_wave_permittivity_pattern(
                amplitudes=pattern_amps,
                basis_indices=wave_pattern_full_basis_indices,
                permittivity=lens_permittivity,
                permittivity_ambience=1.,
                resolution=100
            )

            scattered_field_amps = permittivity_pattern_to_scattered_amps_func(permittivity_pattern)
            scattered_field_amps_re_im = jnp.concatenate([scattered_field_amps.real, scattered_field_amps.imag])
            return scattered_field_amps_re_im

        def simulate_scattering_and_project(primary_pattern_amps_re_im):
            scattered_field_amps_re_im = simulate_scattering(primary_pattern_amps_re_im)
            return jnp.dot(scattered_field_amps_re_im, project_d1_onto)

        projected_jac_func = jax.jacobian(simulate_scattering_and_project)

        wave_pattern_primary_amps_re_im = jnp.concatenate([wave_pattern_primary_amps.real, wave_pattern_primary_amps.imag])
        amps = simulate_scattering(wave_pattern_primary_amps_re_im)
        projected_jac = projected_jac_func(wave_pattern_primary_amps_re_im)
        return amps, projected_jac

    return result_func


wavelength = 650
period = 2000
lens_thickness = 600
substrate_thickness = 500
approximate_number_of_terms = 600

# TODO: calculate n_propagating_waves
n_propagating_waves = 29

lens_permittivity = SI3N4.get_refractive_index(wavelength_um=wavelength / 1000) ** 2
permittivity_substrate = SIO2.get_refractive_index(wavelength_um=wavelength / 1000) ** 2

(
    full_basis_indices, primary_basis_indices, symmetry_indices
) = generate_wave_permittivity_primary_basis_indices(r=10, symmetry_type='main_diagonal')

scattering_and_derivative_func = prepare_simulate_scattering_and_derivative_func(
    wave_pattern_full_basis_indices=full_basis_indices,
    amp_symmetry_indices=symmetry_indices,
    wavelength=wavelength,
    period=period,
    lens_permittivity=lens_permittivity,
    lens_thickness=lens_thickness,
    substrate_thickness=substrate_thickness,
    permittivity_substrate=permittivity_substrate,
    approximate_number_of_terms=approximate_number_of_terms,
)
jitted_scattering_and_derivative_func = jax.jit(scattering_and_derivative_func)


def run_batch_and_save(batch_size, key_seed):
    rng_batch_key = jax.random.key(key_seed)
    results = []
    rng_batch_key, rng_run_key = jax.random.split(rng_batch_key)
    for i in range(batch_size):
        rng_run_key, rng_subkey = jax.random.split(rng_run_key)
        (
            amps_primary, _, _, _
        ) = generate_wave_pattern_basis_and_random_amps(rng_subkey, r_max=10)

        rng_run_key, rng_subkey = jax.random.split(rng_run_key)
        v1 = jax.random.normal(rng_subkey, shape=(2 * n_propagating_waves,))
        v1 = v1 / jnp.linalg.norm(v1)

        results.append((amps_primary, v1) + jitted_scattering_and_derivative_func(amps_primary, v1))
        print('run', i + 1)

    with open(f'wave_pattern_training_data/{key_seed}.pkl', 'wb') as file:
        pickle.dump(results, file)


def load_and_save_as_maps():
    primitive_lattice_vectors = basis.LatticeVectors(
        u=period * basis.X, v=period * basis.Y
    )
    expansion = basis.generate_expansion(
        primitive_lattice_vectors=primitive_lattice_vectors,
        approximate_num_terms=approximate_number_of_terms,
        truncation=basis.Truncation.CIRCULAR,
    )
    basis_indices_norm = np.linalg.norm(expansion.basis_coefficients, axis=-1)
    n_propagating_waves = np.count_nonzero(basis_indices_norm < period / wavelength)
    expansion = expansion.basis_coefficients[:n_propagating_waves]

    data = np.load('wave_pattern_training_data/wave_red_30k.npz')
    pattern_amps = data['primary_pattern_amps']
    field_amps = data['scattered_field_amps']
    field_amps = field_amps[:, :len(expansion)] + 1j * field_amps[:, len(expansion):]

    x = []
    y = []

    for i in range(len(pattern_amps)):
    # for i in range(1000):
        if i % 100 == 0:
            print(i)
        pattern = generate_wave_permittivity_pattern(
            amplitudes=jnp.array(pattern_amps[i])[symmetry_indices],
            basis_indices=full_basis_indices,
            permittivity=1.,
            permittivity_ambience=0.,
            resolution=64
        )
        pattern = np.array(pattern)

        field = np.zeros((64, 64), dtype=complex)
        field[expansion[:, 0], expansion[:, 1]] = field_amps[i]
        field = np.fft.ifft2(field) * (64 ** 2)

        # print(np.linalg.norm(
        #     np.fft.fft2(field)[expansion[:, 0], expansion[:, 1]]
        #     - field_amps[i]
        # ))

        # fig, ax = plt.subplots(1, 3)
        # ax[0].imshow(pattern)
        # ax[1].imshow(field.real)
        # ax[2].imshow(field.imag)
        # plt.show()

        x.append(pattern)
        y.append(field)

    x = np.stack(x)
    y = np.stack(y)
    print(x.shape, y.shape)
    np.savez('wave_pattern_training_data/wave_red_30k_maps.npz', x=x, y=y)


if __name__ == '__main__':
    # TODO: choose wavelengths (either uniformly in the range or one of the samples in the eval)
    #  and use refractiveindex2
    # TODO: experiment with n_pixels

    # show_example_random_permittivity_patterns(shape=[4, 7])

    # run_batch_and_save(batch_size=2, key_seed=1)

    load_and_save_as_maps()
