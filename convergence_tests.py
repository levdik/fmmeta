import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from fmmax import basis, fmm, fields, scattering
from scattering_solver_factory import prepare_shapes_to_amplitudes_function
from scattering_solver_factory import prepare_lens_pixel_width_to_scattered_amplitudes_function
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
    convergence_data = []
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

        # def focusing_efficiency():
        #     fourier_amps = focal_plane_field_fourier_amplitudes()
        #     efficiency = calculate_focusing_efficiency(
        #         amplitudes=fourier_amps[:n_propagating_waves],
        #         basis_indices=propagating_basis_indices
        #     )
        #     return efficiency

        # convergence_data.append(float(focusing_efficiency()))
        convergence_data.append(complex(focal_plane_field_fourier_amplitudes()[0]))


    return convergence_data


def calculate_a00_convergence_data(
        wavelength,
        permittivity,
        lens_subpixel_size,
        n_lens_subpixels,
        lens_thickness,
        focal_length,
        approx_num_terms_range
):
    convergence_data = []

    a = jnp.ones([n_lens_subpixels, n_lens_subpixels]) / 3
    b = a / 2
    rngs_key_a, rngs_key_b = jax.random.split(jax.random.key(0))
    shapes = jnp.stack([
        a + 0.1 * jax.random.uniform(rngs_key_a, shape=a.shape),
        b + 0.05 * jax.random.uniform(rngs_key_b, shape=b.shape),
        jnp.zeros_like(a),
        jnp.zeros_like(a)
    ], axis=-1)
    shapes *= lens_subpixel_size

    for approximate_number_of_terms in approx_num_terms_range:
        common_func_prep_kwargs = {
            'permittivity': permittivity,
            'lens_subpixel_size': lens_subpixel_size,
            'n_lens_subpixels': n_lens_subpixels,
            'lens_thickness': lens_thickness,
            'approximate_number_of_terms': approximate_number_of_terms,
            'include_reflection': False,
            'return_basis_indices': True,
            'propagate_transmitted_amps_by_distance': focal_length
        }

        shapes_to_amps_function, basis_indices = prepare_shapes_to_amplitudes_function(
            wavelength=wavelength, **common_func_prep_kwargs)
        a00 = complex(shapes_to_amps_function(shapes)[0])
        print(approximate_number_of_terms, a00)
        convergence_data.append(a00)
    return convergence_data


def calculate_power_convergence(
        wavelength,
        permittivity,
        lens_subpixel_size,
        n_lens_subpixels,
        lens_thickness,
        approx_num_terms_range
):
    # TODO: compare width.shape=(n_subpixels, n_subpixels) and width.shape=(n_subpixels ** 2,)
    widths = jax.random.uniform(jax.random.key(42), shape=(n_lens_subpixels ** 2,)) * lens_subpixel_size
    power_convergence_data = []
    true_n_terms_data = []
    for approx_num_terms in approx_num_terms_range:
        total_lens_period = n_lens_subpixels * lens_subpixel_size
        primitive_lattice_vectors = basis.LatticeVectors(
            u=total_lens_period * basis.X, v=total_lens_period * basis.Y
        )
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=approx_num_terms,
            truncation=basis.Truncation.CIRCULAR,
        )
        true_n_terms = len(expansion.basis_coefficients)
        if true_n_terms in true_n_terms_data:
            print('already done:', approx_num_terms, true_n_terms)
            continue
        true_n_terms_data.append(true_n_terms)

        width_to_amps_f = prepare_lens_pixel_width_to_scattered_amplitudes_function(
            wavelength=wavelength,
            permittivity=permittivity,
            lens_subpixel_size=lens_subpixel_size,
            n_lens_subpixels=n_lens_subpixels,
            lens_thickness=lens_thickness,
            approximate_number_of_terms=approx_num_terms
        )

        amps = width_to_amps_f(widths)
        power_convergence_data.append(float(jnp.sum(jnp.abs(amps) ** 2)))
        print(approx_num_terms, power_convergence_data[-1])
    return power_convergence_data, true_n_terms_data


if __name__ == '__main__':
    # n_terms_range = list(range(2, 10, 1))
    # n_terms_range = list(range(50, 501, 50))
    # n_terms_range = list(range(250, 2001, 250))
    # n_terms_range = list(range(10, 101, 10))
    # n_terms_range = list(range(300, 1001, 50))
    n_terms_range = list(range(5, 101, 5))
    print(n_terms_range)

    # effs = calculate_lens_efficiency_convergence_data(
    #     wavelength=650,
    #     permittivity=4,
    #     lens_subpixel_size=400,
    #     n_lens_subpixels=1,
    #     lens_thickness=400,
    #     focal_length=4000,
    #     approx_num_terms_range=n_terms_range
    # )
    # print(effs)

    a00s = calculate_a00_convergence_data(
        wavelength=650,
        permittivity=4,
        lens_subpixel_size=400,
        n_lens_subpixels=4,
        lens_thickness=400,
        focal_length=4000,
        approx_num_terms_range=n_terms_range
    )
    print(a00s)

    # data = np.array(effs)
    data = np.array(a00s)
    plt.plot(n_terms_range[1:], np.abs(data[1:] - data[:-1]), '-o')
    plt.yscale('log')
    plt.gray()
    plt.show()

    # n_lens_subpixels = 8
    # powers, true_n_terms = calculate_power_convergence(
    #     wavelength=650,
    #     permittivity=4,
    #     lens_subpixel_size=400,
    #     n_lens_subpixels=n_lens_subpixels,
    #     lens_thickness=800,
    #     approx_num_terms_range=n_terms_range
    # )
    # print(powers)
    # plt.plot(true_n_terms, np.abs(np.array(powers) - 1))
    # plt.yscale('log')
    # plt.xlabel('Number of terms')
    # plt.ylabel('Power error')
    # plt.title(f'Power convergence, {n_lens_subpixels} pixels')
    # plt.show()
