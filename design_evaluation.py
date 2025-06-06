import jax.numpy as jnp
import numpy as np
from design_monochromatic import generate_monochromatic_lens_symmetry_indices
from lens_permittivity_profile_generator import generate_lens_permittivity_map
from scattering_solver_factory import prepare_lens_pixel_width_to_scattered_amplitudes_function
from fmmax import basis, fields, fmm
from field_postprocessing import intensity_map_from_fourier_amplitudes
from field_plotter import plot_amplitude_map

import matplotlib.pyplot as plt


if __name__ == '__main__':
    wavelength = 650
    permittivity = 4
    lens_subpixel_size = 300
    n_lens_subpixels = 7
    lens_thickness = 500
    focal_length = 4000
    approximate_number_of_terms = 300
    total_lens_period = lens_subpixel_size * n_lens_subpixels

    n_unique_widths, symmetry_indices = generate_monochromatic_lens_symmetry_indices(
        n_lens_subpixels=n_lens_subpixels,
        # relative_focal_point_position=(0.5, 0.5)
        relative_focal_point_position=(0.25, 0.25)
    )

    # unique_widths = jnp.array([257, 234, 221, 196, 159, 86, 197, 158, 89], dtype=float)
    # unique_widths = jnp.array([260, 235, 220, 195, 160, 85, 200, 160, 90])
    unique_widths = jnp.array([200, 195, 170, 190, 107, 166, 113, 108, 192, 105, 112, 107, 108])
    widths = unique_widths[symmetry_indices]
    print(widths)

    shapes = jnp.stack([widths] + [jnp.zeros_like(widths)] * 3, axis=-1)
    permittivity_pattern = generate_lens_permittivity_map(
        shapes=shapes,
        sub_pixel_size=lens_subpixel_size,
        n_lens_subpixels=n_lens_subpixels,
        permittivity_pillar=permittivity
    )
    plt.imshow(permittivity_pattern, extent=(0, total_lens_period, 0, total_lens_period))
    plt.show()

    width_to_amps_f = prepare_lens_pixel_width_to_scattered_amplitudes_function(
        wavelength, permittivity, lens_subpixel_size, n_lens_subpixels, lens_thickness, approximate_number_of_terms
    )
    amps = width_to_amps_f(widths)
    trans_amps = amps[:len(amps) // 2]

    # propagate amplitudes by hand
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

    k0 = 2 * np.pi / wavelength
    K = 2 * np.pi / total_lens_period
    kx, ky = K * jnp.array(propagating_basis_indices).T
    kz = jnp.sqrt(k0 ** 2 - kx ** 2 - ky ** 2)
    print(trans_amps.shape, kz.shape)

    focal_plane_amps = trans_amps * jnp.exp(1j * kz * focal_length)
    intensity = intensity_map_from_fourier_amplitudes(
        amplitudes=focal_plane_amps,
        basis_indices=propagating_basis_indices
    )
    intensity = (intensity + intensity.T) / 2
    plot_amplitude_map(*plt.subplots(), intensity, wavelength_nm=wavelength)
    plt.show()
