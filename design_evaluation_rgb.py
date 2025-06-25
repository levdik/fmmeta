import jax
import jax.numpy as jnp

from scattering_solver_factory import prepare_shapes_to_amplitudes_function
from field_postprocessing import calculate_focusing_efficiency, intensity_map_from_fourier_amplitudes
from lens_permittivity_profile_generator import generate_lens_permittivity_map
from field_plotter import plot_amplitude_map

import matplotlib.pyplot as plt

wavelengths = (650, 550, 450)
bayer_relative_focal_points = [
    ((0.25, 0.75),),
    ((0.25, 0.25), (0.75, 0.75)),
    ((0.75, 0.25),)
]

n_lens_subpixels = 8
n_unique_shapes = 10
# symmetry_indices = jnp.array([
#     [1, 2, 2, 1, 8, 9, 9, 8],
#     [3, 0, 0, 3, 9, 7, 7, 9],
#     [3, 0, 0, 3, 9, 7, 7, 9],
#     [1, 2, 2, 1, 8, 9, 9, 8],
#     [5, 6, 6, 5, 1, 2, 2, 1],
#     [6, 4, 4, 6, 3, 0, 0, 3],
#     [6, 4, 4, 6, 3, 0, 0, 3],
#     [5, 6, 6, 5, 1, 2, 2, 1]
# ])
symmetry_indices = jnp.array([
    [8, 9, 9, 8, 1, 2, 2, 1],
    [9, 7, 7, 9, 3, 0, 0, 3],
    [9, 7, 7, 9, 3, 0, 0, 3],
    [8, 9, 9, 8, 1, 2, 2, 1],
    [1, 2, 2, 1, 5, 6, 6, 5],
    [3, 0, 0, 3, 6, 4, 4, 6],
    [3, 0, 0, 3, 6, 4, 4, 6],
    [1, 2, 2, 1, 5, 6, 6, 5]
])

permittivity = 4
lens_subpixel_size = 300
lens_thickness = 1000
focal_length = 4000
approximate_number_of_terms = 500

total_lens_period = n_lens_subpixels * lens_subpixel_size


def evaluate_design(a_unique, b_unique):
    a, b = a_unique[symmetry_indices], b_unique[symmetry_indices]
    ah = bh = jnp.zeros_like(a)
    shapes = jnp.stack([a, b, ah, bh], axis=-1)
    permittivity_map = generate_lens_permittivity_map(
        shapes=shapes,
        sub_pixel_size=lens_subpixel_size,
        n_lens_subpixels=n_lens_subpixels,
        permittivity_pillar=permittivity
    )
    plt.imshow(permittivity_map, origin='lower', extent=(0., total_lens_period, 0., total_lens_period))
    plt.show()

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

    fig, ax = plt.subplots(1, 3)

    for i, wavelength in enumerate(wavelengths):
        shapes_to_amps_function, basis_indices = prepare_shapes_to_amplitudes_function(
            wavelength=wavelength, **common_func_prep_kwargs)
        focal_plane_amps = shapes_to_amps_function(shapes)
        focusing_eff = calculate_focusing_efficiency(focal_plane_amps, basis_indices, bayer_relative_focal_points[i][0])
        print(wavelength, focusing_eff)
        intensity_map = intensity_map_from_fourier_amplitudes(focal_plane_amps, basis_indices)
        plot_amplitude_map(fig, ax[i], intensity_map, wavelength_nm=wavelength, map_bounds=[0, total_lens_period])

    plt.show()


if __name__ == '__main__':
    # a = [96, 88, 110, 107, 103, 88, 85, 94, 95, 93]
    # b = [48, 38, 57, 59, 41, 38, 35, 43, 45, 43]
    # a = [114, 103, 122, 121, 118, 106, 97, 114, 123, 115]
    # b = [28, 12, 35, 35, 31, 19, 23, 17, 22, 20]
    a = [114, 116, 116, 113, 106, 111, 100, 123, 131, 125]
    b = [22, 26, 29, 27, 16, 23, 27, 25, 27, 29]
    a, b = jnp.array(a), jnp.array(b)

    print(a + 2 * b)

    evaluate_design(a, b)
