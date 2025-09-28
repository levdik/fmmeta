import jax.numpy as jnp
import numpy as np

from lens_permittivity_profile_generator import generate_lens_permittivity_map
from scattering_solver_factory import prepare_shapes_to_amplitudes_function
from field_postprocessing import calculate_focusing_efficiency, intensity_map_from_fourier_amplitudes
from field_plotter import plot_amplitude_map

import matplotlib.pyplot as plt


# n_lens_subpixels = 7
# symmetry_indices = jnp.array([
#     [9, 8, 7, 6, 7, 8, 9],
#     [8, 5, 3, 4, 3, 5, 8],
#     [7, 3, 2, 1, 2, 3, 7],
#     [6, 4, 1, 0, 1, 4, 6],
#     [7, 3, 2, 1, 2, 3, 7],
#     [8, 5, 3, 4, 3, 5, 8],
#     [9, 8, 7, 6, 7, 8, 9]
# ], dtype=int)
# n_unique_widths = 10
# unique_widths = jnp.array([71, 211, 204, 154, 159, 102, 186, 106, 111, 43])

# n_lens_subpixels = 8
# symmetry_indices = jnp.array([
#     [1, 2, 2, 1, 3, 4, 4, 3],
#     [2, 0, 0, 2, 4, 5, 5, 4],
#     [2, 0, 0, 2, 4, 5, 5, 4],
#     [1, 2, 2, 1, 3, 4, 4, 3],
#     [3, 4, 4, 3, 1, 2, 2, 1],
#     [4, 5, 5, 4, 2, 0, 0, 2],
#     [4, 5, 5, 4, 2, 0, 0, 2],
#     [3, 4, 4, 3, 1, 2, 2, 1]
# ], dtype=int)
# n_unique_widths = 6
# unique_widths = jnp.array([38, 215, 198, 78, 106, 147])
# unique_widths = jnp.array([146, 76, 104, 216, 198, 45])
# unique_widths = jnp.array([284, 288, 282, 300, 296, 85])
# unique_widths = jnp.array([280, 0, 60, 80, 100, 110])
# unique_widths = jnp.array([134, 110, 102, 188, 183, 38])

# n_lens_subpixels = 5
# widths = jnp.array([
# [228, 161, 181, 161, 228],
# [161, 1,   10,  1,   161],
# [181, 10 , 4,   10,  181],
# [161, 1,   10,  1,   161],
# [228, 161, 181, 161, 228]
# ])
# widths = jnp.array([
# [230, 160, 180, 160, 230],
# [160, 0,   0,   0,   160],
# [180, 0 ,  0,   0,   180],
# [160, 0,   0,   0,   160],
# [230, 160, 180, 160, 230]
# ])
# widths = jnp.roll(widths, 1, axis=(0, 1))

n_lens_subpixels = 8

a = jnp.array([
    [66, 104, 104, 66, 103, 111, 111, 103],
    [112, 110, 110, 112, 111, 124, 124, 111],
    [112, 110, 110, 112, 111, 124, 124, 111],
    [66, 104, 104, 66, 103, 111, 111, 103],
    [116, 64, 64, 116, 66, 112, 112, 66],
    [64, 101, 101, 64, 104, 110, 110, 104],
    [64, 101, 101, 64, 104, 110, 110, 104],
    [116, 64, 64, 116, 66, 112, 112, 66]
])
b = jnp.array([
    [7, 9, 9, 7, 14, 23, 23, 14],
    [23, 24, 24, 23, 23, 39, 39, 23],
    [23, 24, 24, 23, 23, 39, 39, 23],
    [7, 9, 9, 7, 14, 23, 23, 14],
    [33, 1, 1, 33, 7, 23, 23, 7],
    [1, 9, 9, 1, 9, 24, 24, 9],
    [1, 9, 9, 1, 9, 24, 24, 9],
    [33, 1, 1, 33, 7, 23, 23, 7]
])
shapes = jnp.stack([a, b] + [jnp.zeros_like(a)] * 2, axis=-1)



wavelength = 450
permittivity = 4
lens_subpixel_size = 300
lens_thickness = 1700
focal_length = 4000
approximate_number_of_terms = 300
total_lens_period = lens_subpixel_size * n_lens_subpixels


def evaluate_design():
    # shapes = jnp.stack([widths] + [jnp.zeros_like(widths)] * 3, axis=-1)
    permittivity_map = generate_lens_permittivity_map(
        shapes=shapes,
        sub_pixel_size=lens_subpixel_size,
        n_lens_subpixels=n_lens_subpixels,
        permittivity_pillar=permittivity
    )

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(permittivity_map, origin='lower',
                 extent=(0., total_lens_period / 1000, 0., total_lens_period / 1000))

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
    focal_plane_amps = shapes_to_amps_function(shapes)
    # focal_plane_amps = shapes_to_amps_function(jnp.rot90(shapes, k=1))
    focusing_eff = calculate_focusing_efficiency(focal_plane_amps, basis_indices, (0.25, 0.25))
    print('Efficiency 1:', focusing_eff)
    focusing_eff = calculate_focusing_efficiency(focal_plane_amps, basis_indices, (0.75, 0.25))
    print('Efficiency 2:', focusing_eff)
    focusing_eff = calculate_focusing_efficiency(focal_plane_amps, basis_indices, (0.75, 0.75))
    print('Efficiency 3:', focusing_eff)
    focusing_eff = calculate_focusing_efficiency(focal_plane_amps, basis_indices, (0.25, 0.75))
    print('Efficiency 4:', focusing_eff)
    intensity_map = intensity_map_from_fourier_amplitudes(focal_plane_amps, basis_indices).T
    # intensity_map = jnp.rot90(intensity_map, k=-1)
    plot_amplitude_map(fig, ax[1], intensity_map,
                       wavelength_nm=wavelength, map_bounds=[0, total_lens_period / 1000], cbar=False)

    plt.show()


if __name__ == '__main__':
    # print(widths)
    evaluate_design()
