import jax.numpy as jnp

from fmmax import basis, fmm, fields, scattering
from lens_permittivity_profile_generator import generate_lens_permittivity_map

import matplotlib.pyplot as plt


symmetry_indices = jnp.array([
    [9, 8, 7, 6, 7, 8, 9],
    [8, 5, 3, 4, 3, 5, 8],
    [7, 3, 2, 1, 2, 3, 7],
    [6, 4, 1, 0, 1, 4, 6],
    [7, 3, 2, 1, 2, 3, 7],
    [8, 5, 3, 4, 3, 5, 8],
    [9, 8, 7, 6, 7, 8, 9]
], dtype=int)
n_unique_widths = 10
unique_widths = jnp.array([71, 211, 204, 154, 159, 102, 186, 106, 111, 43])

wavelength = 650
n_lens_subpixels = 7
lens_subpixel_size = 300
lens_thickness = 800
permittivity = 4
focal_length = 4000
approximate_number_of_terms = 500

field_samples_per_period = 300
thickness_above = focal_length + 1000
thickness_below = 1000


if __name__ == '__main__':
    widths = unique_widths[symmetry_indices]
    shapes = jnp.stack([widths] + [jnp.zeros_like(widths)] * 3, axis=-1)
    permittivity_pattern = generate_lens_permittivity_map(
        shapes=shapes,
        sub_pixel_size=lens_subpixel_size,
        n_lens_subpixels=n_lens_subpixels,
        permittivity_pillar=permittivity
    )
    # plt.imshow(permittivity_pattern)
    # plt.show()

    total_lens_period = n_lens_subpixels * lens_subpixel_size

    primitive_lattice_vectors = basis.LatticeVectors(
        u=total_lens_period * basis.X, v=total_lens_period * basis.Y
    )
    expansion = basis.generate_expansion(
        primitive_lattice_vectors=primitive_lattice_vectors,
        approximate_num_terms=approximate_number_of_terms,
        truncation=basis.Truncation.CIRCULAR,
    )

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
    zero_mode_index, = jnp.where(jnp.all(expansion.basis_coefficients == 0, axis=1))
    inc_fwd_amplitude = inc_fwd_amplitude.at[zero_mode_index].set(1.)
    fwd_amplitude = jnp.asarray(inc_fwd_amplitude[..., jnp.newaxis], dtype=float)

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
        layer_thicknesses=[thickness_above, lens_thickness, thickness_below]  # type: ignore[arg-type]
    )
    amplitudes_interior = fields.stack_amplitudes_interior(
        s_matrices_interior=s_matrices_interior,
        forward_amplitude_0_start=jnp.zeros_like(fwd_amplitude),
        backward_amplitude_N_end=fwd_amplitude,
    )

    x = jnp.linspace(0, total_lens_period, field_samples_per_period)
    y = jnp.ones_like(x) * total_lens_period / 2
    (ex, ey, ez), (hx, hy, hz), (x, y, z) = fields.stack_fields_3d_on_coordinates(
        amplitudes_interior=amplitudes_interior,
        layer_solve_results=[
            solve_result_ambient,
            solve_result_crystal,
            solve_result_ambient,
        ],
        layer_thicknesses=[thickness_above, lens_thickness, thickness_below],  # type: ignore[arg-type]
        layer_znum=[
            int(thickness_above * field_samples_per_period / total_lens_period),
            int(lens_thickness * field_samples_per_period / total_lens_period),
            int(thickness_below * field_samples_per_period / total_lens_period),
        ],
        x=x,
        y=y,
    )

    ex, ey, ez, hx, hy, hz = [
        jnp.squeeze(field) for field in (ex, ey, ez, hx, hy, hz)
    ]

    xplot, zplot = jnp.meshgrid(x, z, indexing='ij')

    field_plot = ey.real
    # plt.pcolormesh(xplot, zplot, field_plot, shading='nearest', cmap='bwr')

    fig, ax = plt.subplots(dpi=1000)
    ax.pcolormesh(xplot, zplot, jnp.abs(ey) ** 2, shading='nearest', cmap='hot')

    # plt.plot([0, 2100], [thickness_above, thickness_above], '--', color='white')
    # plt.plot([0, 2100], [thickness_above + lens_thickness, thickness_above + lens_thickness], '--', color='white')

    pillar_centers_x = jnp.linspace(
        -total_lens_period / 2, total_lens_period / 2,
        n_lens_subpixels, endpoint=False
    ) + lens_subpixel_size / 2 + total_lens_period / 2
    print(pillar_centers_x)
    for i in range(n_lens_subpixels):
        width = widths[3, i]
        center_x = pillar_centers_x[i]
        ax.plot(
            [center_x - width/2, center_x + width/2, center_x + width/2, center_x - width/2, center_x - width/2],
            [thickness_above] * 2 +  [thickness_above + lens_thickness] * 2  + [thickness_above],
            color='white'
        )

    plt.axis("equal")
    plt.axis("off")
    # plt.savefig("xz-intensity.png", bbox_inches='tight')
    plt.show()
