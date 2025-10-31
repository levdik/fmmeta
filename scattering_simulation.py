import jax
import jax.numpy as jnp
import numpy as np
from fmmax import basis, fmm, fields, scattering

import refractiveindex2 as ri

SIO2 = ri.RefractiveIndexMaterial(shelf="main", book="SiO2", page="Malitson")
SI3N4 = ri.RefractiveIndexMaterial(shelf="main", book="Si3N4", page="Luke")


def prepare_lens_scattering_solver(
        wavelength: float,
        period: float,
        lens_thickness: float,
        substrate_thickness: float,
        approximate_number_of_terms: int,
        propagate_by_distance: float = 0.
):
    lens_permittivity = SI3N4.get_refractive_index(wavelength_um=wavelength / 1000) ** 2
    permittivity_substrate = SIO2.get_refractive_index(wavelength_um=wavelength / 1000) ** 2

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

    def lens_pattern_to_scattered_amps_func(lens_pattern: jnp.ndarray) -> jnp.ndarray:
        lens_permittivity_pattern = lens_pattern * (lens_permittivity - 1.) + 1.
        solve_result_crystal = fmm.eigensolve_isotropic_media(
            permittivity=lens_permittivity_pattern,
            wavelength=jnp.asarray(wavelength),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            expansion=expansion,
            formulation=fmm.Formulation.FFT
        )
        s_matrices_interior = scattering.stack_s_matrices_interior(
            layer_solve_results=[
                solve_result_ambient, solve_result_substrate, solve_result_crystal, solve_result_ambient],
            layer_thicknesses=[0., substrate_thickness, lens_thickness, 0.]  # type: ignore[arg-type]
        )
        amplitudes_interior = fields.stack_amplitudes_interior(
            s_matrices_interior=s_matrices_interior,
            forward_amplitude_0_start=jnp.zeros_like(fwd_amplitude),
            backward_amplitude_N_end=fwd_amplitude,
        )
        _, trans_amps = amplitudes_interior[0]

        if propagate_by_distance != 0:
            trans_amps = fields.propagate_amplitude(
                amplitude=trans_amps,
                distance=propagate_by_distance,  # type: ignore[arg-type]
                layer_solve_result=solve_result_ambient
            )

        (_, trans_e_y, _), _ = fields.fields_from_wave_amplitudes(
            forward_amplitude=trans_amps,
            backward_amplitude=jnp.zeros_like(trans_amps),
            layer_solve_result=solve_result_ambient
        )
        propagating_trans_e_y_amps = trans_e_y[:n_propagating_waves].flatten()
        return propagating_trans_e_y_amps

    return lens_pattern_to_scattered_amps_func, expansion.basis_coefficients[:n_propagating_waves]


def simulate_field_cross_section_along_propagation():
    # TODO
    pass
