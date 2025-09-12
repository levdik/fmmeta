import jax.numpy as jnp
import numpy as np
from fmmax import basis, fmm, fields, scattering


def generate_wave_permittivity_basis_indices(r):
    n_max = int(np.floor(r))

    basis_indices = []
    for n in range(-n_max, n_max + 1):
        for m in range(-n_max, n_max + 1):
            if n ** 2 + m ** 2 <= r ** 2:
                basis_indices.append((n, m))
    return basis_indices


def generate_wave_permittivity_primary_basis_indices(r, symmetry_type='central'):
    full_basis_indices = generate_wave_permittivity_basis_indices(r)
    symmetry_indices = np.zeros(len(full_basis_indices), dtype=int)
    primary_basis = []

    if symmetry_type == 'central':
        for i, (n, m) in enumerate(full_basis_indices):
            if n >= m >= 0:
                primary_basis.append((n, m))
                symmetry_indices[i] = len(primary_basis) - 1

        for i, (n, m) in enumerate(full_basis_indices):
            if not (n >= m >= 0):
                for i1, (n1, m1) in enumerate(primary_basis):
                    if (abs(n1) == abs(n) and abs(m1) == abs(m)) or (abs(n1) == abs(m) and abs(m1) == abs(n)):
                        symmetry_indices[i] = i1
                        break
    elif symmetry_type == 'main_diagonal':
        symmetry_indices = np.zeros(len(full_basis_indices), dtype=int)
        primary_basis = []
        for i, (n, m) in enumerate(full_basis_indices):
            if n >= m:
                primary_basis.append((n, m))
                symmetry_indices[i] = len(primary_basis) - 1

        for i, (n, m) in enumerate(full_basis_indices):
            if n < m:
                for i1, (n1, m1) in enumerate(primary_basis):
                    if n1 == m and m1 == n:
                        symmetry_indices[i] = i1
                        break
    else:
        raise ValueError('Unknown symmetry type')

    full_basis_indices = jnp.array(full_basis_indices)
    primary_basis = jnp.array(primary_basis)

    return full_basis_indices, primary_basis, symmetry_indices


def copy_amps_to_new_basis(amps, old_basis, new_basis):
    new_amps = np.zeros(len(new_basis), dtype=amps.dtype)
    for i, (n, m) in enumerate(new_basis):
        for i1, (n1, m1) in enumerate(old_basis):
            if n1 == n and m1 == m:
                new_amps[i] = amps[i1]
                break
    return jnp.array(new_amps)


def generate_wave_permittivity_pattern(
        amplitudes, basis_indices, permittivity, permittivity_ambience=1., resolution=100
):
    single_coordinate_samples = jnp.linspace(0, 1, resolution, endpoint=False)
    x, y = jnp.meshgrid(single_coordinate_samples, single_coordinate_samples)

    # wave_values = jnp.zeros_like(x, dtype=complex)
    # for a, (n, m) in zip(amplitudes, basis_indices):
    #     wave_values += a * jnp.exp(1j * 2 * jnp.pi * (n * x + m * y))

    n, m = basis_indices.T
    phase = n[None, None, :] * x[:, :, None] + m[None, None, :] * y[:, :, None]
    term = amplitudes[None, None, :] * jnp.exp(1j * 2 * jnp.pi * phase)

    wave_values = jnp.sum(term, axis=-1)

    wave_values = wave_values.real

    cell_corner_values = jnp.stack([
        wave_values,
        jnp.roll(wave_values, -1, axis=0),
        jnp.roll(wave_values, -1, axis=1),
        jnp.roll(wave_values, -1, axis=(0, 1))
    ])
    max_corner_value = jnp.max(cell_corner_values, axis=0)
    min_corner_value = jnp.min(cell_corner_values, axis=0)
    filling_map = max_corner_value / (max_corner_value - min_corner_value)
    # nan_indices = jnp.where(jnp.isnan(filling_map))
    # filling_map = filling_map.at[nan_indices].set(jnp.sign(max_corner_value)[nan_indices])
    filling_map = jnp.where(jnp.isnan(filling_map), jnp.sign(max_corner_value), filling_map)
    filling_map = jnp.clip(filling_map, 0, 1)

    permittivity_pattern = filling_map * (permittivity - permittivity_ambience) + permittivity_ambience
    return permittivity_pattern


def prepare_wave_lens_scattering_simulating_function(
        wavelength, thickness, period, approx_n_terms, propagate_by_distance=0.
):
    primitive_lattice_vectors = basis.LatticeVectors(
        u=period * basis.X, v=period * basis.Y
    )
    expansion = basis.generate_expansion(
        primitive_lattice_vectors=primitive_lattice_vectors,
        approximate_num_terms=approx_n_terms,
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
    inc_fwd_amplitude = jnp.zeros(2 * len(expansion.basis_coefficients))
    zero_mode_index, = jnp.where(np.all(expansion.basis_coefficients == 0, axis=1))
    inc_fwd_amplitude = inc_fwd_amplitude.at[zero_mode_index].set(1.)
    fwd_amplitude = jnp.asarray(inc_fwd_amplitude[..., np.newaxis], dtype=jnp.float32)

    def func(permittivity_pattern):
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
            layer_thicknesses=[0., thickness, 0.]  # type: ignore[arg-type]
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
        amplitudes_to_return = trans_e_y[:n_propagating_waves].flatten()

        return amplitudes_to_return

    return func, expansion.basis_coefficients[:n_propagating_waves]
