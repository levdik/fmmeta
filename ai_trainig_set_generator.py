import jax
import jax.numpy as jnp
import numpy as np

from fmmax import basis, fmm, fields, scattering
from lens_permittivity_profile_generator import generate_lens_permittivity_map

import concurrent.futures
import multiprocessing
import json
import os
import time


def generate_monochromatic_lens_symmetry_indices(
        n_lens_subpixels,
        relative_focal_point_position=(0.5, 0.5),
        tolerance_decimals=6
):
    single_coordinate_samples = np.linspace(0, 1, n_lens_subpixels)
    x_mesh, y_mesh = np.meshgrid(single_coordinate_samples, single_coordinate_samples)
    x_distances = np.abs(x_mesh - relative_focal_point_position[0])
    y_distances = np.abs(y_mesh - relative_focal_point_position[1])
    distances = x_distances ** 2 + y_distances ** 2
    unique_values, symmetry_indices = jnp.unique(np.round(distances, tolerance_decimals), return_inverse=True)
    return len(unique_values), symmetry_indices


def prepare_amplitude_generating_function(
        wavelength,
        permittivity,
        lens_subpixel_size,
        n_lens_subpixels,
        lens_thickness,
        approximate_number_of_terms
):
    total_lens_period = n_lens_subpixels * lens_subpixel_size
    n_unique_widths, symmetry_indices = generate_monochromatic_lens_symmetry_indices(n_lens_subpixels)

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

    def trans_ref_fourier_amplitudes(unique_widths):
        widths = unique_widths[symmetry_indices]
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
        return trans_e_y[:n_propagating_waves].flatten(), ref_e_y[:n_propagating_waves].flatten()

    jit_trans_ref_fourier_amplitudes = jax.jit(trans_ref_fourier_amplitudes)
    return jit_trans_ref_fourier_amplitudes, n_unique_widths


wavelength = 650
permittivity = 4
lens_subpixel_size = 300
n_lens_subpixels = 7
lens_thickness = 500
approximate_number_of_terms = 300

f, n = prepare_amplitude_generating_function(
    wavelength=wavelength,
    permittivity=permittivity,
    lens_subpixel_size=lens_subpixel_size,
    n_lens_subpixels=n_lens_subpixels,
    lens_thickness=lens_thickness,
    approximate_number_of_terms=approximate_number_of_terms
)


def writer(queue, filename):
    with open(filename, 'a') as f:
        while True:
            item = queue.get()
            if item is None:
                break
            json.dump(item, f)
            f.write('\n')
            f.flush()


def run_and_return():
    rand_key = jax.random.key(hash(time.time()))
    unique_widths = lens_subpixel_size * jax.random.uniform(rand_key, shape=(n,))
    trans, ref = f(unique_widths)
    trans_real = [float(x) for x in trans.real]
    trans_imag = [float(x) for x in trans.imag]
    ref_real = [float(x) for x in ref.real]
    ref_imag = [float(x) for x in ref.imag]
    result = {
        "unique_widths": [float(x) for x in unique_widths],
        "trans_real": trans_real,
        "trans_imag": trans_imag,
        "ref_real": ref_real,
        "ref_imag": ref_imag
    }
    return result


if __name__ == "__main__":
    filename = 'results.jsonl'  # Use JSONL format (one JSON object per line)
    manager = multiprocessing.Manager()
    queue = manager.Queue()

    # Start the writer process
    writer_process = multiprocessing.Process(target=writer, args=(queue, filename))
    writer_process.start()

    max_workers = os.cpu_count()
    print('Max Workers:', max_workers)

    n_examples = 10000

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_and_return) for _ in range(n_examples)]

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    print(result)
                    queue.put(result)
                except Exception as e:
                    print(f"Task failed: {e}")

    finally:
        queue.put(None)
        writer_process.join()
