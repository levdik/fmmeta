import jax
import jax.numpy as jnp

from scattering_solver_factory import prepare_lens_pixel_width_to_scattered_amplitudes_function

import matplotlib.pyplot as plt
from lens_permittivity_profile_generator import generate_lens_permittivity_map


def visualize_random_lenses(n_pillars_per_side, subplots_shape, rng_seed):
    fig, ax = plt.subplots(*subplots_shape)
    n = subplots_shape[0] * subplots_shape[1]
    rng_key = jax.random.key(rng_seed)
    for ax_i in ax.flatten():
        rng_key, rng_subkey = jax.random.split(rng_key)
        widths = generate_single_lens_random_widths(n_pillars_per_side, rng_subkey)
        shapes = jnp.stack([widths] + [jnp.zeros_like(widths)] * 3, axis=-1)
        permittivity_map = generate_lens_permittivity_map(
            shapes, 1, n_pillars_per_side, 4.)
        ax_i.imshow(permittivity_map, origin='lower')
        ax_i.xaxis.set_visible(False)
        ax_i.yaxis.set_visible(False)
    plt.show()


def generate_single_lens_random_widths(n_pillars_per_side, rng_key):
    n_pillars = n_pillars_per_side ** 2
    # (
    #     rng_key,
    #     rng_subkey_n, rng_subkey_indices, rng_subkey_widths_level,
    #     rng_subkey_widths_variation_magnitude, rng_subkey_widths_variation,
    #     rng_subkey_shuffle,
    # )  = jax.random.split(rng_key, num=7)
    (
        rng_key,
        rng_subkey_n, rng_subkey_indices,
        rng_subkey_widths, rng_subkey_shuffle,
    )  = jax.random.split(rng_key, num=5)
    n_unique_widths = int(jax.random.randint(key=rng_subkey_n, shape=(), minval=1, maxval=n_pillars + 1))
    symmetry_indices = jnp.concatenate([
        jnp.arange(n_unique_widths),
        jax.random.randint(key=rng_subkey_indices, shape=(n_pillars - n_unique_widths,), minval=0, maxval=n_unique_widths)
    ])
    symmetry_indices = jax.random.permutation(key=rng_subkey_shuffle, x=symmetry_indices)

    # unique_widths_level = jax.random.uniform(rng_subkey_widths_level, maxval=0.95)
    # unique_widths_variation = jax.random.uniform(rng_subkey_widths_variation, shape=(n_unique_widths,), minval=-1, maxval=1)
    # unique_widths_variation *= jax.random.uniform(rng_subkey_widths_variation_magnitude)
    # remains_to_zero = unique_widths_level
    # remains_to_one = 1. - unique_widths_variation
    # multiplier = remains_to_zero * (unique_widths_variation < 0) + remains_to_one * (unique_widths_variation > 0)
    # unique_widths = unique_widths_level + multiplier * unique_widths_variation

    unique_widths = jax.random.uniform(key=rng_subkey_widths, shape=(n_unique_widths,))
    widths = unique_widths[symmetry_indices]
    return widths


def calculate_and_save_training_set(
        wavelength, n_lens_subpixels, lens_subpixel_size, lens_thickness,
        save_filename, batch_size, n_batches, rng_key
):
    permittivity = 4
    approximate_number_of_terms = 1000

    width_to_amps_f = prepare_lens_pixel_width_to_scattered_amplitudes_function(
        wavelength=wavelength,
        permittivity=permittivity,
        lens_subpixel_size=lens_subpixel_size,
        n_lens_subpixels=n_lens_subpixels,
        lens_thickness=lens_thickness,
        approximate_number_of_terms=approximate_number_of_terms
    )
    empty_lens_a00 = -jnp.exp(1j * 2 * jnp.pi * lens_thickness / wavelength)
    map_f = jax.jit(jax.vmap(width_to_amps_f))

    widths_batches = []
    amplitude_batches = []
    for batch_i in range(n_batches):
        print(batch_i)
        rng_key, rng_batch_key = jax.random.split(rng_key)
        # random_widths = jax.random.uniform(
        #     rng_subkey_widths,
        #     shape=(batch_size, n_lens_subpixels ** 2),
        #     minval=0, maxval=1
        # )
        # random_widths *= lens_subpixel_size
        random_widths = []
        for sample_i in range(batch_size):
            rng_batch_key, rng_subkey = jax.random.split(rng_batch_key)
            random_widths.append(generate_single_lens_random_widths(n_lens_subpixels, rng_subkey))
        random_widths = jnp.stack(random_widths) * lens_subpixel_size

        amps = map_f(random_widths)
        amps = amps.at[:, :amps.shape[-1] // 2].divide(empty_lens_a00)

        amplitude_batches.append(amps)
        widths_batches.append(random_widths)

    all_widths = jnp.vstack(widths_batches)
    all_amplitudes = jnp.vstack(amplitude_batches)
    # print(all_amplitudes)
    # print(jnp.sum(jnp.abs(all_amplitudes) ** 2, axis=-1))
    # print(all_amplitudes.shape)
    # print(all_widths)
    # print(all_widths.shape)
    jnp.savez(save_filename, amps=all_amplitudes, widths=all_widths)


if __name__ == "__main__":
    jnp.set_printoptions(linewidth=1000)

    calculate_and_save_training_set(
        wavelength=650,
        n_lens_subpixels=4,
        lens_subpixel_size=300,
        lens_thickness=800,
        save_filename='ai_training_data/test.npz',
        batch_size=5,
        n_batches=2,
        rng_key=jax.random.key(0)
    )

    # for i in range(10):
    #     visualize_random_lenses(4, (3, 3), rng_seed=i)
