import jax
import jax.numpy as jnp

from scattering_solver_factory import prepare_lens_pixel_width_to_scattered_amplitudes_function


def calculate_and_save_training_set(
        wavelength, n_lens_subpixels, lens_subpixel_size, lens_thickness,
        save_filename, batch_size, n_batches,
):
    permittivity = 4
    approximate_number_of_terms = 500

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
    rng_key = jax.random.key(hash(save_filename))

    widths_batches = []
    amplitude_batches = []
    for i in range(n_batches):
        print(i)
        rng_key, rng_subkey_widths = jax.random.split(rng_key)
        random_widths = jax.random.uniform(
            rng_subkey_widths,
            shape=(batch_size, n_lens_subpixels ** 2),
            minval=0, maxval=1
        )
        random_widths *= lens_subpixel_size

        amps = map_f(random_widths)
        amps = amps.at[:, :amps.shape[-1] // 2].divide(empty_lens_a00)

        amplitude_batches.append(amps)
        widths_batches.append(random_widths)

    all_widths = jnp.vstack(widths_batches)
    all_amplitudes = jnp.vstack(amplitude_batches)
    jnp.savez(save_filename, amps=all_amplitudes, widths=all_widths)


if __name__ == "__main__":
    calculate_and_save_training_set(
        wavelength=650,
        n_lens_subpixels=8,
        lens_subpixel_size=300,
        lens_thickness=1000,
        save_filename='ai_training_data/test.npz',
        batch_size=3,
        n_batches=2,
    )
