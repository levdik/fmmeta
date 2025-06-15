import jax
import jax.numpy as jnp

from ai_model_forward import SquarePixelLensScatteringModel
from scattering_solver_factory import prepare_lens_pixel_width_to_scattered_amplitudes_function

import matplotlib.pyplot as plt


if __name__ == "__main__":
    jnp.set_printoptions(linewidth=10000)

    wavelength=650
    permittivity=4
    n_pillars = 7
    lens_subpixel_size=300
    lens_thickness=500

    model = SquarePixelLensScatteringModel.load(
        filename='ai_models/red_7x7_forward.pkl',
        n_propagating_waves=37,
        n_lens_params=n_pillars ** 2,
        hidden_layer_dims=[1024] * 3
    )

    true_widths_to_amps_f = prepare_lens_pixel_width_to_scattered_amplitudes_function(
        wavelength=wavelength,
        permittivity=permittivity,
        lens_subpixel_size=lens_subpixel_size,
        n_lens_subpixels=n_pillars,
        lens_thickness=lens_thickness,
        approximate_number_of_terms=300
    )
    reference_a00 = -jnp.exp(1j * 2 * jnp.pi * lens_thickness / wavelength)

    # widths = jnp.zeros((n_pillars, n_pillars))
    # widths = 200 * jnp.ones((n_pillars, n_pillars))
    widths = 300 * jax.random.uniform(key=jax.random.key(0), shape=(n_pillars ** 2,))

    predicted_amps = jnp.squeeze(model(widths.reshape(1, -1)))
    true_amps = true_widths_to_amps_f(widths)
    true_amps = true_amps.at[:len(true_amps) // 2].divide(reference_a00)
    true_amps /= jnp.linalg.norm(true_amps)

    plt.plot(jnp.abs(predicted_amps), 'o', label='Predicted')
    plt.plot(jnp.abs(true_amps), '--', label='True')
    plt.legend()
    plt.grid()
    plt.show()

    # fig, ax = plt.subplots(1, 2)
    # ax[0].plot(true_amps.real, '--', label='True')
    # ax[0].plot(predicted_amps.real, 'o', label='Predicted')
    # ax[0].legend()
    # ax[0].set_title('Imag')
    # ax[1].plot(true_amps.imag, '--', label='True')
    # ax[1].plot(predicted_amps.imag, 'o', label='Predicted')
    # ax[1].legend()
    # ax[1].set_title('Imag')
    # plt.show()

    print(predicted_amps)
    print(true_amps)
    print(jnp.linalg.norm(true_amps - predicted_amps))
