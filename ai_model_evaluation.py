import jax.numpy as jnp
from ai_model import SquarePixelLensOptimizingModel
from scattering_solver_factory import prepare_lens_pixel_width_to_scattered_amplitudes_function
from field_postprocessing import find_focusing_efficiency_eigenstates
from fmmax import fields


def test_model():
    wavelength = 650
    permittivity = 4
    lens_subpixel_size = 300
    n_lens_subpixels = 7
    lens_thickness = 500
    approximate_number_of_terms = 300

    max_width = lens_subpixel_size
    n_widths = n_lens_subpixels ** 2
    n_propagating = 37
    # hidden_layer_dims = [256, 256, 128]
    # model = SquarePixelLensOptimizingModel.load(
    #     'ai_models/red_7x7_p300_th500_med.pkl',
    #         n_propagating_waves=n_propagating,
    #         n_lens_params=n_widths,
    #         hidden_layer_dims=hidden_layer_dims
    # )
    hidden_layer_dims = [256] * 4
    model = SquarePixelLensOptimizingModel.load(
        'ai_models/red_7x7_p300_th500_online_inversion_256_256_256_256.pkl',
        n_propagating_waves=n_propagating,
        n_lens_params=n_widths,
        hidden_layer_dims=hidden_layer_dims
    )

    target_amps = jnp.zeros(2 * n_propagating, dtype=complex).at[1].set(1.)
    target_amps = target_amps / jnp.linalg.norm(target_amps)
    print(target_amps)
    predicted_widths = jnp.round(model(target_amps) * max_width).reshape(n_lens_subpixels, n_lens_subpixels)

    f = prepare_lens_pixel_width_to_scattered_amplitudes_function(
        wavelength=wavelength,
        permittivity=permittivity,
        lens_subpixel_size=lens_subpixel_size,
        n_lens_subpixels=n_lens_subpixels,
        lens_thickness=lens_thickness,
        approximate_number_of_terms=approximate_number_of_terms
    )
    actual_amps = f(predicted_widths.reshape(n_lens_subpixels, n_lens_subpixels))
    print(jnp.sum(jnp.abs(actual_amps[:n_propagating]) ** 2))
    print(jnp.sum(jnp.abs(actual_amps[n_propagating:]) ** 2))
    print(jnp.round(jnp.abs(actual_amps[:n_propagating]), 3))
    print(jnp.round(jnp.abs(actual_amps[n_propagating:]), 3))


if __name__ == '__main__':
    jnp.set_printoptions(linewidth=1000)

    test_model()
