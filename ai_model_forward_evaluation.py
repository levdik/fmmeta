import jax
import jax.numpy as jnp

# from ai_model_forward import SquarePixelLensScatteringModel
from ai_model_forward_no_ref import SquarePixelLensScatteringModel
from scattering_solver_factory import prepare_lens_pixel_width_to_scattered_amplitudes_function
from field_postprocessing import calculate_focusing_efficiency
from fmmax import basis, fields, fmm

import matplotlib.pyplot as plt


def compare_trained_model_validation_losses():
    max_width = 300.
    batch_size = 500

    incidence_reference_amplitude = -1.20536745e-01 + 9.9270886e-01j  # -exp(1j * k0 * thickness)
    data = jnp.load('ai_training_data/red_7x7_th500_p300_120k.npz')

    widths = jnp.array(data['widths'])
    widths = widths.reshape(widths.shape[0], -1)
    amplitudes = jnp.array(data['amps'])

    n_propagating_waves = amplitudes.shape[-1] // 2
    amplitudes = amplitudes[:, :n_propagating_waves]

    widths /= max_width
    amplitudes /= incidence_reference_amplitude


    approximate_test_to_train_ratio = 0.1
    total_n_samples = widths.shape[0]
    n_batches = int(((1 - approximate_test_to_train_ratio) * total_n_samples) // batch_size)
    n_training_samples = n_batches * batch_size
    validation_widths = widths[n_training_samples:]
    validation_amplitudes = amplitudes[n_training_samples:]

    def validate_loss(model, x, y):
        y_pred = model(x)
        loss_per_sample = jnp.sum(jnp.abs(y_pred - y) ** 2, axis=-1)
        return jnp.mean(loss_per_sample)

    for n in range(3, 7):
        for d in [128, 256, 512, 1024, 2048]:
            model = SquarePixelLensScatteringModel.load(
                filename=f'ai_models/red_7x7_forward_no_ref_[{d}]x{n}.pkl',
                n_propagating_waves=37,
                n_lens_params=7 ** 2,
                hidden_layer_dims=[d] * n,
            )
            validation_loss = validate_loss(model, validation_widths, validation_amplitudes)
            print(d, n, validation_loss, sep='\t')


if __name__ == "__main__":
    jnp.set_printoptions(linewidth=10000)

    # compare_trained_model_validation_losses()
    # exit()

    # wavelength=650
    # permittivity=4
    # n_pillars = 4
    # lens_subpixel_size=300
    # lens_thickness=800
    # model = SquarePixelLensScatteringModel.load(
    #     filename='ai_models/red_4x4_forward',
    #     n_propagating_waves=9,
    #     n_lens_params=n_pillars ** 2,
    #     hidden_layer_dims=[32, 32, 32]
    # )

    wavelength = 650
    permittivity = 4
    n_pillars = 7
    lens_subpixel_size = 300
    lens_thickness = 500
    model = SquarePixelLensScatteringModel.load(
        filename='ai_models/red_7x7_forward_no_ref_[256]x6_v2.pkl',
        n_propagating_waves=37,
        n_lens_params=n_pillars ** 2,
        hidden_layer_dims=[256] * 6,
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
    widths = 150 * jnp.ones((n_pillars, n_pillars))
    # widths = 300 * jax.random.uniform(key=jax.random.key(0), shape=(n_pillars ** 2,))
    # widths = jnp.array([
    #     [90,  90,  160, 200, 160, 90,  90],
    #     [90,  85,  160, 195, 160, 85,  90],
    #     [160, 160, 220, 235, 220, 160, 160],
    #     [200, 195, 235, 260, 235, 195, 200],
    #     [160, 160, 220, 235, 220, 160, 160],
    #     [90,  85,  160, 195, 160, 85,  90],
    #     [90,  90,  160, 200, 160, 90,  90]
    # ])

    predicted_amps = jnp.squeeze(model(widths.reshape(1, -1) / lens_subpixel_size))
    true_amps = true_widths_to_amps_f(widths)
    # true_amps = true_amps.at[:len(true_amps) // 2].divide(reference_a00)
    true_amps = true_amps[:len(true_amps) // 2] / reference_a00
    true_amps /= jnp.linalg.norm(true_amps)

    propagating_basis_indices = basis.generate_expansion(
        basis.LatticeVectors(basis.X, basis.Y), model.n_propagating_waves).basis_coefficients
    total_lens_period = n_pillars * lens_subpixel_size
    K = 2 * jnp.pi / total_lens_period
    kx, ky = propagating_basis_indices.T * K
    k0 = 2 * jnp.pi / wavelength
    kz = jnp.sqrt(k0 ** 2 - kx ** 2 - ky ** 2)
    focal_length = 4000
    true_amps_propagated = true_amps[:model.n_propagating_waves] * jnp.exp(1j * kz * focal_length)
    predicted_amps_propagated = predicted_amps[:model.n_propagating_waves] * jnp.exp(1j * kz * focal_length)
    true_efficiency = calculate_focusing_efficiency(true_amps_propagated[:model.n_propagating_waves], propagating_basis_indices)
    predicted_efficiency = calculate_focusing_efficiency(predicted_amps_propagated[:model.n_propagating_waves], propagating_basis_indices)
    print(true_efficiency, predicted_efficiency)

    plt.plot(jnp.abs(predicted_amps), 'o', label='Predicted')
    plt.plot(jnp.abs(true_amps), '--', label='True')
    plt.title('Predicted and True amplitudes for random 7x7 lens')
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(jnp.angle(predicted_amps), 'o', label='Predicted')
    plt.plot(jnp.angle(true_amps), '--', label='True')
    plt.title('Predicted and True phases for random 7x7 lens')
    plt.legend()
    plt.grid()
    plt.show()

    print(predicted_amps)
    print(true_amps)
    print(jnp.linalg.norm(true_amps - predicted_amps))

    for tr, pr in zip(jnp.abs(true_amps), jnp.abs(predicted_amps)):
        print(tr, pr, '\\\\', sep='\t')
