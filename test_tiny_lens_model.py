import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from scattering_solver_factory import prepare_lens_pixel_width_to_scattered_amplitudes_function

import optax
from flax import nnx


class SquarePixelLensOptimizingModel(nnx.Module):
    def __init__(
            self,
            n_propagating_waves: int,
            n_lens_params: int,
            hidden_layer_dims: list[int],
            rngs: nnx.Rngs,
    ):
        input_dim = 4 * n_propagating_waves
        layer_dims = [input_dim] + hidden_layer_dims + [n_lens_params]
        layers = []

        for i in range(len(layer_dims) - 1):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]
            linear = nnx.Linear(in_dim, out_dim, rngs=rngs)
            layers.append(linear)

        self.layers = layers

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x = jnp.hstack([jnp.abs(x), jnp.angle(x) / (2 * jnp.pi) + 1.])
        x = jnp.hstack([x.real, x.imag])
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = nnx.leaky_relu(x)
            else:
                x = nnx.sigmoid(x)
        return x


def train_1x1():
    wavelength = 650
    permittivity = 4
    lens_subpixel_size = 300
    n_lens_subpixels = 1
    lens_thickness = 500
    approximate_number_of_terms = 100

    # widths_to_amps_func = prepare_lens_pixel_width_to_scattered_amplitudes_function(
    #     wavelength=wavelength,
    #     permittivity=permittivity,
    #     lens_subpixel_size=lens_subpixel_size,
    #     n_lens_subpixels=n_lens_subpixels,
    #     lens_thickness=lens_thickness,
    #     approximate_number_of_terms=approximate_number_of_terms
    # )
    # map_f = jax.vmap(widths_to_amps_func)

    # widths = jnp.linspace(0, lens_subpixel_size, 301)
    # widths = widths.reshape(-1, 1)
    # amps = map_f(widths)
    # print(amps)

    from ai_training_data.trans_ref_1x1 import widths
    from ai_training_data.trans_ref_1x1 import trans_ref_amps as amps
    amps /= amps[0, 0]

    # trans, ref = amps.T
    # # print(jnp.abs(trans) ** 2 + jnp.abs(ref) ** 2)
    # plt.plot(widths, jnp.abs(trans) ** 2, label='Transmission')
    # plt.plot(widths, jnp.abs(ref) ** 2, label='Reflection')
    # plt.xlabel('Pillar width, nm')
    # plt.ylabel('Power')
    # plt.legend()
    # plt.grid()
    # plt.show()
    #
    # plt.plot(widths, jnp.angle(trans), label='Transmission')
    # plt.plot(widths, jnp.angle(ref), label='Reflection')
    # plt.xlabel('Pillar width, nm')
    # plt.ylabel('Phase')
    # plt.legend()
    # plt.grid()
    # plt.show()
    # exit()

    @nnx.jit
    def train_step(model, optimizer, x, y):
        def loss_fn(model):
            y_pred = model(x)
            return jnp.mean((y_pred - y) ** 2)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)
        return loss

    model = SquarePixelLensOptimizingModel(
        n_propagating_waves=1,
        n_lens_params=1,
        hidden_layer_dims=[32, 32, 32],
        rngs=nnx.Rngs(0),
    )

    learning_rate = 1e-3
    n_epochs = 3000
    n_training_samples = len(widths)
    normalized_widths = widths / lens_subpixel_size
    # trans_phase = jnp.angle(amps[:, 0]) / (2 * jnp.pi) + 1.
    # args = trans_phase[:, jnp.newaxis]
    # args = amps[:, 0, jnp.newaxis]
    args = amps
    normalized_widths = normalized_widths[:, jnp.newaxis]

    optimizer = nnx.Optimizer(model, optax.adam(learning_rate))

    for epoch in range(n_epochs):
        current_loss = train_step(model, optimizer, args, normalized_widths)
        if epoch % 10 == 0:
            print(epoch, current_loss, sep='\t')

    pred_widths = model(args) * lens_subpixel_size
    pred_widths = jnp.squeeze(pred_widths)

    for w, pw in zip(widths, pred_widths):
        print(int(w), float(pw), '\\\\', sep='\t')

    plt.plot(widths, pred_widths, '.')
    plt.plot(widths, widths, '--')
    plt.xlabel('True width')
    plt.ylabel('Predicted width')
    plt.legend(['NN Data', 'y=x'])
    plt.show()


def train_2x2():
    wavelength = 650
    permittivity = 4
    lens_subpixel_size = 300
    n_lens_subpixels = 2
    lens_thickness = 500
    approximate_number_of_terms = 213

    # unique_n_terms = [
    #     1, 5, 9, 13, 21, 29, 37, 45, 57, 69, 81, 89, 97,
    #     101, 109, 121, 129, 145, 161, 169, 185, 193, 197,
    #     213, 221, 233, 249, 253, 261, 277, 285
    # ]
    # for approximate_number_of_terms in unique_n_terms:

    # widths_to_amps_func = prepare_lens_pixel_width_to_scattered_amplitudes_function(
    #     wavelength=wavelength,
    #     permittivity=permittivity,
    #     lens_subpixel_size=lens_subpixel_size,
    #     n_lens_subpixels=n_lens_subpixels,
    #     lens_thickness=lens_thickness,
    #     approximate_number_of_terms=approximate_number_of_terms
    # )
    # map_f = jax.vmap(widths_to_amps_func)
    # batch_size = 25
    # n_batches = 4000 // batch_size
    # widths = lens_subpixel_size * jax.random.uniform(jax.random.key(0), (batch_size * n_batches, 4))
    # amps = []
    # for i in range(n_batches):
    #     print(i)
    #     batch_start_index = i * batch_size
    #     batch_end_index = batch_start_index + batch_size
    #     batch_amps = map_f(widths[batch_start_index:batch_end_index])
    #     amps.append(batch_amps)
    # amps = jnp.vstack(amps)
    # print('widths = jnp.array([')
    # for w in widths:
    #     print('\t[', end='')
    #     print(*w, sep=', ', end='')
    #     print('],')
    # print('])')
    #
    # print()
    #
    # print('amps = jnp.array([')
    # for a in amps:
    #     print('\t[', end='')
    #     print(*a, sep=', ', end='')
    #     print('],')
    # print('])')

    from ai_training_data.trans_ref_2x2 import amps, widths
    # print(widths[jnp.argmin(jnp.linalg.norm(widths, axis=-1))])
    amps = amps[:1000]
    widths = widths[:1000]
    normalized_widths = widths / lens_subpixel_size

    @nnx.jit
    def train_step(model, optimizer, x, y):
        def loss_fn(model):
            y_pred = model(x)
            return jnp.mean((y_pred - y) ** 2)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)
        return loss

    model = SquarePixelLensOptimizingModel(
        n_propagating_waves=1,
        n_lens_params=4,
        hidden_layer_dims=[32, 32, 32],
        rngs=nnx.Rngs(0),
    )

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    width_index_for_testing = 2
    single_predicted_width_before_training = model(amps)[:, width_index_for_testing]
    single_width_true = normalized_widths[:, width_index_for_testing]
    ax[0].plot(single_width_true, single_predicted_width_before_training, '.')
    ax[0].plot([0, 1], [0, 1], '--')
    ax[0].set_title('Before training')
    ax[0].set_xlabel('True')
    ax[0].set_ylabel('Predicted')
    ax[0].legend(['NN Data', 'y=x'])

    learning_rate = 1e-3
    n_epochs = 500
    n_training_samples = len(widths)
    batch_size = 500
    n_batches_in_epoch = n_training_samples // batch_size
    args = amps
    normalized_widths = normalized_widths[:, jnp.newaxis]

    optimizer = nnx.Optimizer(model, optax.adam(learning_rate))

    for epoch in range(n_epochs):
        for i in range(n_batches_in_epoch):
            batch_start = i * batch_size
            batch_end = (i + 1) * batch_size
            current_loss = train_step(
                model, optimizer, args[batch_start:batch_end], normalized_widths[batch_start:batch_end])
        print(epoch, current_loss, sep='\t')

    print('NN Data before training')
    for w, pw in zip(single_width_true, single_predicted_width_before_training):
        print(round(float(w), 5), round(float(pw), 5), '\\\\', sep='\t')

    print('NN Data after training')
    single_predicted_width_after_training = model(amps)[:, width_index_for_testing]
    for w, pw in zip(single_width_true, single_predicted_width_after_training):
        print(round(float(w), 5), round(float(pw), 5), '\\\\', sep='\t')

    ax[1].plot(single_width_true, single_predicted_width_after_training, '.')
    ax[1].plot([0, 1], [0, 1], '--')
    ax[1].set_title('After training')
    ax[1].set_xlabel('True')
    ax[1].set_ylabel('Predicted')
    ax[1].legend(['NN Data', 'y=x'])

    plt.suptitle('Single NN output values (width of one of the pillars)')

    plt.show()


if __name__ == '__main__':
    # train_1x1()
    train_2x2()
