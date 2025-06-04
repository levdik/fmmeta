import jax.numpy as jnp
import matplotlib.pyplot as plt

import optax
from flax import nnx

import pickle


class SquarePixelLensOptimizingModelWithAutoencodingParams(nnx.Module):
    def __init__(
            self,
            n_propagating_waves: int,
            n_lens_params: int,
            hidden_layer_dims: list[int],
            n_autoencoding_neurons: int,
            rngs: nnx.Rngs
    ):
        input_dim = 4 * n_propagating_waves + n_autoencoding_neurons
        layer_dims = [input_dim] + hidden_layer_dims + [n_lens_params]
        layers = []

        for i in range(len(layer_dims) - 1):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]
            linear = nnx.Linear(in_dim, out_dim, rngs=rngs)
            layers.append(linear)

        self.n_autoencoding_neurons = n_autoencoding_neurons
        self.n_lens_params = n_lens_params
        self.layers = layers

    def __call__(self, x: jnp.ndarray, autoencoding_params: jnp.array) -> jnp.ndarray:
        x = jnp.hstack([autoencoding_params, x.real, x.imag])
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = nnx.leaky_relu(x)
            else:
                x = nnx.sigmoid(x)
        return x


class SquarePixelLensWidthAutoencodingModel(nnx.Module):
    def __init__(self, amps_to_widths_model: SquarePixelLensOptimizingModelWithAutoencodingParams, rngs: nnx.Rngs):
        self.amps_to_widths_model = amps_to_widths_model
        n_lens_params = amps_to_widths_model.n_lens_params
        n_autoencoding_neurons = amps_to_widths_model.n_autoencoding_neurons
        self.autoencoding_layer = nnx.Linear(n_lens_params, n_autoencoding_neurons, rngs=rngs)

    def __call__(self, widths: jnp.ndarray, corresponding_amplitudes: jnp.array) -> jnp.ndarray:
        autoencoding_params = self.autoencoding_layer(widths)
        return self.amps_to_widths_model(corresponding_amplitudes, autoencoding_params)

    def save(self, filename):
        _, state = nnx.split(self)
        with open(filename, 'wb') as f:
            pickle.dump(state, f)


def train_2x2():
    lens_subpixel_size = 300
    n_lens_params = 2 ** 2

    from ai_training_data.trans_ref_2x2 import amps, widths
    n_training_samples = 4000
    n_validation_samples = 1000
    amps = amps[:n_training_samples]
    widths = widths[:n_training_samples]
    # print(amps.shape)
    # print(widths.shape)
    # exit()
    normalized_widths = widths / lens_subpixel_size

    # @nnx.jit
    def train_step(autoencoding_model, optimizer, widths, amps):
        def loss_fn(autoencoding_model):
            widths_pred = autoencoding_model(widths, amps)
            return jnp.mean((widths_pred - widths) ** 2)

        loss, grads = nnx.value_and_grad(loss_fn)(autoencoding_model)
        optimizer.update(grads)
        return loss

    n_autoencoding_neurons = 1

    model = SquarePixelLensOptimizingModelWithAutoencodingParams(
        n_propagating_waves=1,
        n_lens_params=n_lens_params,
        hidden_layer_dims=[64] * 3,
        n_autoencoding_neurons=n_autoencoding_neurons,
        rngs=nnx.Rngs(0),
    )
    width_autoencoding_model = SquarePixelLensWidthAutoencodingModel(
        amps_to_widths_model=model,
        rngs=nnx.Rngs(1),
    )

    learning_rate = 1e-2
    n_epochs = 5000
    batch_size = 250
    n_batches_in_epoch = n_training_samples // batch_size

    optimizer = nnx.Optimizer(width_autoencoding_model, optax.adam(learning_rate))
    current_loss = 1.

    for epoch in range(n_epochs):
        for i in range(n_batches_in_epoch):
            batch_start = i * batch_size
            batch_end = (i + 1) * batch_size
            batch_amps = amps[batch_start:batch_end]
            batch_widths = normalized_widths[batch_start:batch_end]
            current_loss = train_step(width_autoencoding_model, optimizer, batch_widths, batch_amps)
        print(epoch, current_loss, sep='\t')

    width_autoencoding_model.save('ai_models/red_2x2_autoencoding')


if __name__ == '__main__':
    train_2x2()
