import jax.numpy as jnp
import jax.random
from flax import nnx
import optax
import json

import matplotlib.pyplot as plt


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
        x = jnp.hstack([x.real, x.imag])
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = nnx.leaky_relu(x)
            else:
                x = nnx.sigmoid(x)
        return x


def load_training_data(filename, max_lines=None):
    data = {'unique_widths': [], 'trans_real': [], 'trans_imag': [], 'ref_real': [], 'ref_imag': []}

    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            data_entry = json.loads(line)
            for key in data:
                data[key].append(data_entry[key])

    unique_widths = jnp.array(data['unique_widths'])
    trans_real = jnp.array(data['trans_real'])
    trans_imag = jnp.array(data['trans_imag'])
    ref_real = jnp.array(data['ref_real'])
    ref_imag = jnp.array(data['ref_imag'])
    trans = trans_real + 1j * trans_imag
    ref = ref_real + 1j * ref_imag
    amplitudes = jnp.block([trans, ref])

    return unique_widths, amplitudes


def evaluate_loss_convergence(model, x, y):
    y_pred = model(x)
    model_errors = jnp.mean((y - y_pred) ** 2, axis=-1)
    mean_errors_wrt_batch_size = jnp.cumsum(model_errors) / jnp.arange(1, len(x) + 1)
    return mean_errors_wrt_batch_size


@nnx.jit
def train_step(model, optimizer, x, y):
  def loss_fn(model):
    y_pred = model(x)
    return jnp.mean((y_pred - y) ** 2)

  loss, grads = nnx.value_and_grad(loss_fn)(model)
  optimizer.update(grads)

  return loss


def define_and_train_amplitudes_to_widths_model():
    unique_widths, amplitudes = load_training_data('ai_training_data/red_th500.jsonl')
    max_width = 300.
    unique_widths /= max_width
    print(unique_widths.shape)
    print(amplitudes.shape)

    model = SquarePixelLensOptimizingModel(
        n_propagating_waves=amplitudes.shape[-1] // 2,
        n_lens_params=unique_widths.shape[-1],
        hidden_layer_dims=[128] * 3,
        rngs=nnx.Rngs(0)
    )

    # error_convergence = evaluate_loss_convergence(model, amplitudes[:1000], unique_widths[:1000])
    # error_convergence = jnp.abs(jnp.diff(error_convergence)) / error_convergence[1:]
    # plt.plot(error_convergence)
    # plt.xscale('log')
    # # plt.yscale('log')
    # plt.grid()
    # plt.xlabel('Batch size')
    # plt.ylabel('Loss relative difference')
    # plt.show()

    learning_rate = 1e-2
    batch_size = 100
    n_epochs = 500
    n_training_samples = round(0.9 * len(unique_widths))
    n_validation_samples = len(unique_widths) - n_training_samples
    n_batches_per_epoch = n_training_samples // batch_size

    rng_key = jax.random.key(0)

    optimizer = nnx.Optimizer(model, optax.adam(learning_rate))
    for epoch in range(n_epochs):
        rng_key, rng_subkey = jax.random.split(rng_key)
        data_permutation = jax.random.permutation(rng_subkey, n_training_samples)
        current_loss = 1.
        for i in range(n_batches_per_epoch):
            batch_start_index = i * batch_size
            batch_end_index = batch_start_index + batch_size
            x = amplitudes[data_permutation[batch_start_index:batch_end_index]]
            y = unique_widths[data_permutation[batch_start_index:batch_end_index]]
            current_loss = train_step(model, optimizer, x, y)
        print(epoch, current_loss, sep='\t')


if __name__ == '__main__':
    define_and_train_amplitudes_to_widths_model()
