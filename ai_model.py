<<<<<<< HEAD
import jax.numpy as jnp
import jax.random
from flax import nnx
import optax

import matplotlib.pyplot as plt

import pickle


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

    def save(self, filename):
        _, state = nnx.split(self)
        with open(filename, 'wb') as f:
            pickle.dump(state, f)

    @staticmethod
    def load(
            filename: str,
            n_propagating_waves: int,
            n_lens_params: int,
            hidden_layer_dims: list[int]
    ):
        abstract_model = nnx.eval_shape(
            lambda: SquarePixelLensOptimizingModel(n_propagating_waves, n_lens_params, hidden_layer_dims, nnx.Rngs(0)))
        graph_def, abstract_state = nnx.split(abstract_model)
        with open(filename, 'rb') as f:
            state_restored = pickle.load(f)
        model = nnx.merge(graph_def, state_restored)
        return model


def evaluate_loss_convergence(model, x, y):
    y_pred = model(x)
    model_errors = jnp.mean((y - y_pred) ** 2, axis=-1)
    error_convergence = jnp.cumsum(model_errors) / jnp.arange(1, len(x) + 1)
    error_convergence = jnp.abs(jnp.diff(error_convergence)) / error_convergence[1:]
    plt.plot(error_convergence)
    plt.xscale('log')
    # plt.yscale('log')
    plt.grid()
    plt.xlabel('Batch size')
    plt.ylabel('Loss relative difference')
    plt.show()


@nnx.jit
def train_step(model, optimizer, x, y):
    def loss_fn(model):
        y_pred = model(x)
        return jnp.mean((y_pred - y) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)

    return loss

@nnx.jit
def validate_loss(model, x, y):
    def loss_fn(model):
        y_pred = model(x)
        return jnp.mean((y_pred - y) ** 2)

    return loss_fn(model)


def define_and_train_amplitudes_to_widths_model():
    data = jnp.load('ai_training_data/red_th500.npz')
    widths = jnp.array(data['widths'])
    widths = widths.reshape(widths.shape[0], -1)
    amplitudes = jnp.array(data['amps'])
    max_width = 300.
    widths /= max_width
    print(widths.shape)
    print(amplitudes.shape)

    dimss = [
        [64],
        [64, 64],
        [64, 64, 64],
        [128],
        [128, 128],
        [128, 128, 128],
        [128, 64],
        [128, 64, 32]
    ]

    for dims in dimss:
        print('Start', dims)
        save_filename = f'ai_models/red_7x7_p300_th500_{'_'.join(map(str, dims))}'
        model = SquarePixelLensOptimizingModel(
            n_propagating_waves=amplitudes.shape[-1] // 2,
            n_lens_params=widths.shape[-1],
            # hidden_layer_dims=[256, 256, 128],
            # hidden_layer_dims=[512, 256, 128, 64],
            # hidden_layer_dims=[1024, 512, 256, 128, 64],
            hidden_layer_dims=dims,
            rngs=nnx.Rngs(0)
        )
        # model = SquarePixelLensOptimizingModel.load(
        #     'ai_models/red_7x7_p300_th500.pkl',
        #         n_propagating_waves=amplitudes.shape[-1] // 2,
        #         n_lens_params=widths.shape[-1],
        #         hidden_layer_dims=[256, 256, 128]
        # )

        # evaluate_loss_convergence(model, amplitudes[:1000], widths[:1000])

        learning_rate = 1e-3
        batch_size = 200
        n_epochs = 1000
        n_training_samples = round(0.9 * len(widths))
        n_validation_samples = len(widths) - n_training_samples
        n_batches_per_epoch = n_training_samples // batch_size

        rng_key = jax.random.key(0)

        optimizer = nnx.Optimizer(model, optax.adam(learning_rate))
        current_loss = min_validation_loss = 1.
        for epoch in range(n_epochs):
            rng_key, rng_subkey = jax.random.split(rng_key)
            data_permutation = jax.random.permutation(rng_subkey, n_training_samples)
            rng_key, rng_subkey = jax.random.split(rng_key)
            constant_phase_multipliers = jnp.exp(
                1j * 2 * jnp.pi *
                jax.random.uniform(rng_subkey, (n_training_samples, 1)))
            for i in range(n_batches_per_epoch):
                batch_start_index = i * batch_size
                batch_end_index = batch_start_index + batch_size
                x = amplitudes[data_permutation[batch_start_index:batch_end_index]]
                x *= constant_phase_multipliers[batch_start_index:batch_end_index]
                y = widths[data_permutation[batch_start_index:batch_end_index]]
                current_loss = train_step(model, optimizer, x, y)
            if epoch % 10 == 0:
                validation_widths = widths[-n_validation_samples:]
                validation_amps = amplitudes[-n_validation_samples:]
                rng_key, rng_subkey = jax.random.split(rng_key)
                constant_phase_multipliers = jnp.exp(
                    1j * 2 * jnp.pi *
                    jax.random.uniform(rng_subkey, (n_validation_samples, 1)))
                validation_amps *= constant_phase_multipliers
                validated_loss = validate_loss(model, validation_amps, validation_widths)
                print('Loss on validation:', validated_loss)
                if validated_loss < min_validation_loss:
                    model.save(f'{save_filename}_intermediate.pkl')
                    min_validation_loss = validated_loss
            print(epoch, current_loss, sep='\t')
        model.save(f'{save_filename}.pkl')


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
    hidden_layer_dims = [512, 256, 128, 64]
    model = SquarePixelLensOptimizingModel.load(
        'ai_models/red_7x7_p300_th500_big.pkl',
            n_propagating_waves=n_propagating,
            n_lens_params=n_widths,
            hidden_layer_dims=hidden_layer_dims
    )

    target_amps = jnp.zeros(2 * n_propagating, dtype=complex).at[jnp.array([0, 1])].set(1.)
    target_amps = target_amps / jnp.linalg.norm(target_amps)
    print(target_amps)
    predicted_widths = jnp.round(model(target_amps) * max_width).reshape(n_lens_subpixels, n_lens_subpixels)

    from ai_trainig_set_generator import prepare_amplitude_generating_function
    f = prepare_amplitude_generating_function(
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
    print(jnp.round(jnp.abs(actual_amps[:n_propagating]) ** 2, 3))
    print(jnp.round(jnp.abs(actual_amps[n_propagating:]) ** 2, 3))


if __name__ == '__main__':
    define_and_train_amplitudes_to_widths_model()
    # test_model()
=======
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
>>>>>>> 0621fbc2b595c0be385a7275272578b85e6502db
