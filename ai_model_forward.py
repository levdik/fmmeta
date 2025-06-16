import jax
import jax.numpy as jnp
from flax import nnx
import optax

import pickle


class SquarePixelLensScatteringModel(nnx.Module):
    def __init__(
            self,
            n_propagating_waves: int,
            n_lens_params: int,
            hidden_layer_dims: list[int],
            rngs: nnx.Rngs,
            include_transmission=False
    ):
        self.include_transmission = include_transmission
        self.n_lens_params = n_lens_params
        self.n_propagating_waves = n_propagating_waves
        if include_transmission:
            output_dim = 4 * n_propagating_waves
        else:
            output_dim = 2 * n_propagating_waves
        layer_dims = [n_lens_params] + hidden_layer_dims + [output_dim]
        layers = []

        for i in range(len(layer_dims) - 1):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]
            linear = nnx.Linear(in_dim, out_dim, rngs=rngs)
            layers.append(linear)

        self.layers = layers
        self.dropout = nnx.Dropout(0.1, rngs=rngs)

    def __call__(self, lens_widths: jnp.ndarray) -> jnp.ndarray:
        x = lens_widths.reshape(lens_widths.shape[0], -1)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = nnx.leaky_relu(x)

        if self.include_transmission:
            trans_ref_amps = x[..., :2 * self.n_propagating_waves] + 1j * x[..., 2 * self.n_propagating_waves:]
            power_per_sample = jnp.linalg.norm(trans_ref_amps, axis=-1)
            normalized_trans_ref_amps = trans_ref_amps / power_per_sample[:, jnp.newaxis]
            return normalized_trans_ref_amps
        return x[..., :self.n_propagating_waves] + 1j * x[..., self.n_propagating_waves:]

    def save(self, filename):
        _, state = nnx.split(self)
        with open(filename, 'wb') as f:
            pickle.dump(state, f)

    @staticmethod
    def load(
            filename: str,
            n_propagating_waves: int,
            n_lens_params: int,
            hidden_layer_dims: list[int],
            include_transmission=False
    ):
        abstract_model = nnx.eval_shape(
            lambda: SquarePixelLensScatteringModel(
                n_propagating_waves,
                n_lens_params,
                hidden_layer_dims,
                nnx.Rngs(0),
                include_transmission))
        graph_def, abstract_state = nnx.split(abstract_model)
        with open(filename, 'rb') as f:
            state_restored = pickle.load(f)
        model = nnx.merge(graph_def, state_restored)
        return model


def train_forward_model(hidden_dims, batch_size, n_epochs, learning_rate):
    max_width = 300.
    # incidence_reference_amplitude = -1.20536745e-01 + 9.9270886e-01j  # -exp(1j * k0 * thickness)
    # data = jnp.load('ai_training_data/red_7x7_th500_p300_120k.npz')
    # data = jnp.load('ai_training_data/red_8x8_th800_p300.npz')
    data = jnp.load('ai_training_data/red_4x4_symmetries.npz')

    widths = jnp.array(data['widths'])
    widths = widths.reshape(widths.shape[0], -1)
    amplitudes = jnp.array(data['amps'])
    # amplitudes /= jnp.linalg.norm(amplitudes, axis=-1, keepdims=True)

    n_propagating_waves = amplitudes.shape[-1] // 2
    amplitudes = amplitudes[:, :n_propagating_waves]
    n_lens_params = widths.shape[-1]

    widths /= max_width
    # amplitudes = amplitudes.at[:, :n_propagating_waves].divide(incidence_reference_amplitude)

    print('Widths shape:', widths.shape)
    print('Amps shape:', amplitudes.shape)
    print('n_propagating_waves:', n_propagating_waves)
    print('n_lens_params:', n_lens_params)

    # approximate_test_to_train_ratio = 1 / 6
    approximate_test_to_train_ratio = 0.05
    total_n_samples = widths.shape[0]
    n_batches = int(((1 - approximate_test_to_train_ratio) * total_n_samples) // batch_size)
    n_training_samples = n_batches * batch_size
    print(total_n_samples, n_training_samples)
    training_width = widths[:n_training_samples]
    validation_width = widths[n_training_samples:]
    training_amplitudes = amplitudes[:n_training_samples]
    validation_amplitudes = amplitudes[n_training_samples:]

    model = SquarePixelLensScatteringModel(
        n_propagating_waves=n_propagating_waves,
        n_lens_params=n_lens_params,
        hidden_layer_dims=hidden_dims,
        rngs=nnx.Rngs(0)
    )
    model.train()

    @nnx.jit
    def train_step(model, optimizer, x, y):
        def loss_fn(model):
            y_pred = model(x)
            loss_per_sample = jnp.sum(jnp.abs(y_pred - y) ** 2, axis=-1)
            return jnp.mean(loss_per_sample)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)
        return loss

    @nnx.jit
    def validate_loss(model, x, y):
        y_pred = model(x)
        loss_per_sample = jnp.sum(jnp.abs(y_pred - y) ** 2, axis=-1)
        return jnp.mean(loss_per_sample)

    rng_key = jax.random.key(0)
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate))
    current_loss = min_validation_loss = 100.

    for epoch in range(n_epochs):
        rng_key, rng_subkey = jax.random.split(rng_key)
        data_permutation = jax.random.permutation(rng_subkey, n_training_samples)

        for i in range(n_batches):
            batch_start_index = i * batch_size
            batch_end_index = batch_start_index + batch_size
            x = widths[data_permutation[batch_start_index:batch_end_index]]
            y = amplitudes[data_permutation[batch_start_index:batch_end_index]]
            current_loss = train_step(model, optimizer, x, y)
            if i == 0 and epoch == 0:
                validation_loss = validate_loss(model, validation_width, validation_amplitudes)
                print(0, current_loss, validation_loss, sep='\t')
            # print(epoch, i, current_loss, sep='\t')
        # model.eval()
        validation_loss = validate_loss(model, validation_width, validation_amplitudes)
        # model.train()
        print(epoch + 1, current_loss, validation_loss, sep='\t')
        if validation_loss < min_validation_loss:
            min_validation_loss = validation_loss
            model.save(f'ai_models/red_{int(jnp.sqrt(n_lens_params))}x{int(jnp.sqrt(n_lens_params))}_forward')


if __name__ == '__main__':
    train_forward_model(
        hidden_dims=[32, 32, 32],
        batch_size=200,
        n_epochs=1000,
        learning_rate=1e-2
    )
