import jax
import jax.numpy as jnp
from flax import nnx
import optax

import pickle


class WavyLensScatteringModel(nnx.Module):
    def __init__(
            self,
            n_propagating_waves: int,
            n_lens_amps: int,
            hidden_layer_dims: list[int],
            rngs: nnx.Rngs,
    ):
        self.n_lens_amps = n_lens_amps
        self.n_propagating_waves = n_propagating_waves

        input_dim = 2 * n_lens_amps - 1
        output_dim = 2 * n_propagating_waves
        layer_dims = [input_dim] + hidden_layer_dims + [output_dim]
        self.layers = [
            nnx.Linear(layer_dims[i], layer_dims[i + 1], rngs=rngs)
            for i in range(len(layer_dims) - 1)
        ]


    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = nnx.leaky_relu(x)
            # else:
                # x = nnx.tanh(x)
        return x

    def save(self, filename):
        _, state = nnx.split(self)
        with open(filename, 'wb') as f:
            pickle.dump(state, f)

    @staticmethod
    def load(
            filename: str,
            n_propagating_waves: int,
            n_lens_amps: int,
            hidden_layer_dims: list[int],
    ):
        abstract_model = nnx.eval_shape(
            lambda: WavyLensScatteringModel(
                n_propagating_waves,
                n_lens_amps,
                hidden_layer_dims,
                nnx.Rngs(0)
            )
        )
        graph_def, abstract_state = nnx.split(abstract_model)
        with open(filename, 'rb') as f:
            state_restored = pickle.load(f)
        model = nnx.merge(graph_def, state_restored)
        return model


def train_model(training_data, hidden_dims, batch_size, n_epochs, learning_rate, save_filename=None):
    pattern_amps = training_data['primary_pattern_amps']
    project_d1_onto = training_data['v1']
    field_amps = training_data['scattered_field_amps']
    projected_jac = training_data['projected_jac']

    n_propagating_waves = field_amps.shape[1] // 2
    n_lens_amps = pattern_amps.shape[1]
    total_n_samples = pattern_amps.shape[0]

    pattern_amps = jnp.hstack([pattern_amps[..., 0].real[:, None], pattern_amps[..., 1:].real, pattern_amps[..., 1:].imag])
    projected_jac = jnp.hstack([projected_jac[:, :n_lens_amps], projected_jac[:, n_lens_amps + 1:]])

    approximate_test_to_train_ratio = 0.1
    n_batches = int(((1 - approximate_test_to_train_ratio) * total_n_samples) // batch_size)
    n_training_samples = n_batches * batch_size

    model = WavyLensScatteringModel(
        n_propagating_waves=n_propagating_waves,
        n_lens_amps=n_lens_amps,
        hidden_layer_dims=hidden_dims,
        rngs=nnx.Rngs(42)
    )

    @nnx.jit
    def train_step(model, optimizer, x, y, v1, proj_jac):
        def loss_fn(model):
            # def calc_proj_jax_single_input(x, v1):
            #     def calculate_y_and_project_single_input(x):
            #         y = model(x)
            #         return jnp.dot(y, v1)
            #     return jax.grad(calculate_y_and_project_single_input)(x)
            # proj_jac_pred = jax.vmap(calc_proj_jax_single_input)(x, v1)

            y_pred = model(x)
            loss_per_sample = (
                    jnp.sum(jnp.abs(y_pred - y) ** 2, axis=-1)
                    # + 0.1 * jnp.sum(jnp.clip(jnp.abs(proj_jac_pred - proj_jac) ** 2, 0, 1), axis=-1)
            )
            return jnp.mean(loss_per_sample)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)
        return loss

    @nnx.jit
    def validate_model(model, x, y):
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
            current_loss = train_step(
                model, optimizer,
                x=pattern_amps[:n_training_samples][data_permutation[batch_start_index:batch_end_index]],
                y=field_amps[:n_training_samples][data_permutation[batch_start_index:batch_end_index]],
                v1=project_d1_onto[:n_training_samples][data_permutation[batch_start_index:batch_end_index]],
                proj_jac=projected_jac[:n_training_samples][data_permutation[batch_start_index:batch_end_index]]
            )

            # print(epoch, i, current_loss)
        validation_loss = validate_model(model, pattern_amps[n_training_samples:], field_amps[n_training_samples:])
        print(epoch, current_loss, validation_loss)

            # if i == 0 and epoch == 0:
            #     validation_loss = validate_loss(model, validation_widths, validation_amplitudes)
            #     print(0, current_loss, validation_loss, sep='\t')

        # validation_loss = validate_loss(model, validation_widths, validation_amplitudes)
        # print(epoch + 1, current_loss, validation_loss, sep='\t')
        # if (save_filename is not None) and (validation_loss < min_validation_loss):
        #     min_validation_loss = validation_loss
        #     model.save(f'ai_models/{save_filename}.pkl')


if __name__ == '__main__':
    data = jnp.load('wave_pattern_training_data/wave_red_30k.npz')
    train_model(
        training_data=data,
        hidden_dims= [64] * 6,
        batch_size=512,
        n_epochs=100,
        learning_rate=1e-3,
        # save_filename=
    )
