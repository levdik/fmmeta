import jax.numpy as jnp
import jax.random
from flax import nnx
import optax

from scattering_solver_factory import prepare_lens_pixel_width_to_scattered_amplitudes_function
from field_postprocessing import min_difference_between_amplitude_vectors, min_distance_between_amplitude_vectors
from fmmax import basis

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


# def evaluate_loss_convergence(model, x, y):
#     y_pred = model(x)
#     model_errors = jnp.mean((y - y_pred) ** 2, axis=-1)
#     error_convergence = jnp.cumsum(model_errors) / jnp.arange(1, len(x) + 1)
#     error_convergence = jnp.abs(jnp.diff(error_convergence)) / error_convergence[1:]
#     plt.plot(error_convergence)
#     plt.xscale('log')
#     # plt.yscale('log')
#     plt.grid()
#     plt.xlabel('Batch size')
#     plt.ylabel('Loss relative difference')
#     plt.show()


def define_and_train_amplitudes_to_widths_model(hidden_dims):
    data = jnp.load('ai_training_data/red_th500.npz')
    widths = jnp.array(data['widths'])
    widths = widths.reshape(widths.shape[0], -1)
    amplitudes = jnp.array(data['amps'])
    max_width = 300.
    widths /= max_width
    print(widths.shape)
    print(amplitudes.shape)

    model = SquarePixelLensOptimizingModel(
        n_propagating_waves=amplitudes.shape[-1] // 2,
        n_lens_params=widths.shape[-1],
        # hidden_layer_dims=[256, 256, 128],
        # hidden_layer_dims=[512, 256, 128, 64],
        hidden_layer_dims=hidden_dims,
        rngs=nnx.Rngs(0)
    )
    # model = SquarePixelLensOptimizingModel.load(
    #     'ai_models/red_7x7_p300_th500.pkl',
    #         n_propagating_waves=amplitudes.shape[-1] // 2,
    #         n_lens_params=widths.shape[-1],
    #         hidden_layer_dims=[256, 256, 128]
    # )

    learning_rate = 1e-3
    batch_size = 100
    n_epochs = 1000
    n_training_samples = round(0.9 * len(widths))
    n_validation_samples = len(widths) - n_training_samples
    n_batches_per_epoch = n_training_samples // batch_size

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
        y_pred = model(x)
        loss = jnp.mean((y_pred - y) ** 2)
        return loss

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
                model.save('ai_models/red_7x7_p300_th500_big_intermediate.pkl')
                min_validation_loss = validated_loss
        print(epoch, current_loss, sep='\t')
    model.save('ai_models/red_7x7_p300_th500_big.pkl')


def define_and_train_amplitudes_to_widths_model_online_inversion(hidden_dims):
    max_width = 300.
    n_propagating = 37
    n_lens_params = 7 ** 2
    # hidden_dims = [128, 128, 128]

    model = SquarePixelLensOptimizingModel(
        n_propagating_waves=n_propagating,
        n_lens_params=n_lens_params,
        hidden_layer_dims=hidden_dims,
        rngs=nnx.Rngs(0)
    )
    # model = SquarePixelLensOptimizingModel.load(
    #     'ai_models/red_7x7_p300_th500.pkl',
    #         n_propagating_waves=n_propagating,
    #         n_lens_params=n_lens_params,
    #         hidden_layer_dims=hidden_dims
    # )

    learning_rate = 1e-3
    batch_size = 25
    n_epochs = 1000

    widths_to_amps_function = prepare_lens_pixel_width_to_scattered_amplitudes_function(
        wavelength=650,
        permittivity=4,
        lens_subpixel_size=max_width,
        n_lens_subpixels=int(n_lens_params ** 0.5),
        lens_thickness=500,
        approximate_number_of_terms=300
    )
    vmap_f = jax.vmap(jax.jit(widths_to_amps_function))

    # expansion = basis.generate_expansion(
    #     primitive_lattice_vectors=basis.LatticeVectors(u=basis.X, v=basis.Y),
    #     approximate_num_terms=n_propagating,
    # )
    # basis_indices = expansion.basis_coefficients
    # n, m = basis_indices.T
    # weights = 1 / (jnp.sqrt(n ** 2 + m ** 2) + 1)
    # weights = jnp.concatenate([weights, weights])
    # def weighted_min_distance_between_amplitude_vectors(a1, a2):
    #     diff = min_difference_between_amplitude_vectors(a1, a2)
    #     mean_weighted_distance = jnp.sum(diff * weights) / jnp.sum(weights)
    #     return mean_weighted_distance
    # vmap_weighted_distance_f = jax.vmap(weighted_min_distance_between_amplitude_vectors)

    @nnx.jit
    def train_step(model, optimizer, amps):
        def loss_fn(model):
            widths = model(amps) * max_width
            true_amps = vmap_f(widths)
            min_distances = jax.vmap(min_distance_between_amplitude_vectors)(amps, true_amps)
            # min_distances = vmap_weighted_distance_f(amps, true_amps)
            loss = jnp.mean(min_distances)
            return loss

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)

        return loss

    rng_key = jax.random.key(0)
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate))
    min_loss = 100.

    for epoch in range(n_epochs):
        rng_key, rng_subkey = jax.random.split(rng_key)
        amps_re_im = jax.random.uniform(rng_subkey, shape=(batch_size, 4 * n_propagating))
        amps = amps_re_im[:, :2 * n_propagating] + 1j * amps_re_im[:, 2 * n_propagating:]
        amps /= jnp.linalg.norm(amps, axis=-1).reshape(-1, 1)
        current_loss = train_step(model, optimizer, amps)
        print(epoch, current_loss, sep='\t')
        if current_loss < min_loss:
            min_loss = current_loss
            model.save(
                f'ai_models/red_7x7_p300_th500_online_inversion_{"_".join(list(map(str, hidden_dims)))}_batch{batch_size}_intermediate.pkl')
    model.save(
        f'ai_models/red_7x7_p300_th500_online_inversion_{"_".join(list(map(str, hidden_dims)))}_batch{batch_size}.pkl')


if __name__ == '__main__':
    dims = list(map(int, input().split()))

    # define_and_train_amplitudes_to_widths_model(dims)
    define_and_train_amplitudes_to_widths_model_online_inversion(dims)
