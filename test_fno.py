import jax
import jax.numpy as jnp
import optax
from flax import nnx

from ai_fno import FourierNeuralOperator
from lens_topology_parametrization import FourierInterpolationTopologyParametrization

import matplotlib.pyplot as plt
import matplotlib

# jax.config.update('jax_enable_x64', True)
# jax.config.update("jax_debug_nans", True)
matplotlib.use('TkAgg')


n_pixels = 64

x, y = jnp.meshgrid(jnp.linspace(0, 1, n_pixels), jnp.linspace(0, 1, n_pixels))
def f(input2d):
    xp = jnp.sin(x * jnp.pi)
    yp = jnp.sin(y * jnp.pi)
    values = (
        jnp.sin(2 * jnp.pi * xp * input2d)
        # * jnp.sin(2 * jnp.pi * yp * input2d)
        # * ((input2d % 1) ** 4 - 2 * (input2d % 1) ** 3 + (input2d % 1) ** 2)
    )
    return nnx.sigmoid(3 * values)


def generate_training_set_single_gridsize(n, grid_size, rngs):
    input_parametrization = FourierInterpolationTopologyParametrization(grid_size=grid_size)
    z = jax.vmap(
        input_parametrization._generate_pattern_cutoff_values, (0, None)
    )(
        jax.random.uniform(rngs(), (n, grid_size, grid_size), minval=-1, maxval=1), n_pixels
    )
    z = nnx.sigmoid(2 * z)
    fz = jax.vmap(f)(z)
    return z, fz

def generate_training_set(n, gs, rngs):
    z = []
    fz = []
    for ni, gsi in zip(n, gs):
        zi, fzi = generate_training_set_single_gridsize(ni, gsi, rngs)
        z.append(zi)
        fz.append(fzi)
    z = jnp.concatenate(z, axis=0)
    fz = jnp.concatenate(fz, axis=0)
    return z, fz


def visualize_training_set_examples():
    rngs = nnx.Rngs(42)
    n = 10
    z, fz = generate_training_set([1] * n, list(range(1, n + 1)), rngs)
    print(z.shape)

    fig, ax = plt.subplots(2, n)
    for i in range(n):
        ax[0, i].imshow(z[i], vmin=0, vmax=1)
        ax[1, i].imshow(fz[i], vmin=0, vmax=1)

    for ax_i in ax.flatten():
        ax_i.set_axis_off()

    plt.tight_layout()
    plt.show()


def train_model(n_epochs, training_set, validation_set, batch_size, hidden_dims, rngs):
    x_train, y_train = training_set
    x_val, y_val = validation_set
    n_batches = len(x_train) // batch_size

    model = FourierNeuralOperator(
        n_in_channels=1,
        n_out_channels=1,
        hidden_n_channels=hidden_dims,
        n_pixels=n_pixels,
        mode_threshold=10,
        activation_fn=nnx.leaky_relu,
        rngs=rngs
    )

    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(x_train[5000, ..., 0])
    # ax[1].imshow(model(x_train[5000][jnp.newaxis, ...])[0, ..., 0])
    # plt.show()
    # print(model.n_selected_modes)
    # print(model(x_train[0][jnp.newaxis, ...]).shape)

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
        return jnp.mean((y_pred - y) ** 2)

    optimizer = nnx.Optimizer(model, optax.adam(1e-4))

    for epoch in range(n_epochs):
        data_permutation = jax.random.permutation(rngs(), len(x_train))
        for i in range(n_batches):
            batch_start_index = i * batch_size
            batch_end_index = batch_start_index + batch_size
            x = x_train[data_permutation[batch_start_index:batch_end_index]]
            y = y_train[data_permutation[batch_start_index:batch_end_index]]
            current_loss = train_step(model, optimizer, x, y)
            print(epoch, i, current_loss)
        val_loss = validate_loss(model, x_val, y_val)
        print(epoch, val_loss)


if __name__ == '__main__':
    # visualize_training_set_examples()
    # exit()

    n_training = 10000
    n_val = n_training // 5
    rngs = nnx.Rngs(42)

    # x_train, y_train = generate_training_set([n_training // 5] * 5, list(range(1, 6)), rngs)
    # x_val, y_val = generate_training_set([n_val // 5] * 5, list(range(1, 6)), rngs)
    # jnp.savez('example_fno_training_data/ex1.npz', x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
    # exit()
    training_data = jnp.load('example_fno_training_data/ex1.npz')
    x_train = training_data['x_train'][..., jnp.newaxis]
    y_train = training_data['y_train'][..., jnp.newaxis]
    x_val = training_data['x_val'][..., jnp.newaxis]
    y_val = training_data['y_val'][..., jnp.newaxis]

    print(jnp.mean(jnp.var((
                                   x_train ** 2 + jnp.sin(0.5 * jnp.pi * x_train)
                           ).reshape(10000, -1), axis=-1)))

    train_model(
        n_epochs=100,
        # training_set=(x_train, y_train),
        # validation_set=(x_val, y_val),
        training_set=(x_train, x_train ** 2 + jnp.sin(0.5 * jnp.pi * x_train)),
        validation_set=(x_val, x_val ** 2 + jnp.sin(0.5 * jnp.pi * x_val)),
        batch_size=500,
        hidden_dims=[64] * 2,
        rngs=rngs
    )
