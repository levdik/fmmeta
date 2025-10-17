import jax
import jax.numpy as jnp
from flax import nnx
import optax

from ai_fno import RealToComplexFNO

jax.config.update('jax_enable_x64', True)


def train_fno_surrogate():
    rngs = nnx.Rngs(42)
    data = jnp.load('wave_pattern_training_data/wave_red_30k_maps.npz')
    x_data = data['x'].astype(float)
    y_data = data['y'].astype(complex)
    print(x_data.shape, y_data.shape)

    # n_train = 27000
    # n_val = 3000
    n_train = 100
    n_val = 10
    batch_size = 100
    n_batches = n_train // batch_size
    x_train = x_data[:n_train]
    y_train = y_data[:n_train]
    x_val = x_data[n_train:n_train + n_val]
    y_val = y_data[n_train:n_train + n_val]

    model = RealToComplexFNO(
        hidden_n_channels=[64] * 4,
        n_pixels=64,
        mode_threshold=16,
        activation_fn=nnx.gelu,
        rngs=rngs
    )

    @nnx.jit
    def train_step(model, optimizer, x, y):
        def loss_fn(model):
            y_pred = model(x)
            return jnp.mean(jnp.abs(y_pred - y) ** 2)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)
        return loss

    @nnx.jit
    def validate_loss(model, x, y):
        y_pred = model(x)
        return jnp.mean(jnp.abs(y_pred - y) ** 2)

    optimizer = nnx.Optimizer(model, optax.adam(1e-4))
    min_loss = 1000.

    for epoch in range(100):
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

        if val_loss < min_loss:
            min_loss = val_loss
            model.save('ai_models/fno_red.pkl')


if __name__ == '__main__':
    train_fno_surrogate()
