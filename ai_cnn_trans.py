import numpy as np
import jax.numpy as jnp
import jax
from flax import nnx
import optax
import orbax.checkpoint as ocp

jax.config.update('jax_enable_x64', True)


class MetasurfaceTransmissionCNN(nnx.Module):

    def __init__(self, rngs):
        self.conv1 = nnx.Conv(1, 16, (5, 5), strides=(2, 2), rngs=rngs)
        self.conv2 = nnx.Conv(16, 32, (5, 5), strides=(2, 2), rngs=rngs)
        self.conv3 = nnx.Conv(32, 64, (3, 3), strides=(2, 2), rngs=rngs)
        self.conv4 = nnx.Conv(64, 128, (3, 3), strides=(2, 2), rngs=rngs)
        self.conv5 = nnx.Conv(128, 128, (3, 3), strides=(2, 2), rngs=rngs)
        self.conv6 = nnx.Conv(128, 128, (3, 3), strides=(2, 2), rngs=rngs)
        self.linear1 = nnx.Linear(128, 64, rngs=rngs)
        self.linear2 = nnx.Linear(64, 1, rngs=rngs)

    def __call__(self, x):
        x = x[..., None]

        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6]:
            x = layer(x)
            x = nnx.gelu(x)

        x = jnp.squeeze(x, axis=(-2, -3))
        x = self.linear1(x)
        x = nnx.gelu(x)
        x = self.linear2(x)
        x = nnx.sigmoid(x)

        return jnp.squeeze(x, axis=-1)


class HybridFNOCNN(nnx.Module):
    def __init__(self, fno, cnn):
        self.fno = fno
        self.cnn = cnn

    def __call__(self, x):
        amps = self.fno(x)
        cnn_trans = self.cnn(x)
        fno_trans = jnp.sum(jnp.abs(amps) ** 2, axis=-1)
        amps = amps * jnp.sqrt(cnn_trans / fno_trans)[..., jnp.newaxis]
        return amps


def train_trans_cnn(training_dataset_path):
    rngs = nnx.Rngs(42)
    checkpointer = ocp.StandardCheckpointer()
    checkpoint_dir = 'C:/Users/eugene/PycharmProjects/fmmeta/ai_models_fno/'

    data = np.load(training_dataset_path)
    x_data = jnp.array(data['patterns'].astype(float))
    y_data = jnp.array(data['trans'].astype(float))

    print(x_data.shape, y_data.shape)
    print('Training set variance:', np.var(y_data))

    # n_train = 27000
    # n_val = 3000
    n_train = 1000
    n_val = 500
    batch_size = 250
    n_batches = n_train // batch_size
    x_train = x_data[:n_train]
    y_train = y_data[:n_train]
    x_val = x_data[n_train:n_train + n_val]
    y_val = y_data[n_train:n_train + n_val]

    model = MetasurfaceTransmissionCNN(rngs=rngs)

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

    # optimizer = nnx.Optimizer(model, optax.adam(1e-4))
    optimizer = nnx.Optimizer(model, optax.adamw(1e-4, weight_decay=1e-4))
    min_loss = 1000.

    for epoch in range(11):
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
            _, state = nnx.split(model)
            checkpointer.save(checkpoint_dir + 'trans_cnn_checkpoint', state, force=True)


if __name__ == '__main__':
    # train_trans_cnn('wave_pattern_training_data/wave_red_30k_trans.npz')

    abstract_model = nnx.eval_shape(lambda: MetasurfaceTransmissionCNN(rngs=nnx.Rngs(0)))
    graphdef, abstract_state = nnx.split(abstract_model)
    checkpointer = ocp.StandardCheckpointer()
    checkpoint_dir = 'C:/Users/eugene/PycharmProjects/fmmeta/ai_models_fno/'
    state_restored = checkpointer.restore(checkpoint_dir + 'trans_cnn_checkpoint', abstract_state)
    model = nnx.merge(graphdef, state_restored)
    print(model)
