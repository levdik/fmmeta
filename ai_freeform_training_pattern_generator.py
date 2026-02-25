import numpy as np
import jax
import jax.numpy as jnp
import optax

import topology_parametrization
from manufacturing_constraints import too_thin_area
from design_optimizer import run_gradient_ascent

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

jax.config.update('jax_enable_x64', True)


optimizer = optax.adam(learning_rate=1e-2)

def make_x_valid(x, filling):
    opt_state = optimizer.init(x)

    min_f = 1000
    best_x = x.copy()

    for i in range(100):
        new_x, opt_state, loss = step(x, opt_state, filling)
        min_f = jnp.minimum(loss, min_f)
        best_x = jnp.where(loss < min_f, new_x, x)
        x = new_x
        if min_f < 1e-3:
            break

    return best_x


if __name__ == '__main__':
    min_width_px_range = [1, 3, 5, 7, 9, 11]
    sigma_range = [16, 32, 64]
    # n_samples = 4224
    n_samples = 33792

    seed_count = 0
    patterns = []

    parameter_count = 0
    # nn = [234] * 6 + [235] * 12
    nn = [1877] * 12 + [1878] * 6
    assert np.sum(nn) == n_samples

    for sigma in sigma_range:
        for min_width_px in min_width_px_range:
            topology = topology_parametrization.GaussianField(100, sigma, symmetry_type='main_diagonal')

            @jax.jit
            def step(x, opt_state, filling):
                def loss_fn(x):
                    pattern = topology(x)
                    invalid_area = too_thin_area(pattern, min_width_px)
                    filling_mismatch = jnp.abs(jnp.sum(pattern) / x.size - filling)
                    return invalid_area + filling_mismatch

                loss_value_and_grad_fn = jax.value_and_grad(loss_fn)
                loss, grad = loss_value_and_grad_fn(x)
                updates, opt_state = optimizer.update(grad, opt_state)
                x = optax.apply_updates(x, updates)
                x = jnp.clip(x, -1, 1)
                return x, opt_state, loss

            for _ in range(nn[parameter_count % len(nn)]):
                key = jax.random.key(seed_count)
                key, filling_key, values_key = jax.random.split(key, 3)
                filling = jnp.clip(0.2 * jax.random.normal(filling_key) + 0.5, 0.05, 0.95)
                x = jax.random.uniform(values_key, topology.n_geometrical_parameters, minval=-1, maxval=1)

                x_valid = make_x_valid(x, filling)
                pattern_valid = topology(x_valid)
                patterns.append(np.array(pattern_valid))

                print(seed_count)
                seed_count += 1

            parameter_count += 1

    patterns = np.array(patterns)
    np.save('freeform_training_patterns.npy', patterns)
