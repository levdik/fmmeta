import jax
import jax.numpy as jnp
import numpy as np
import optax

from collections.abc import Callable


def run_gradient_ascent(
        target_function: Callable,
        x_init: jnp.ndarray,
        learning_rate: float,
        n_steps: int,
        boundary_projection_function: Callable = lambda x: x,
        except_keyboard_interrupt: bool = False
):
    loss_value_and_grad_fn = jax.value_and_grad(lambda x: -target_function(x))

    optimizer = optax.adam(learning_rate, b1=0.9)
    opt_state = optimizer.init(x_init)

    @jax.jit
    def step(x, opt_state):
        loss, grad = loss_value_and_grad_fn(x)
        updates, opt_state = optimizer.update(grad, opt_state)
        x = optax.apply_updates(x, updates)
        x = boundary_projection_function(x)
        return x, opt_state, loss, grad

    x = x_init
    max_f = 0.
    best_x = x

    try:
        for i in range(n_steps):
            new_x, opt_state, loss, grad = step(x, opt_state)
            # print(i, repr(x), repr(new_x), -loss, repr(grad), sep='\n')

            if -loss > max_f:
                max_f = -loss
                best_x = x

            print(i, -loss, sep='\t')
            x = new_x
    except KeyboardInterrupt as e:
        if not except_keyboard_interrupt:
            raise e
        else:
            pass

    final_f = target_function(x)
    print(n_steps, final_f, sep='\t')
    if final_f > max_f:
        max_f = final_f
        best_x = x

    return best_x, max_f
