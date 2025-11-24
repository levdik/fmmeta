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
        except_keyboard_interrupt: bool = False,
        print_step: bool = True
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

            if print_step:
                print(i, -loss, sep='\t')
            x = new_x
    except KeyboardInterrupt as e:
        if not except_keyboard_interrupt:
            raise e
        else:
            pass

    final_f = target_function(x)
    if print_step:
        print(n_steps, final_f, sep='\t')
    if final_f > max_f:
        max_f = final_f
        best_x = x

    return best_x, max_f


def run_basinhopping(
        target_function: Callable,
        x_init: jnp.ndarray,
        initial_temp: float,
        cooling_rate: float,
        n_bassinhopping_steps: int,
        learning_rate: float,
        n_descent_steps: int,
        mutation_minval: float,
        mutation_maxval: float,
        boundary_projection_function: Callable = lambda x: x,
        except_keyboard_interrupt: bool = False,
        print_descent: bool = False,
        print_bassinhopping: bool = True
):
    rng_key = jax.random.key(42)

    temp = initial_temp
    x = x_init
    f = -100
    best_x = None
    max_f = -100

    for i in range(n_bassinhopping_steps):
        new_x, new_f = run_gradient_ascent(
            target_function, x, learning_rate, n_descent_steps,
            boundary_projection_function, except_keyboard_interrupt, print_descent)
        rng_key, acceptance_key, mutation_index_key, mutation_value_key = jax.random.split(rng_key, 4)
        acceptance_probability = jnp.exp((new_f - f) / temp)
        accepted = jax.random.uniform(acceptance_key) < acceptance_probability

        if print_bassinhopping:
            print(f'Step: {i},\ttemp: {temp},\tprev_f: {f},\tnew_f: {new_f},\taccept: {accepted},\tbest_f: {max_f}')

        if accepted:
            f = new_f
            x = new_x
        if new_f > max_f:
            max_f = new_f
            best_x = new_x

        x = x.at[
            jax.random.randint(mutation_index_key, (), 0, x.size)
        ].set(
            jax.random.uniform(mutation_value_key, minval=mutation_minval, maxval=mutation_maxval)
        )
        temp *= cooling_rate

    return best_x, max_f
