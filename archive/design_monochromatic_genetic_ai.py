import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp
from scipy.optimize import differential_evolution

from ai_fno import PatternToAmpsFNO
from field_postprocessing import calculate_focusing_efficiency, propagate_amps_in_free_space
import topology_parametrization
from design_optimizer import run_gradient_ascent
from scattering_simulation import prepare_lens_scattering_solver
from manufacturing_constraints import too_thin_area
from design_monochromatic_fno import load_fno_model

import pickle
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
jax.config.update('jax_enable_x64', True)

if __name__ == '__main__':
    model = load_fno_model()

    topology = topology_parametrization.GaussianField(100, 16, 'central')

    x = jax.random.uniform(
        jax.random.key(8),
        shape=(topology.n_geometrical_parameters,),
        minval=topology.minval,
        maxval=topology.maxval
    )

    simulate_scattering, expansion = prepare_lens_scattering_solver(
        wavelength=650,
        period=2000,
        lens_thickness=600,
        substrate_thickness=500,
        approximate_number_of_terms=300,
        propagate_by_distance=3500
    )


    # def predicted_efficiency(x):
    #     pattern = topology(x, n_samples=64)
    #     y = model(pattern[None, :])[0]
    #     amps = y
    #     focal_amps = propagate_amps_in_free_space(amps, 3500, expansion, 650, 2000)
    #     transmission_efficiency = jnp.sum(jnp.abs(focal_amps) ** 2)
    #     transmission_efficiency = jnp.minimum(transmission_efficiency, 1.)
    #     focusing_efficiency = calculate_focusing_efficiency(focal_amps, expansion)
    #     invalid_area = too_thin_area(topology(x, n_samples=100), 5)
    #     return transmission_efficiency * focusing_efficiency - 4 * invalid_area
    #
    # def target_func(x):
    #     x = jnp.array(x)
    #     opt_x, max_f = run_gradient_ascent(
    #         target_function=predicted_efficiency,
    #         x_init=x,
    #         learning_rate=1e-2,
    #         n_steps=50,
    #         boundary_projection_function=lambda x: jnp.clip(
    #             x, topology.minval, topology.maxval),
    #         print_step=False
    #     )
    #     return 1 - max_f
    #
    #
    # opt_res = differential_evolution(
    #     # func=lambda x: 1 - predicted_efficiency(jnp.array(x)),
    #     func=target_func,
    #     bounds=((-1., 1.),) * topology.n_geometrical_parameters,
    #     init=np.random.rand(5, topology.n_geometrical_parameters) * 2 - 1,
    #     maxiter=100,
    #     polish=False,
    #     disp=True
    # )
    # print(opt_res)
    # x = opt_res.x
    #
    # x, max_f = run_gradient_ascent(
    #         target_function=predicted_efficiency,
    #         x_init=x,
    #         learning_rate=1e-2,
    #         n_steps=50,
    #         boundary_projection_function=lambda x: jnp.clip(
    #             x, topology.minval, topology.maxval),
    #         print_step=False
    #     )
    # print(max_f)


    def calculate_true_eff(x):
        pattern = topology(x, 100)
        amps = simulate_scattering(pattern)
        foc_eff = calculate_focusing_efficiency(amps, expansion)
        trans_eff = jnp.sum(jnp.abs(amps) ** 2)
        invalid_area = too_thin_area(pattern, 5)
        return foc_eff * trans_eff - 4 * invalid_area


    x, max_f = run_gradient_ascent(
        target_function=calculate_true_eff,
        x_init=x,
        learning_rate=5e-3,
        n_steps=100,
        boundary_projection_function=lambda x: jnp.clip(
            x, topology.minval, topology.maxval)
    )

    print(max_f)

    plt.imshow(topology(x, 500))
    plt.show()
