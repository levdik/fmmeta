import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp

from ai_fno import RealToComplexFNO, PatternToAmpsFNO
from ai_cnn_trans import MetasurfaceTransmissionCNN, HybridFNOCNN
import ai_fno
from field_postprocessing import calculate_focusing_efficiency, propagate_amps_in_free_space
from design_optimizer import run_gradient_ascent
from lens_topology_parametrization import FourierInterpolationTP, FourierExpansionTP, BicubicInterpolationTP
from lens_topology_parametrization import SquarePillarTP
from scattering_simulation import prepare_lens_scattering_solver

import pickle
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


def load_fno_model(state_restored=None):
    if state_restored is None:
        with open('ai_models_fno/fno_red_pattern_to_amps_not_normalized.pkl', 'rb') as f:
            state_restored = pickle.load(f)

    model = PatternToAmpsFNO(
        hidden_n_channels=[64] * 5,
        n_pixels=64,
        mode_threshold=16,
        activation_fn=nnx.gelu,
        rngs=nnx.Rngs(0)
    )

    model.fno.lifting.kernel = nnx.Param(state_restored.fno.lifting.kernel.value)
    model.fno.projection.kernel = nnx.Param(state_restored.fno.projection.kernel.value)
    for i in range(len(state_restored.fno.fourier_layers)):
        model.fno.fourier_layers[i].fourier_linear_block.w_re = nnx.Param(
            state_restored.fno.fourier_layers[i].fourier_linear_block.w_re.value)
        model.fno.fourier_layers[i].fourier_linear_block.w_im = nnx.Param(
            state_restored.fno.fourier_layers[i].fourier_linear_block.w_im.value)
        model.fno.fourier_layers[i].bypass_convolution.kernel = nnx.Param(
            state_restored.fno.fourier_layers[i].bypass_convolution.kernel.value)

    return model


def load_cnn_trans_model(state_restored=None):
    abstract_model = nnx.eval_shape(lambda: MetasurfaceTransmissionCNN(rngs=nnx.Rngs(0)))
    graphdef, abstract_state = nnx.split(abstract_model)

    if state_restored is None:
        with open('ai_models_fno/trans_cnn_model.pkl', 'rb') as f:
            state_restored = pickle.load(f)

    model = nnx.merge(graphdef, state_restored)
    return model


def load_hybrid_model():
    with open('ai_models_fno/fno_cnn_hybrid.pkl', 'rb') as f:
        state_restored = pickle.load(f)

    fno = load_fno_model(state_restored.fno)
    cnn = load_cnn_trans_model(state_restored.cnn)
    model = HybridFNOCNN(fno, cnn)

    return model


if __name__ == '__main__':
    # model = load_hybrid_model()
    model = load_fno_model()
    # trans_cnn = load_cnn_trans_model()

    topology_parametrization = FourierInterpolationTP(grid_size=10, symmetry_type='main_diagonal')
    # topology_parametrization = FourierExpansionTP(r_max=7, symmetry_type='central')
    # topology_parametrization = BicubicInterpolationTP(grid_size=10, symmetry_type='main_diagonal')
    # topology_parametrization = SquarePillarTP(grid_size=7, symmetry_type='central')

    x = jax.random.uniform(
        jax.random.key(1),
        shape=(topology_parametrization.n_geometrical_parameters,),
        minval=topology_parametrization.minval / 2,
        maxval=topology_parametrization.maxval / 2
    )
    # x = 0.5 * jnp.ones(topology_parametrization.n_geometrical_parameters)

    simulate_scattering, expansion = prepare_lens_scattering_solver(
        wavelength=650,
        period=2000,
        lens_thickness=600,
        substrate_thickness=500,
        approximate_number_of_terms=300,
        propagate_by_distance=2500
    )


    # pattern = topology_parametrization(x, n_samples=64)
    # y = model(pattern[None, :])[0]
    # y_filtered = np.zeros((64, 64), dtype=complex)
    # y_filtered[expansion[:, 0], expansion[:, 1]] = np.fft.fft2(y)[expansion[:, 0], expansion[:, 1]]
    # y_filtered = np.fft.ifft2(y_filtered)

    # true_amps = simulate_scattering(pattern)
    # true_y = np.zeros((64, 64), dtype=complex)
    # true_y[expansion[:, 0], expansion[:, 1]] = true_amps
    # true_y = np.fft.ifft2(true_y) * (64 ** 2)

    # fig, ax = plt.subplots(1, 4)
    # ax[0].imshow(pattern)
    # ax[1].imshow(y.real, vmin=-1, vmax=1)
    # ax[2].imshow(y_filtered.real, vmin=-1, vmax=1)
    # ax[3].imshow(true_y.real, vmin=-1, vmax=1)
    #
    # for ax_i in ax.flatten():
    #     ax_i.set_axis_off()
    # plt.show()

    def predicted_efficiency(x):
        pattern = topology_parametrization(x, n_samples=64)
        y = model(pattern[None, :])[0]
        # amps = jnp.fft.fft2(y)[expansion[:, 0], expansion[:, 1]] / (64 ** 2)
        amps = y
        focal_amps = propagate_amps_in_free_space(amps, 2500, expansion, 650, 2000)
        transmission_efficiency = jnp.sum(jnp.abs(focal_amps) ** 2)
        # transmission_efficiency = jnp.clip(transmission_efficiency, 0, 1)
        # transmission_efficiency = trans_cnn(pattern)
        focusing_efficiency = calculate_focusing_efficiency(focal_amps, expansion)
        return transmission_efficiency * focusing_efficiency


    optimized_x, optimized_eff = run_gradient_ascent(
        target_function=predicted_efficiency,
        x_init=x,
        learning_rate=1e-2,
        n_steps=300,
        boundary_projection_function=lambda x: jnp.clip(
            x, topology_parametrization.minval, topology_parametrization.maxval)
    )

    # def calculate_true_eff(x):
    #     pattern = topology_parametrization(x)
    #     amps = simulate_scattering(pattern)
    #     foc_eff = calculate_focusing_efficiency(amps, expansion)
    #     trans_eff = jnp.sum(jnp.abs(amps) ** 2)
    #     return foc_eff * trans_eff
    #
    # reoptimized_x, reoptimized_eff = run_gradient_ascent(
    #     target_function=calculate_true_eff,
    #     x_init=optimized_x,
    #     learning_rate=1e-2,
    #     n_steps=100,
    #     boundary_projection_function=lambda x: jnp.clip(x, -1, 1)
    # )

    optimized_pattern = topology_parametrization(optimized_x, n_samples=64)
    pred_amps = model(optimized_pattern[None, :])[0]
    pred_amps = propagate_amps_in_free_space(pred_amps, 2500, expansion, 650, 2000)
    pred_foc_eff = calculate_focusing_efficiency(pred_amps, expansion)
    pred_trans_eff = np.sum(np.abs(pred_amps) ** 2)
    # pred_trans_eff = trans_cnn(optimized_pattern)
    print(pred_foc_eff, pred_trans_eff, pred_foc_eff * pred_trans_eff)

    true_amps = simulate_scattering(topology_parametrization(optimized_x))
    true_foc_eff = calculate_focusing_efficiency(true_amps, expansion)
    true_trans_eff = np.sum(np.abs(true_amps) ** 2)
    print(true_foc_eff, true_trans_eff, true_foc_eff * true_trans_eff)
