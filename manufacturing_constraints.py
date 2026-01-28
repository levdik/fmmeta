import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp


def _circular_kernel(width_px: int):
    r = width_px / 2
    x = np.linspace(-r, r, width_px, endpoint=False) + 0.5
    x, y = np.meshgrid(x, x)
    kernel = x ** 2 + y ** 2 <= r ** 2
    kernel = kernel.astype(float)
    kernel /= np.sum(kernel)
    return kernel


def _periodic_convolution(x, kernel):
    pad_width = kernel.shape[0] // 2
    convolved = jsp.signal.convolve(jnp.pad(x, pad_width, mode='wrap'), kernel, mode='valid')
    return convolved


def detect_too_thin_solid(pattern, min_width_px):
    min_width_px = (min_width_px // 2) * 2 + 1
    kernel = _circular_kernel(min_width_px)
    pattern_binary = pattern.astype(bool).astype(float)
    narrowed = 1. - _periodic_convolution(1. - pattern_binary, kernel)
    narrowed = (narrowed == 1).astype(float)

    # plt.imshow(narrowed, cmap='gray_r')
    # plt.xticks([], [])
    # plt.yticks([], [])
    # plt.show()

    narrowed_widened = _periodic_convolution(narrowed, kernel)
    narrowed_widened = narrowed_widened.astype(bool).astype(float)

    # plt.imshow(narrowed_widened, cmap='gray_r')
    # plt.xticks([], [])
    # plt.yticks([], [])
    # plt.show()

    to_remove = jnp.abs(pattern_binary - narrowed_widened) * pattern
    return to_remove


def detect_too_thin_void(pattern, min_width_px):
    return detect_too_thin_solid(1. - pattern, min_width_px)


def too_thin_area(pattern, min_width_px):
    too_thin_solid = detect_too_thin_solid(pattern, min_width_px)
    too_thin_void = detect_too_thin_void(pattern, min_width_px)
    too_thin_relative_area = (jnp.sum(too_thin_solid) + jnp.sum(too_thin_void)) / pattern.size
    return too_thin_relative_area


if __name__ == '__main__':
    import jax
    import topology_parametrization
    from design_optimizer import run_gradient_ascent
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use('TkAgg')

    period = 2000
    min_width = 250
    n_px = 256
    min_width_px = round(n_px * min_width / period)
    min_width_px = (min_width_px // 2) * 2 + 1
    print(min_width_px)

    topology = topology_parametrization.FourierInterpolation(6)
    x_init = jax.random.uniform(jax.random.key(2), topology.n_geometrical_parameters, minval=-1, maxval=1)

    x_init = x_init.reshape(6, 6)
    x_init = x_init.at[0, :].set(-1)
    x_init = x_init.at[-1, :].set(-1)
    x_init = x_init.at[:, 0].set(-1)
    x_init = x_init.at[:, -1].set(-1)
    x_init = x_init.flatten()

    x = topology(x_init, 500)
    plt.imshow(x, cmap='gray_r')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.show()

    detect_too_thin_solid(x, 50)

    exit()

    # optimized_x, optimized_f = run_gradient_ascent(
    #     lambda x: -too_thin_area(topology(x), min_width_px),
    #     x_init,
    #     learning_rate=1e-2,
    #     n_steps=500,
    #     boundary_projection_function=lambda x: jnp.clip(x, -1, 1)
    # )
    # print(optimized_f)
    optimized_x = x_init

    x = topology(optimized_x, 500)
    print(too_thin_area(x, min_width_px))

    too_thin_solid = detect_too_thin_solid(x, min_width_px)
    too_thin_void = detect_too_thin_solid(1. - x, min_width_px)

    overlay_image = np.zeros(x.shape + (3,))
    overlay_image[()] = (1 - x)[:, :, None]
    overlay_image[:, :, 0][too_thin_solid.astype(bool)] = 1
    overlay_image[:, :, 1][too_thin_solid.astype(bool)] = 1 - too_thin_solid[too_thin_solid.astype(bool)]
    overlay_image[:, :, 2][too_thin_solid.astype(bool)] = 1 - too_thin_solid[too_thin_solid.astype(bool)]

    overlay_image[:, :, 0][too_thin_void.astype(bool)] = 0
    overlay_image[:, :, 1][too_thin_void.astype(bool)] = 0
    overlay_image[:, :, 2][too_thin_void.astype(bool)] = too_thin_void[too_thin_void.astype(bool)]

    both = too_thin_solid.astype(bool) & too_thin_void.astype(bool)
    overlay_image[:, :, 0][both] = too_thin_solid[both]
    overlay_image[:, :, 1][both] = (1 - x)[both]
    overlay_image[:, :, 2][both] = too_thin_void[both]

    plt.imshow(overlay_image)
    plt.axis('off')
    plt.show()
