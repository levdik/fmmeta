import jax
import jax.numpy as jnp
import optax

from scattering_solver_factory import prepare_shapes_to_amplitudes_function
from field_postprocessing import calculate_focusing_efficiency


color_names = ('red', 'green', 'blue')
wavelengths = (650, 550, 450)
bayer_relative_focal_points = [
    ((0.25, 0.25),),
    ((0.25, 0.75), (0.75, 0.25)),
    ((0.75, 0.75),)
]

n_lens_subpixels = 8
n_unique_shapes = 9
symmetry_indices = jnp.array([
    [5, 8, 8, 5, 4, 7, 7, 4],
    [8, 2, 2, 8, 7, 1, 1, 7],
    [8, 2, 2, 8, 7, 1, 1, 7],
    [5, 8, 8, 5, 4, 7, 7, 4],
    [3, 6, 6, 3, 5, 8, 8, 5],
    [6, 0, 0, 6, 8, 2, 2, 8],
    [6, 0, 0, 6, 8, 2, 2, 8],
    [3, 6, 6, 3, 5, 8, 8, 5]
])

permittivity = 4
lens_subpixel_size = 300
# lens_thickness = 1000
focal_length = 4000
approximate_number_of_terms = 500


def design_by_gradient_descent(lens_thickness):
    def params_to_shapes(params):
        a_unique, b_unique = params * lens_subpixel_size
        a, b = a_unique[symmetry_indices], b_unique[symmetry_indices]
        ah = bh = jnp.zeros_like(a)
        shapes = jnp.stack([a, b, ah, bh], axis=-1)
        return shapes

    def project_params_onto_boundaries(params):
        a, b = params
        a = jnp.clip(a, 0, 1)
        b = jnp.clip(b, 0, (1 - a) / 2)
        return jnp.stack([a, b])

    common_func_prep_kwargs = {
        'permittivity': permittivity,
        'lens_subpixel_size': lens_subpixel_size,
        'n_lens_subpixels': n_lens_subpixels,
        'lens_thickness': lens_thickness,
        'approximate_number_of_terms': approximate_number_of_terms,
        'include_reflection': False,
        'return_basis_indices': True,
        'propagate_transmitted_amps_by_distance': focal_length
    }

    red_shapes_to_amps_function, red_basis_indices = prepare_shapes_to_amplitudes_function(
        wavelength=wavelengths[0], **common_func_prep_kwargs)
    green_shapes_to_amps_function, green_basis_indices = prepare_shapes_to_amplitudes_function(
        wavelength=wavelengths[1], **common_func_prep_kwargs)
    blue_shapes_to_amps_function, blue_basis_indices = prepare_shapes_to_amplitudes_function(
        wavelength=wavelengths[2], **common_func_prep_kwargs)

    def red_focusing_efficiency_function(params):
        shapes = params_to_shapes(params)
        focal_plane_amps = red_shapes_to_amps_function(shapes)
        focusing_efficiency = calculate_focusing_efficiency(
            focal_plane_amps, red_basis_indices, bayer_relative_focal_points[0][0])
        transmission_efficiency = jnp.sum(jnp.abs(focal_plane_amps) ** 2)
        return focusing_efficiency * transmission_efficiency

    def green_focusing_efficiency_function(params):
        shapes = params_to_shapes(params)
        focal_plane_amps = green_shapes_to_amps_function(shapes)
        focusing_efficiency = calculate_focusing_efficiency(
            focal_plane_amps, green_basis_indices, bayer_relative_focal_points[1][0])
        transmission_efficiency = jnp.sum(jnp.abs(focal_plane_amps) ** 2)
        return focusing_efficiency * transmission_efficiency

    def blue_focusing_efficiency_function(params):
        shapes = params_to_shapes(params)
        focal_plane_amps = blue_shapes_to_amps_function(shapes)
        focusing_efficiency = calculate_focusing_efficiency(
            focal_plane_amps, blue_basis_indices, bayer_relative_focal_points[2][0])
        transmission_efficiency = jnp.sum(jnp.abs(focal_plane_amps) ** 2)
        return focusing_efficiency * transmission_efficiency

    def loss_function(params):
        overall_efficiency = (red_focusing_efficiency_function(params)
                              + green_focusing_efficiency_function(params)
                              + blue_focusing_efficiency_function(params)) / 3
        return -overall_efficiency

    optimizer = optax.adam(learning_rate=0.01, b1=0.5)

    init_params = jnp.stack([jnp.ones(n_unique_shapes) / 3, jnp.ones(n_unique_shapes) / 6])
    optimizer_state = optimizer.init(init_params)

    @jax.jit
    def descent_step(params, optimizer_state):
        loss, grad = jax.value_and_grad(loss_function)(params)
        updates, opt_state = optimizer.update(grad, optimizer_state)
        params = optax.apply_updates(params, updates)
        params = project_params_onto_boundaries(params)
        return params, opt_state, loss, grad


    params = init_params
    for i in range(31):
        new_params, optimizer_state, loss, grad = descent_step(params, optimizer_state)
        avg_grad_norm = jnp.linalg.norm(grad) / params.size
        rounded_params = jnp.round(params * lens_subpixel_size).astype(int)
        print(f"Step {i}: efficiency={-loss:.4f}, a={rounded_params[0]}, b={rounded_params[1]}, |grad|={avg_grad_norm}")
        if avg_grad_norm < 0.001:
            print("Stop on gradient norm criteria")
            break
        params = new_params


if __name__ == '__main__':
    jnp.set_printoptions(linewidth=1000)

    design_by_gradient_descent(1000)
