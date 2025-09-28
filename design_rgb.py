import jax
import jax.numpy as jnp
import optax
from matplotlib import pyplot as plt

from scattering_solver_factory import prepare_shapes_to_amplitudes_function
from field_postprocessing import calculate_focusing_efficiency
from field_postprocessing import intensity_map_from_fourier_amplitudes
from lens_permittivity_profile_generator import generate_lens_permittivity_map
from field_plotter import plot_amplitude_map

n_lens_subpixels = 8
n_unique_shapes = 10
symmetry_indices = jnp.array([
    [1, 2, 2, 1, 8, 9, 9, 8],
    [3, 0, 0, 3, 9, 7, 7, 9],
    [3, 0, 0, 3, 9, 7, 7, 9],
    [1, 2, 2, 1, 8, 9, 9, 8],
    [5, 6, 6, 5, 1, 3, 3, 1],
    [6, 4, 4, 6, 2, 0, 0, 2],
    [6, 4, 4, 6, 2, 0, 0, 2],
    [5, 6, 6, 5, 1, 3, 3, 1]
])

# symmetry_indices = jnp.unique(
#     jnp.arange(n_lens_subpixels ** 2).reshape(n_lens_subpixels, -1) * jnp.tri(n_lens_subpixels),
#     return_inverse=True)[1]
# symmetry_indices += symmetry_indices.T * (1 - jnp.tri(n_lens_subpixels))
# symmetry_indices = symmetry_indices.astype(int)
# n_unique_shapes = len(jnp.unique(symmetry_indices))

permittivity = 4
lens_subpixel_size = 300
focal_length = 4000
approximate_number_of_terms = 300
total_lens_period = n_lens_subpixels * lens_subpixel_size

lens_thickness = 1700

wavelengths = (450, 550, 650)
relative_focal_points = ((0.75, 0.25), ((0.25, 0.25), (0.75, 0.75)), (0.25, 0.75))


def params_to_shapes(params):
    a_unique, b_unique = params * lens_subpixel_size
    a, b = a_unique[symmetry_indices], b_unique[symmetry_indices]
    ah = bh = jnp.zeros_like(a)
    shapes = jnp.stack([a, b, ah, bh], axis=-1)
    return shapes


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
    wavelength=wavelengths[2], **common_func_prep_kwargs
)
green_shapes_to_amps_function, green_basis_indices = prepare_shapes_to_amplitudes_function(
    wavelength=wavelengths[1], **common_func_prep_kwargs
)
blue_shapes_to_amps_function, blue_basis_indices = prepare_shapes_to_amplitudes_function(
    wavelength=wavelengths[0], **common_func_prep_kwargs
)


def design_lens():
    def project_params_onto_boundaries(params):
        a, b = params
        a = jnp.clip(a, 0, 1)
        b = jnp.clip(b, 0, (1 - a) / 2)
        return jnp.stack([a, b])

    def red_focusing_efficiency_function(params):
        shapes = params_to_shapes(params)
        focal_plane_amps = red_shapes_to_amps_function(shapes)
        focusing_efficiency = calculate_focusing_efficiency(
            focal_plane_amps, red_basis_indices, relative_focal_points[2])
        transmission_efficiency = jnp.sum(jnp.abs(focal_plane_amps) ** 2)
        total_efficiency = transmission_efficiency * focusing_efficiency
        return total_efficiency

    def green_focusing_efficiency_function(params):
        shapes = params_to_shapes(params)
        focal_plane_amps = green_shapes_to_amps_function(shapes)
        focusing_efficiency = (
            calculate_focusing_efficiency(
                focal_plane_amps, green_basis_indices, relative_focal_points[1][0])
            + calculate_focusing_efficiency(
                focal_plane_amps, green_basis_indices, relative_focal_points[1][1])
        )
        transmission_efficiency = jnp.sum(jnp.abs(focal_plane_amps) ** 2)
        total_efficiency = transmission_efficiency * focusing_efficiency
        return total_efficiency

    def blue_focusing_efficiency_function(params):
        shapes = params_to_shapes(params)
        focal_plane_amps = blue_shapes_to_amps_function(shapes)
        focusing_efficiency = calculate_focusing_efficiency(
            focal_plane_amps, blue_basis_indices, relative_focal_points[0])
        transmission_efficiency = jnp.sum(jnp.abs(focal_plane_amps) ** 2)
        total_efficiency = transmission_efficiency * focusing_efficiency
        return total_efficiency

    def loss_fn(params):
        overall_efficiency = (
            red_focusing_efficiency_function(params)
            + green_focusing_efficiency_function(params)
            + blue_focusing_efficiency_function(params)
        ) / 3
        # overall_efficiency = (
        #         red_focusing_efficiency_function(params)
        #         + blue_focusing_efficiency_function(params)
        # ) / 2
        return -overall_efficiency

    optimizer = optax.adam(learning_rate=1e-2)

    # init_params = jnp.stack([jnp.ones(n_unique_shapes) / 2.5, jnp.ones(n_unique_shapes) / 10])
    init_params = jnp.stack([
        jnp.array([109, 93, 93, 110, 91, 118, 96, 119, 90, 111]),
        jnp.array([20, 3, 0, 22, 0, 31, 3, 37, 0, 23])
    ]) / 300
    optimizer_state = optimizer.init(init_params)

    @jax.jit
    def descent_step(params, optimizer_state):
        loss, grad = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grad, optimizer_state)
        params = optax.apply_updates(params, updates)
        params = project_params_onto_boundaries(params)
        return params, opt_state, loss, grad

    params = init_params
    max_eff = 0
    best_params = None

    try:
        for i in range(100):
            new_params, optimizer_state, loss, grad = descent_step(params, optimizer_state)
            avg_grad_norm = jnp.linalg.norm(grad) / params.size
            rounded_params = jnp.round(params * lens_subpixel_size).astype(int)
            print(f"Step {i}: efficiency={-loss:.4f}, a={rounded_params[0]}, b={rounded_params[1]}, |grad|={avg_grad_norm}")

            if -loss > max_eff:
                max_eff = -loss
                best_params = params

            if avg_grad_norm < 0.001:
                print("Stop on gradient norm criteria")
                break
            params = new_params
    except KeyboardInterrupt:
        pass

    print('a')
    print(repr(jnp.round(params[0] * lens_subpixel_size).astype(int)[symmetry_indices]))
    print('b')
    print(repr(jnp.round(params[1] * lens_subpixel_size).astype(int)[symmetry_indices]))

    return best_params


def show_lens_and_intensity(params):
    shapes = params_to_shapes(params)
    red_focal_plane_amps = red_shapes_to_amps_function(shapes)
    green_focal_plane_amps = green_shapes_to_amps_function(shapes)
    blue_focal_plane_amps = blue_shapes_to_amps_function(shapes)
    focal_plane_amps = [red_focal_plane_amps, green_focal_plane_amps, blue_focal_plane_amps]
    basis_indices = [red_basis_indices, green_basis_indices, blue_basis_indices]

    fig, ax = plt.subplots(2, 2, )
    intensity_ax = [ax[0, 1], ax[1, 0], ax[1, 1]]

    lens_permittivity_map = generate_lens_permittivity_map(
        shapes=shapes, sub_pixel_size=lens_subpixel_size, n_lens_subpixels=n_lens_subpixels,
        permittivity_pillar=permittivity
    )
    ax[0, 0].imshow(lens_permittivity_map, extent=(0, total_lens_period, 0, total_lens_period))
    ax[0, 0].set_axis_off()

    for i in range(3):
        intensity = intensity_map_from_fourier_amplitudes(
            amplitudes=focal_plane_amps[i],
            basis_indices=basis_indices[i]
        )
        # intensity = (intensity + jnp.rot90(intensity, 2).T) / 2

        plot_amplitude_map(
            fig, intensity_ax[i], intensity,
            wavelength_nm=wavelengths[2 - i],
            map_bounds=[0, total_lens_period],
            cbar=False
        )
        intensity_ax[i].set_axis_off()

    plt.show()


if __name__ == '__main__':
    jnp.set_printoptions(linewidth=1000)

    optimized_params = design_lens()
    show_lens_and_intensity(optimized_params)
