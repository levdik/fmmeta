import numpy as np

import jax
import jax.numpy as jnp
import jaxopt

from scattering_solver_factory import prepare_shapes_to_amplitudes_function

import matplotlib.pyplot as plt


def find_best_fit_pillars(target_transmissions, lib_transmissions, n_wavelengths, return_distance=False):
    target_transmissions = target_transmissions.reshape(-1, n_wavelengths)
    lib_transmissions = lib_transmissions.reshape(-1, n_wavelengths)
    diff = jnp.zeros((len(target_transmissions), len(lib_transmissions)))

    for i in range(n_wavelengths):
        diff += jnp.abs(jnp.subtract.outer(target_transmissions[:, i], lib_transmissions[:, i]))

    best_fit_indices = jnp.argmin(diff, axis=1)

    if not return_distance:
        return best_fit_indices

    return best_fit_indices, diff[jnp.arange(len(target_transmissions)), best_fit_indices]


def optimize_uniform_phase_shift(target_transmissions, lib_transmissions, n_wavelengths):
    target_transmissions = target_transmissions.reshape(-1, n_wavelengths)
    lib_transmissions = lib_transmissions.reshape(-1, n_wavelengths)

    @jax.jit
    def loss(uniform_shift):
        shifted_target_transmissions = target_transmissions * jnp.exp(1j * uniform_shift.reshape(1, n_wavelengths))
        best_fit_indices, diff = find_best_fit_pillars(
            shifted_target_transmissions, lib_transmissions, n_wavelengths, return_distance=True)
        return jnp.mean(diff)

    grid = jnp.meshgrid(*([jnp.linspace(-jnp.pi, jnp.pi, 30)] * n_wavelengths))
    flat_grid = jnp.stack([grid_i.flatten() for grid_i in grid], axis=-1)
    min_loss_on_grid = 10000
    best_shifts_on_grid = None
    for shifts in flat_grid:
        current_loss_on_grid = loss(shifts)
        if current_loss_on_grid < min_loss_on_grid:
            min_loss_on_grid = current_loss_on_grid
            best_shifts_on_grid = shifts
    print('Results of uniform phase brute-force search:', min_loss_on_grid, best_shifts_on_grid)

    solver = jaxopt.LBFGS(fun=loss, maxiter=100)
    init_shifts = best_shifts_on_grid
    res = solver.run(init_shifts)
    print('After local optimization:', res.state.value, res.params)

    return res.params


def generate_transmission_library(
        shapes,
        wavelength,
        permittivity,
        period,
        lens_thickness,
        approximate_number_of_terms
):
    transmission_library = []
    shapes_to_amplitudes_func = prepare_shapes_to_amplitudes_function(
        wavelength=wavelength,
        permittivity=permittivity,
        lens_subpixel_size=period,
        n_lens_subpixels=1,
        lens_thickness=lens_thickness,
        approximate_number_of_terms=approximate_number_of_terms,
        include_reflection=False,
    )

    phase_factor = complex(-jnp.exp(1j * 2 * jnp.pi * lens_thickness / wavelength))

    for shape in shapes:
        trans_coeff = complex(shapes_to_amplitudes_func(shape)[0]) / phase_factor
        # print(shape[0], shape[1], np.abs(trans_coeff), np.angle(trans_coeff))
        transmission_library.append(trans_coeff)

    print(transmission_library)
    return transmission_library


def visualize_3d_library(t_red, t_green, t_blue):
    min_amps = np.min(np.vstack([np.abs(t_red), np.abs(t_green), np.abs(t_blue)]), axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    img = ax.scatter(
        np.angle(t_red),
        np.angle(t_green),
        np.angle(t_blue),
        c=min_amps,
        cmap='plasma_r',
        vmin=0, vmax=1
    )

    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_zlim(-np.pi, np.pi)

    ax.set_xticks([-np.pi, 0, np.pi], labels=['-π', '0', 'π'])
    ax.set_yticks([-np.pi, 0, np.pi], labels=['-π', '0', 'π'])
    ax.set_zticks([-np.pi, 0, np.pi], labels=['-π', '0', 'π'])

    ax.set_xlabel(f'Δφ red')
    ax.set_ylabel(f'Δφ green')
    ax.set_zlabel(f'Δφ blue')

    plt.colorbar(
        img,
        orientation='horizontal',
        label='Minimal transmission amplitude',
        fraction=0.046, pad=0.04
    )

    plt.show()


def visualize_3d_library_projections(t_red, t_green, t_blue):
    min_amps = np.min(np.vstack([np.abs(t_red), np.abs(t_green), np.abs(t_blue)]), axis=0)

    fig, ax = plt.subplots(1, 3, figsize=(20, 5))

    ax[0].scatter(np.angle(t_red), np.angle(t_green), c=min_amps, cmap='plasma_r', vmin=0, vmax=1)
    ax[0].set_xlabel(f'Δφ red')
    ax[0].set_ylabel(f'Δφ green')
    ax[0].set_xlim(-np.pi, np.pi)
    ax[0].set_ylim(-np.pi, np.pi)

    ax[1].scatter(np.angle(t_red), np.angle(t_blue), c=min_amps, cmap='plasma_r', vmin=0, vmax=1)
    ax[1].set_xlabel(f'Δφ red')
    ax[1].set_ylabel(f'Δφ blue')
    ax[1].set_xlim(-np.pi, np.pi)
    ax[1].set_ylim(-np.pi, np.pi)

    ax[2].scatter(np.angle(t_green), np.angle(t_blue), c=min_amps, cmap='plasma_r', vmin=0, vmax=1)
    ax[2].set_xlabel(f'Δφ green')
    ax[2].set_ylabel(f'Δφ blue')
    ax[2].set_xlim(-np.pi, np.pi)
    ax[2].set_ylim(-np.pi, np.pi)

    plt.show()


if __name__ == '__main__':
    # print(find_best_fit_pillars(
    #     target_transmissions=jnp.array([[1, 2, 3.1], [0.1, 0.1, 0.1], [0.2, 0, 0]]),
    #     lib_transmissions=jnp.array([[1, 2, 3], [0, 0, 0]]),
    #     n_wavelengths=3,
    #     return_distance=True
    # ))

    # a_range = jnp.arange(0, 300, 10)
    # shapes = jnp.vstack([a_range] + [jnp.zeros_like(a_range)] * 3).T

    # shapes = []
    # for a in range(0, 300, 10):
    #     for b in range(0, 300, 10):
    #         if a == 0 and b != 0:
    #             continue
    #         if a + 2 * b < 300:
    #             shapes.append([a, b, 0, 0])
    # shapes = jnp.array(shapes)
    #
    # for wl in [650, 550, 450]:
    #     print(wl)
    #     generate_transmission_library(
    #         # shapes=jnp.array([(0.1, 0, 0, 0)]),
    #         shapes=shapes,
    #         wavelength=wl,
    #         permittivity=4,
    #         period=300,
    #         lens_thickness=3000,
    #         approximate_number_of_terms=100
    #     )

    from translibs.translib_cross_th3000_p300 import shapes, transmissions_red, transmissions_green, transmissions_blue
    # visualize_3d_library_projections(transmissions_red, transmissions_green, transmissions_blue)

    # import matplotlib as mpl
    # mpl.use('TkAgg')
    # visualize_3d_library(transmissions_red, transmissions_green, transmissions_blue)

    lib_transmissions = jnp.vstack([transmissions_red, transmissions_green, transmissions_blue]).T
    from phase_profile_manager import generate_target_phase
    from lens_permittivity_profile_generator import generate_pillar_center_positions
    pillar_centers = generate_pillar_center_positions(lens_subpixel_size=300, n_lens_subpixels=8)
    target_phases_red = generate_target_phase(points=pillar_centers, focal_points=[-600, -600, 4000], wavelength=650)
    target_phases_green = generate_target_phase(points=pillar_centers, focal_points=[[-600, 600, 4000], [600, -600, 4000]], wavelength=550, xy_period=8 * 300)
    target_phases_blue = generate_target_phase(points=pillar_centers, focal_points=[600, 600, 4000], wavelength=450)
    target_transmissions = jnp.exp(1j * jnp.vstack([target_phases_red, target_phases_green, target_phases_blue]).T)

    uniform_target_shifts = optimize_uniform_phase_shift(target_transmissions, lib_transmissions, n_wavelengths=3)
    target_transmissions = target_transmissions * jnp.exp(1j * uniform_target_shifts.reshape(1, 3))
    best_fit_pillar_indices = find_best_fit_pillars(target_transmissions, lib_transmissions, n_wavelengths=3)

    fig, ax = plt.subplots(1, 3, figsize=(30, 10))
    for i, ax_i in enumerate(ax):
        ax[i].plot(jnp.angle(target_transmissions[:, i]), '--')
        ax[i].plot(jnp.angle(lib_transmissions[:, i][best_fit_pillar_indices]), 'o')
    plt.show()
