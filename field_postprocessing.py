import jax
import jax.numpy as jnp
import numpy as np
from functools import partial


def _generate_focusing_efficiency_quadratic_form_matrix(basis_indices, relative_focal_point):
    n_harmonics = len(basis_indices)

    x_phase_shift = 2 * np.pi * (relative_focal_point[0] - 1 / 4)
    y_phase_shift = 2 * np.pi * (relative_focal_point[1] - 1 / 4)

    matrix = (1 / 4) * np.eye(n_harmonics, dtype=complex)

    for i, (n, m) in enumerate(basis_indices):
        for j, (n1, m1) in enumerate(basis_indices):
            dn = n - n1
            dm = m - m1
            phase_shift_factor = np.exp(1j * (x_phase_shift * dn + y_phase_shift * dm))
            if dn == 0 and dm % 2 == 1:
                matrix[i, j] = phase_shift_factor * 1j / (2 * jnp.pi * dm)
            elif dm == 0 and dn % 2 == 1:
                matrix[i, j] = phase_shift_factor * 1j / (2 * jnp.pi * dn)
            elif dn % 2 == 1 and dm % 2 == 1:
                matrix[i, j] = -phase_shift_factor / (jnp.pi * jnp.pi * dn * dm)

    return jnp.array(matrix)


# @partial(jax.jit, static_argnames=('basis_indices', 'relative_focal_point'))
def calculate_focusing_efficiency(amplitudes, basis_indices, relative_focal_point=jnp.array([0.5, 0.5])):
    quadratic_form_matrix = _generate_focusing_efficiency_quadratic_form_matrix(basis_indices, relative_focal_point)
    focused_power = jnp.dot(jnp.dot(amplitudes, quadratic_form_matrix), jnp.conj(amplitudes)).real
    total_power = jnp.sum(jnp.abs(amplitudes) ** 2)
    return focused_power / total_power


def make_jit_focusing_efficiency_function(basis_indices, relative_focal_point=(0.5, 0.5)):
    quadratic_form_matrix = _generate_focusing_efficiency_quadratic_form_matrix(basis_indices, relative_focal_point)

    def focusing_efficiency_function(amplitudes):
        focused_power = jnp.dot(jnp.dot(amplitudes, quadratic_form_matrix), jnp.conj(amplitudes)).real
        total_power = jnp.sum(jnp.abs(amplitudes) ** 2)
        return focused_power / total_power

    return jax.jit(focusing_efficiency_function)


# def calculate_focusing_efficiency_from_field_map(field_map):
#     n_samples = field_map.shape[0]
#     intensity_map = jnp.abs(field_map) ** 2
#     total_power = jnp.sum(intensity_map)
#     focused_power = jnp.sum(intensity_map[
#                             n_samples // 4:(3 * n_samples) // 4,
#                             n_samples // 4:(3 * n_samples) // 4
#                             # :n_samples // 2,
#                             # :n_samples // 2
#                             ])
#     return focused_power / total_power


def _calculate_max_theoretical_focusing_efficiency(basis_indices):
    matrix = _generate_focusing_efficiency_quadratic_form_matrix(
        basis_indices, relative_focal_point=(0.25, 0.25)
    )
    eigenvalues, _ = jnp.linalg.eig(matrix)
    return float(jnp.max(eigenvalues.real))


def _calculate_max_theoretical_focusing_efficiencies(max_period_relative_to_wavelength):
    n_max = np.floor(max_period_relative_to_wavelength)
    n, m = np.meshgrid(np.arange(-n_max, n_max + 1), np.arange(-n_max, n_max + 1))
    n, m = n.flatten(), m.flatten()
    r = n ** 2 + m ** 2
    r_max = max_period_relative_to_wavelength ** 2
    sorted_indices = np.argsort(r)[:np.count_nonzero(r <= r_max)]
    n, m, r = (x[sorted_indices] for x in [n, m, r])
    unique_r_limits, = np.nonzero(np.diff(r, append=np.inf))

    unique_max_efficiencies = [
        _calculate_max_theoretical_focusing_efficiency(tuple(zip(n[:max_index + 1], m[:max_index + 1])))
        for max_index in unique_r_limits
    ]
    unique_periods = np.sqrt(r[unique_r_limits])
    return unique_periods, unique_max_efficiencies


def intensity_map_from_fourier_amplitudes(amplitudes, basis_indices, n_samples=100):
    single_coordinate_samples = np.linspace(0, 1, n_samples)
    x_mesh, y_mesh = np.meshgrid(single_coordinate_samples, single_coordinate_samples)
    field_map = np.zeros((n_samples, n_samples), dtype=complex)

    for a, (n, m) in zip(amplitudes, basis_indices):
        field_map += a * np.exp(1j * 2 * np.pi * (n * x_mesh + m * y_mesh))

    return np.abs(field_map) ** 2


def _test_focusing_efficiency():
    p = 1.1
    n_max = int(np.floor(p))
    basis = []
    for n in range(-n_max, n_max + 1):
        for m in range(-n_max, n_max + 1):
            if n ** 2 + m ** 2 < p ** 2:
                basis.append((n, m))
    basis = tuple(basis)
    print(basis)

    amps = jnp.array([0., 0., 1., 0., 0.])
    eff_func = make_focusing_efficiency_function(basis)
    eff = eff_func(amps)
    print(eff)


def _test_intensity_map():
    p = 1.1
    n_max = int(np.floor(p))
    basis = []
    for n in range(-n_max, n_max + 1):
        for m in range(-n_max, n_max + 1):
            if n ** 2 + m ** 2 < p ** 2:
                basis.append((n, m))
    basis = tuple(basis)
    print(basis)

    amps = np.array([-1., 1., 1.4j, 1., -1.])
    intensity = intensity_map_from_fourier_amplitudes(amps, basis)
    import matplotlib.pyplot as plt
    plt.imshow(intensity)
    plt.colorbar()
    plt.show()


def _calculate_example_max_eff_and_plot():
    periods, efficiencies = _calculate_max_theoretical_focusing_efficiencies(4)
    import matplotlib.pyplot as plt
    plt.plot([0, max(periods)], [1, 1], '--')
    plt.step(periods, efficiencies, where='post')
    plt.xlabel('Period relative to Wavelength')
    plt.ylabel('Maximal far-field focusing efficiency')
    plt.grid()
    plt.show()


def min_distance_between_amplitude_vectors(x, y):
    return jnp.linalg.norm(x) ** 2 + jnp.linalg.norm(y) ** 2 - 2 * jnp.abs(jnp.dot(x, jnp.conj(y)))


if __name__ == '__main__':
    _test_intensity_map()
    # _test_focusing_efficiency()
    # _calculate_example_max_eff_and_plot()
