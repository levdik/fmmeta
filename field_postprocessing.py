import jax
import jax.numpy as jnp
import numpy as np

from fmmax import fields, basis


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


def intensity_map_from_fourier_amplitudes(amplitudes, basis_indices, n_samples=100):
    single_coordinate_samples = np.linspace(0, 1, n_samples)
    x_mesh, y_mesh = np.meshgrid(single_coordinate_samples, single_coordinate_samples)
    field_map = np.zeros((n_samples, n_samples), dtype=complex)

    for a, (n, m) in zip(amplitudes, basis_indices):
        field_map += a * np.exp(1j * 2 * np.pi * (n * x_mesh + m * y_mesh))

    return np.abs(field_map) ** 2


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


def find_focusing_efficiency_eigenstates(
        basis_indices,
        relative_focal_point=(0.5, 0.5),
        max_number_to_return=None,
        min_eigenvalue_to_return=0.
):
    matrix = _generate_focusing_efficiency_quadratic_form_matrix(
        basis_indices, relative_focal_point=relative_focal_point
    )
    eigenvalues, eigenvectors = jnp.linalg.eig(matrix)
    eigenvalues = eigenvalues.real
    sorted_ind = jnp.argsort(eigenvalues)
    eigenvalues, eigenvectors = eigenvalues[sorted_ind], eigenvectors[sorted_ind]

    threshold_index = jnp.searchsorted(eigenvalues, min_eigenvalue_to_return)
    n_above_threshold = len(eigenvalues) - int(threshold_index)
    if max_number_to_return is None:
        max_number_to_return = len(eigenvalues)
    max_number_to_return = min(n_above_threshold, max_number_to_return)

    eigenvalues, eigenvectors = eigenvalues[-max_number_to_return:], eigenvectors[-max_number_to_return:]

    return eigenvalues[::-1], eigenvectors[::-1]


def generate_high_focusing_efficiency_target_on_lens_amplitudes():
    pass


def _calculate_max_theoretical_focusing_efficiencies(max_period_relative_to_wavelength):
    n_max = np.floor(max_period_relative_to_wavelength)
    n, m = np.meshgrid(np.arange(-n_max, n_max + 1), np.arange(-n_max, n_max + 1))
    n, m = n.flatten(), m.flatten()
    r = n ** 2 + m ** 2
    r_max = max_period_relative_to_wavelength ** 2
    sorted_indices = np.argsort(r)[:np.count_nonzero(r <= r_max)]
    n, m, r = (x[sorted_indices] for x in [n, m, r])
    unique_r_limits, = np.nonzero(np.diff(r, append=np.inf))

    unique_max_efficiencies = []
    for max_index in unique_r_limits:
        basis = tuple(zip(n[:max_index + 1], m[:max_index + 1]))
        max_eigenvalue, _ = find_focusing_efficiency_eigenstates(basis, max_number_to_return=1)
        unique_max_efficiencies.append(float(max_eigenvalue[0]))
    unique_periods = np.sqrt(r[unique_r_limits])
    return unique_periods, unique_max_efficiencies


def _calculate_example_max_eff_and_plot():
    periods, efficiencies = _calculate_max_theoretical_focusing_efficiencies(4)
    import matplotlib.pyplot as plt
    plt.plot([0, max(periods)], [1, 1], '--')
    plt.step(periods, efficiencies, where='post')
    plt.xlabel('Period relative to Wavelength')
    plt.ylabel('Maximal far-field focusing efficiency')
    plt.grid()
    plt.show()


def _examine_efficiency_eigenvalues():
    p = 7 * 300 / 650
    basis = []
    max_n = int(np.floor(p))
    for n in range(-max_n, max_n + 1):
        for m in range(-max_n, max_n + 1):
            if n ** 2 + m ** 2 < p ** 2:
                basis.append((n, m))

    eigenvalues, eigenvectors = find_focusing_efficiency_eigenstates(
        basis_indices=basis,
        max_number_to_return=None,
        min_eigenvalue_to_return=0.
    )

    import matplotlib.pyplot as plt

    print(len(eigenvalues))
    # print(eigenvalues)
    # print(jnp.abs(eigenvectors))

    # print(jnp.abs(eigenvectors[:, 0]))
    # return

    # plt.plot(eigenvalues, 'o')
    # plt.show()

    for eigenvector in eigenvectors[:3]:
        plt.plot(jnp.abs(eigenvector), 'o')
    plt.show()

    for eigenvector in eigenvectors[:3]:
        e0 = eigenvector[0] / jnp.abs(eigenvector[0])
        eigenvector /= e0
        plt.plot(jnp.angle(eigenvector), 'o')
    plt.show()


def min_distance_between_amplitude_vectors(x, y):
    return jnp.linalg.norm(x) ** 2 + jnp.linalg.norm(y) ** 2 - 2 * jnp.abs(jnp.dot(x, jnp.conj(y)))


def min_difference_between_amplitude_vectors(x, y):
    arg_min = jnp.angle(jnp.dot(x, jnp.conj(y)))
    return jnp.abs(jnp.exp(1j * arg_min) * x - y)

# def min_weighted_distance_between_amplitude_vectors(x, y):
#     n_waves = x.shape[-1]
#     expansion = basis.generate_expansion(
#         primitive_lattice_vectors=basis.LatticeVectors(u=basis.X, v=basis.Y),
#         approximate_num_terms=n_waves,
#     )
#     basis_indices = expansion.basis_coefficients
#     diff = min_difference_between_amplitude_vectors(x, y)
#     n, m = basis_indices.T
#     weights = 1 / (jnp.sqrt(n ** 2 + m ** 2) + 1)
#     mean_weighted_distance = jnp.sum(diff * weights) / jnp.sum(weights)
#     return mean_weighted_distance


# def arg_min_distance_between_amplitude_vectors(x, y):
#     return jnp.angle(jnp.dot(x, jnp.conj(y)))


if __name__ == '__main__':
    jnp.set_printoptions(linewidth=1000)

    # _test_intensity_map()
    # _test_focusing_efficiency()
    # _calculate_example_max_eff_and_plot()
    # _examine_efficiency_eigenvalues()
