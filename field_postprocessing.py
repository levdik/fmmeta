import jax
import jax.numpy as jnp
import numpy as np


def calculate_total_power(amps_xy, relative_period, basis_indices):
    kx, ky = basis_indices / relative_period
    kz = np.sqrt(1 - kx ** 2 - ky ** 2)

    ax, ay = amps_xy
    az = (ax * kx + ay * ky) / kz
    modewise_a_norm = jnp.abs(ax) ** 2 + jnp.abs(ay) ** 2 + jnp.abs(az) ** 2

    total_power = jnp.sum(modewise_a_norm * kz)
    return total_power


def calculate_focused_power(amps_xy, relative_period, basis_indices, relative_focal_point=(0.5, 0.5)):
    n_modes = basis_indices.shape[-1]

    x_shift = relative_focal_point[0] - 0.25
    y_shift = relative_focal_point[1] - 0.25

    ax, ay = amps_xy
    a = jnp.concat([ax, ay])

    mode_n, mode_m = basis_indices
    kx, ky = basis_indices / relative_period
    kz = np.sqrt(1 - kx ** 2 - ky ** 2)

    n1, n2 = np.meshgrid(mode_n, mode_n)
    m1, m2, = np.meshgrid(mode_m, mode_m)
    dn = n1 - n2
    dm = m1 - m2
    theta_n = np.where(dn == 0, 1 / 2, np.where(dn % 2 == 1, 1j / (np.pi * np.where(dn == 0, 1, dn)), 0))
    theta_m = np.where(dm == 0, 1 / 2, np.where(dm % 2 == 1, 1j / (np.pi * np.where(dm == 0, 1, dm)), 0))
    theta = theta_n * theta_m
    theta *= np.exp(2j * np.pi * (x_shift * dn + y_shift * dm))

    Fxx = theta * (kz[None, :] + kx[None, :] ** 2 / kz[None, :])
    Fyy = theta * (kz[None, :] + ky[None, :] ** 2 / kz[None, :])
    Fxy = theta * (kx[None, :] * ky[None, :] / kz[None, :])
    F = np.block([[Fxx, Fxy], [Fxy, Fyy]])
    F = (F + F.conj().T) / 2

    focused_power = (a.conj() @ F @ a).real
    return focused_power


def propagate_amps_in_free_space(amps, distnce, basis_indices, wavelength, period):
    # TODO: check how it works with amps_xy
    k0 = 2 * np.pi / wavelength
    lattice_k = 2 * np.pi / period
    kx, ky = basis_indices * lattice_k
    kz = np.sqrt(k0 ** 2 - kx ** 2 - ky ** 2, dtype=complex)
    propagated_amps = amps * jnp.exp(1j * kz * distnce)
    return propagated_amps
