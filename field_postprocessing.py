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


def normalize_power(amps, relative_period, basis_indices):
    trans_amps_xy = amps[:2]
    ref_amps_xy = amps[2:]
    total_power = (
        calculate_total_power(trans_amps_xy, relative_period, basis_indices)
        + calculate_total_power(ref_amps_xy, relative_period, basis_indices)
    )
    return amps / jnp.sqrt(total_power)


def propagate_amps_in_free_space(amps, distnce, basis_indices, wavelength, period):
    k0 = 2 * np.pi / wavelength
    lattice_k = 2 * np.pi / period
    kx, ky = basis_indices * lattice_k
    kz = np.sqrt(k0 ** 2 - kx ** 2 - ky ** 2, dtype=complex)
    propagated_amps = amps * jnp.exp(1j * kz * distnce)
    return propagated_amps


def extract_amps_from_fields(fields, basis_indices):
    h, w, c = fields.shape[-3:]
    all_modes = jnp.fft.fft2(fields, axes=(-3, -2)) / (h * w)
    return all_modes[..., basis_indices[0], basis_indices[1], :]


def amps_to_field_maps(amps, basis_indices, n_samples):
    '''
    Calculates 2D field complex values from given Fourier amplitudes. amps may have a leading batch dimenstion.

    Args:
        amps : jax.Array
            Field Fourier amplitudes of shape ([batch_size,] basis_size,)
        basis_indices : np.ndarray
            Integer indices of 2D Fourier expansion with shape (2, basis_size)
        n_samples : int
            Size of resulting field map
    Returns:
        field_maps : jax.Array
            Respective complex-valued fields with shape ([batch_size,] n_samples, n_samples)
    '''

    fields_fft = jnp.zeros(amps.shape[:-1] + (n_samples, n_samples), dtype=complex)
    n, m = basis_indices
    fields_fft = fields_fft.at[..., n, m].set(amps)
    fields = jnp.fft.fft2(fields_fft)
    return fields


def amps_to_intensity_map(amps_xy, basis_indices, relative_period, n_samples=100):
    '''
        Calculates 2D intensity map from given electric field Fourier amplitudes Ex,nm and Ey,nm.

        Args:
            amps_xy : jax.Array
                Electric field Fourier amplitudes Ex,nm and Ey,nm of shape (2, basis_size)
            basis_indices : np.ndarray
                Integer indices of 2D Fourier expansion with shape (2, basis_size)
            relative_period : float
                Unit cell size relative to wavelength
            n_samples (optional) : int
                Size of resulting field map, 100 by default
        Returns:
            intensity_map : jax.Array
                Intensity with shape (n_samples, n_samples)
        '''

    kx, ky = basis_indices / relative_period
    kz = np.sqrt(1 - kx ** 2 - ky ** 2)

    ex, ey = amps_xy
    ez = (kx * ex + ky * ey) / kz
    hx = ky * ez + kz * ey
    hy = -(kz * ex + kx * ez)

    ex_map, ey_map, hx_map, hy_map = amps_to_field_maps(jnp.stack([ex, ey, hx, hy], axis=0), basis_indices, n_samples)
    intensity_map = -jnp.real(ex_map * jnp.conj(hy_map) - ey_map * jnp.conj(hx_map))
    return intensity_map
