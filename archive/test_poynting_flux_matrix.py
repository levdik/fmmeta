import numpy as np
from fmmax import basis, fields, fmm

import matplotlib.pyplot as plt
import matplotlib

np.set_printoptions(linewidth=10000)
matplotlib.use('TkAgg')


def calculate_n_propagating_waves(relative_period):
    r = relative_period
    n = np.sum(2 * np.floor(np.sqrt(r ** 2 - np.arange(-int(r), int(r) + 1) ** 2)) + 1)
    return int(n)


if __name__ == '__main__':
    period_nm = 2000
    wavelength_nm = 650
    p = period_nm / wavelength_nm
    n_propagating = calculate_n_propagating_waves(p)

    mode_n, mode_m = basis.generate_expansion(
        primitive_lattice_vectors=basis.LatticeVectors(u=basis.X, v=basis.Y),
        approximate_num_terms=n_propagating,
        truncation=basis.Truncation.CIRCULAR
    ).basis_coefficients.T

    (true_tex, true_tey, true_tez), (true_thx, true_thy, true_thz), (true_rex, true_rey, true_rez), _ = np.load(
        'C:/Users/eugene/OneDrive/Рабочий стол/red_freeform.npz')['amps'][0, ..., :n_propagating]

    # all the wavevectors are normalized by k0
    kx = mode_n / p
    ky = mode_m / p
    kz = np.sqrt(1 - kx ** 2 - ky ** 2)

    tex = true_tex
    tey = true_tey
    rex = true_rex
    rey = true_rey

    tez = (kx * tex + ky * tey) / kz
    assert np.linalg.norm(tez - true_tez) < 5e-4
    rez = -(kx * rex + ky * rey) / kz
    assert np.linalg.norm(rez - true_rez) < 5e-4

    thx = ky * tez + kz * tey
    assert np.linalg.norm(thx - true_thx) < 5e-4
    thy = -(kz * tex + kx * tez)
    assert np.linalg.norm(thy - true_thy) < 5e-4

    total_transmitted_power = np.sum((np.abs(tex) ** 2 + np.abs(tey) ** 2 + np.abs(tez) ** 2) * kz)
    total_reflected_power = np.sum((np.abs(rex) ** 2 + np.abs(rey) ** 2 + np.abs(rez) ** 2) * kz)
    assert (total_transmitted_power + total_reflected_power - 1) < 1e-3

    primitive_lattice_vectors = basis.LatticeVectors(u=p * basis.X, v=p * basis.Y)
    solve_result_ambient = fmm.eigensolve_isotropic_media(
        permittivity=np.atleast_2d(1.),
        wavelength=np.asarray(1.),
        in_plane_wavevector=np.array([0., 0.]),
        primitive_lattice_vectors=primitive_lattice_vectors,
        expansion=basis.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors, approximate_num_terms=n_propagating),
        formulation=fmm.Formulation.FFT
    )
    n_samples = 256
    (true_ex, true_ey, true_ez), (true_hx, true_hy, true_hz), (x, y) = fields.fields_on_grid(
        (true_tex[:, None], true_tey[:, None], true_tez[:, None]),
        (true_thx[:, None], true_thy[:, None], true_thz[:, None]),
        solve_result_ambient, shape=(n_samples, n_samples), num_unit_cells=(1, 1))
    true_ex, true_ey, true_ez, true_hx, true_hy, true_hz = [
        a.squeeze()
        for a in (true_ex, true_ey, true_ez, true_hx, true_hy, true_hz)
    ]

    ex = np.sum(
        tex[:, None, None] * np.exp(2j * np.pi * (
                kx[:, None, None] * x[None] + ky[:, None, None] * y[None]
        )), axis=0)
    assert np.linalg.norm(ex - true_ex) / x.size < 1e-6

    true_intensity = -np.real(true_ex * np.conj(true_hy) - true_ey * np.conj(true_hx))
    assert np.abs((np.sum(true_intensity) / true_intensity.size) - total_transmitted_power) < 5e-4


    # Check that
    # 1. Sz(x, y) = Re[sum_nm sum_n'm' exp(i kxdn x + i kydm y) Anmn'm']
    # 2. Anmn'm' = an'm'^*T K anm
    # 3. Integral of exp(...) = theta(dn) * theta(dm)
    # 4. Power in the area [0,p/2]x[0,p/2] is the same as sum A12 * theta_n * theta_m

    def theta(dn):
        if dn == 0:
            return 1 / 2
        if dn % 2 == 1:
            return 1j / (np.pi * dn)
        return 0


    intensity_xy = np.zeros_like(ex, dtype=float)
    focused_power = 0
    for Ex1, Ey1, kx1, ky1, kz1, n1, m1 in zip(tex, tey, kx, ky, kz, mode_n, mode_m):
        for Ex2, Ey2, kx2, ky2, kz2, n2, m2 in zip(tex, tey, kx, ky, kz, mode_n, mode_m):
            A12 = (
                    (kz2 + kx2 ** 2 / kz2) * (Ex1 * Ex2.conj()) + (kz2 + ky2 ** 2 / kz2) * (Ey1 * Ey2.conj())
                    + (kx2 * ky2 / kz2) * (Ex1 * Ey2.conj() + Ey1 * Ex2.conj())
            )
            a1 = np.array([[Ex1, Ey1]]).T
            a2 = np.array([[Ex2, Ey2]]).T
            K12 = np.array([
                [kz2 + kx2 ** 2 / kz2, (kx2 * ky2) / kz2],
                [(kx2 * ky2) / kz2, kz2 + ky2 ** 2 / kz2]
            ])
            A12_quadratic_form = np.dot(a2.conj().T, np.dot(K12, a1))
            assert np.abs(A12_quadratic_form - A12) / np.abs(A12) < 1e-6

            intensity_xy += (A12 * np.exp(2j * np.pi * ((kx1 - kx2) * x + (ky1 - ky2) * y))).real
            focused_power += (A12 * theta(n1 - n2) * theta(m1 - m2)).real

    assert np.mean(np.abs(intensity_xy - true_intensity)) < 5e-4

    true_focused_power = np.sum(true_intensity[:n_samples // 2, :n_samples // 2]) / (n_samples ** 2)
    assert abs(true_focused_power - focused_power) < 1e-3
    print(true_focused_power, focused_power)

    # ---------------------- #
    # BELOW CHECK EFFICIENCY #
    # ---------------------- #

    n1, n2 = np.meshgrid(mode_n, mode_n)
    m1, m2, = np.meshgrid(mode_m, mode_m)
    dn = n1 - n2
    dm = m1 - m2
    theta_n = np.where(dn == 0, 1 / 2, np.where(dn % 2 == 1, 1j / (np.pi * np.where(dn == 0, 1, dn)), 0))
    theta_m = np.where(dm == 0, 1 / 2, np.where(dm % 2 == 1, 1j / (np.pi * np.where(dm == 0, 1, dm)), 0))
    theta = theta_n * theta_m

    Fxx = theta * (kz[None, :] + kx[None, :] ** 2 / kz[None, :])
    Fyy = theta * (kz[None, :] + ky[None, :] ** 2 / kz[None, :])
    Fxy = theta * (kx[None, :] * ky[None, :] / kz[None, :])
    F = np.block([[Fxx, Fxy], [Fxy, Fyy]])
    F = (F + F.conj().T) / 2

    amps = np.concatenate([tex, tey])

    focused_power = (amps.conj() @ F @ amps).real
    print(focused_power)

    print(total_transmitted_power, np.sum(true_intensity) / true_intensity.size)
    print(np.sum(true_intensity[:n_samples // 2, :n_samples // 2]) / true_intensity.size)
    print(np.sum(true_intensity[n_samples // 2:, :n_samples // 2]) / true_intensity.size)
    print(np.sum(true_intensity[:n_samples // 2, n_samples // 2:]) / true_intensity.size)
    print(np.sum(true_intensity[n_samples // 2:, n_samples // 2:]) / true_intensity.size)

    # test full power matrix
    P = 4 * F * np.tile(np.logical_and(dn == 0, dm == 0), (2, 2))
    print((amps.conj() @ P @ amps).real)
