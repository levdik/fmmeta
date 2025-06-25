import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from field_postprocessing import calculate_focusing_efficiency
from field_postprocessing import _calculate_max_theoretical_focusing_efficiencies
from phase_profile_manager import generate_target_phase
from time import time


def calculate_max_go_efficiency(period, focal_distance):
    k0 = 2 * np.pi
    K = 2 * np.pi / period

    n_samples = 100
    x = np.linspace(0, period, n_samples)
    x, y = np.meshgrid(x, x)
    # TODO: does it depend on focal length?
    phase = generate_target_phase(
        points=np.stack([x, y], axis=-1),
        focal_points=[period/4, period/4, focal_distance],
        wavelength=1.,
        xy_period=period
    )
    # plt.imshow(phase, cmap='hsv', origin='lower')
    # plt.colorbar()
    # plt.show()
    lens_field = np.exp(1j * phase)
    amps = np.fft.fft2(lens_field)
    n = np.fft.fftfreq(n_samples, 1 / n_samples)

    n, m = np.meshgrid(n, n)
    kz = np.sqrt(k0 ** 2 - (K * n) ** 2 - (K * m) ** 2, dtype=complex)
    focal_amps = amps * np.exp(1j * kz * focal_distance)
    propagating_array_indices = np.where(n ** 2 + m ** 2 < period ** 2)
    propagating_amps = focal_amps[propagating_array_indices]
    propagating_basis_indices = list(zip(n[propagating_array_indices], m[propagating_array_indices]))
    eff = calculate_focusing_efficiency(propagating_amps, propagating_basis_indices, relative_focal_point=(0.25, 0.25))

    # focal_field = np.fft.ifft2(focal_amps)
    # plt.imshow(np.abs(focal_field) ** 2, origin='lower')
    # plt.show()

    return eff



if __name__ == '__main__':
    jnp.set_printoptions(linewidth=1000)

    # p_max = 5
    # periods, efficiencies = _calculate_max_theoretical_focusing_efficiencies(p_max)
    # plt.plot([0, p_max], [1, 1], '--')
    # plt.step(periods, efficiencies, where='post')
    # plt.xlabel('Period relative to Wavelength')
    # plt.ylabel('Maximal far-field focusing efficiency')
    # plt.grid()
    # plt.show()

    # effs = []
    # focal_distances = np.concatenate([np.linspace(0, 10, 101), np.linspace(10, 50, 101)[1:]])
    #
    # for i, fd in enumerate(focal_distances):
    #     if i % 10 == 9:
    #         progress = i / len(focal_distances)
    #         print(round(progress * 100), '%')
    #     effs.append(calculate_max_go_efficiency(period=7 * 300 / 650, focal_distance=fd))
    #
    # plt.plot(focal_distances, effs)
    # plt.xlabel('Focal distance relative to Wavelength')
    # plt.ylabel('Focusing efficiency given by GO')
    # plt.grid()
    # plt.show()

    effs = []
    focal_distance = 4000 / 650
    periods = np.linspace(0, 100, 51)[1:]

    for i, p in enumerate(periods):
        effs.append(calculate_max_go_efficiency(period=p, focal_distance=focal_distance))
        print(i, p, effs[-1])

    for p, e in zip(periods, effs):
        print(p, e)

    # plt.plot(periods, effs)
    # plt.xlabel('Focal distance relative to Wavelength')
    # plt.ylabel('Focusing efficiency given by GO')
    # plt.grid()
    # plt.show()
