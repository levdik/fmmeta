import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from field_postprocessing import calculate_focusing_efficiency
from field_postprocessing import _calculate_max_theoretical_focusing_efficiencies
from phase_profile_manager import generate_target_phase


def calculate_max_go_efficiency(period):
    k0 = 2 * np.pi
    K = 2 * np.pi / period

    focal_distance = 4000 / 650

    n_samples = 100
    x = np.linspace(0, period, n_samples)
    x, y = np.meshgrid(x, x)
    # TODO: does it depend on focal length?
    phase = generate_target_phase(
        points=np.stack([x, y], axis=-1),
        focal_points=[period/2, period/2, focal_distance],
        wavelength=1.,
        xy_period=period
    )
    plt.imshow(phase, cmap='hsv')
    plt.colorbar()
    plt.show()
    lens_field = np.exp(1j * phase)
    amps = np.fft.fft2(lens_field)

    n = np.fft.fftfreq(n_samples, 1 / n_samples)
    n, m = np.meshgrid(n, n)
    kz = np.sqrt(k0 ** 2 - (K ** 2 * (n ** 2 + m ** 2)), dtype=complex)
    focal_amps = amps * np.exp(1j * kz * focal_distance)
    propagating_indices = np.where(n ** 2 + m ** 2 < period ** 2)
    propagating_amps = focal_amps[propagating_indices]
    propagating_indices = list(zip(n[propagating_indices], m[propagating_indices]))
    eff = calculate_focusing_efficiency(propagating_amps, propagating_indices)

    focal_field = np.fft.ifft2(focal_amps)
    plt.imshow(np.abs(focal_field) ** 2)
    plt.show()

    return eff



if __name__ == '__main__':
    # p_max = 5
    # periods, efficiencies = _calculate_max_theoretical_focusing_efficiencies(p_max)
    # plt.plot([0, p_max], [1, 1], '--')
    # plt.step(periods, efficiencies, where='post')
    # plt.xlabel('Period relative to Wavelength')
    # plt.ylabel('Maximal far-field focusing efficiency')
    # plt.grid()
    # plt.show()

    print(calculate_max_go_efficiency(7 * 300 / 650))
