import numpy as np
import h5py

from scattering_simulation import prepare_lens_scattering_solver
import topology_parametrization

import matplotlib.pyplot as plt


if __name__ == '__main__':
    with h5py.File('rgb4224.hdf5', 'r') as f:
        patterns = f['patterns'][()]
        red_amps = f['red_amps'][()]
        green_amps = f['green_amps'][()]
        blue_amps = f['blue_amps'][()]

    print(patterns.shape, red_amps.shape, green_amps.shape, blue_amps.shape)

    propagating_indices = np.array([
        [0, -1, 0, 0, 1, -1, -1, 1, 1, -2, 0, 0, 2, -2, -2, -1, -1, 1, 1, 2, 2, -2, -2, 2, 2, -3, 0, 0, 3, -3, -3, -1,
         -1, 1, 1, 3, 3, -3, -3, -2, -2, 2, 2, 3, 3, -4, 0, 0, 4, -4, -4, -1, -1, 1, 1, 4, 4, -3, -3, 3, 3],
        [0, 0, -1, 1, 0, -1, 1, -1, 1, 0, -2, 2, 0, -1, 1, -2, 2, -2, 2, -1, 1, -2, 2, -2, 2, 0, -3, 3, 0, -1, 1, -3, 3,
         -3, 3, -1, 1, -2, 2, -3, 3, -3, 3, -2, 2, 0, -4, 4, 0, -1, 1, -4, 4, -4, 4, -1, 1, -3, 3, -3, 3]
    ])
    ipj = propagating_indices[0] + propagating_indices[1]
    red_ipj = ipj[:29]
    green_ipj = ipj[:45]
    blue_ipj = ipj[:61]

    n = 2 ** 12
    n_augmented = 2 ** 15
    n_to_add = n_augmented - n
    n_additional_copies = n_to_add // n
    print(n, n_augmented, n_to_add, n_additional_copies)

    n_val = 128

    rng = np.random.default_rng(42)
    rolls = rng.integers(0, 63, n_to_add)
    shifts = rolls / 64

    augmented_patterns = np.zeros((n_augmented + n_val,) + patterns.shape[1:])
    augmented_red_amps = np.zeros((n_augmented + n_val,) + red_amps.shape[1:], dtype=complex)
    augmented_green_amps = np.zeros((n_augmented + n_val,) + green_amps.shape[1:], dtype=complex)
    augmented_blue_amps = np.zeros((n_augmented + n_val,) + blue_amps.shape[1:], dtype=complex)

    augmented_patterns[:n] = patterns[:n]
    augmented_red_amps[:n] = red_amps[:n]
    augmented_green_amps[:n] = green_amps[:n]
    augmented_blue_amps[:n] = blue_amps[:n]

    for i in range(n_to_add):
        if i % 100 == 0:
            print(i)
        augmented_patterns[n + i] = np.roll(patterns[i % n], rolls[i], axis=(0, 1))
        augmented_red_amps[n + i] = red_amps[i % n] * np.exp(-2j * np.pi * shifts[i] * red_ipj)
        augmented_green_amps[n + i] = green_amps[i % n] * np.exp(-2j * np.pi * shifts[i] * green_ipj)
        augmented_blue_amps[n + i] = blue_amps[i % n] * np.exp(-2j * np.pi * shifts[i] * blue_ipj)

    augmented_patterns[n_augmented:] = patterns[n:]
    augmented_red_amps[n_augmented:] = red_amps[n:]
    augmented_green_amps[n_augmented:] = green_amps[n:]
    augmented_blue_amps[n_augmented:] = blue_amps[n:]

    with h5py.File('rgb_augmented32k.hdf5', 'a') as f:
        f.create_dataset('patterns', data=augmented_patterns)
        f.create_dataset('red_amps', data=augmented_red_amps)
        f.create_dataset('green_amps', data=augmented_green_amps)
        f.create_dataset('blue_amps', data=augmented_blue_amps)
