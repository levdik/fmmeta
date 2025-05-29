import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from field_postprocessing import _calculate_max_theoretical_focusing_efficiencies


def calculate_max_go_efficiency(period):
    wavelength = 1
    x, y = np.meshgrid()


if __name__ == '__main__':
    p_max = 5
    periods, efficiencies = _calculate_max_theoretical_focusing_efficiencies(p_max)
    plt.plot([0, p_max], [1, 1], '--')
    plt.step(periods, efficiencies, where='post')
    plt.xlabel('Period relative to Wavelength')
    plt.ylabel('Maximal far-field focusing efficiency')
    plt.grid()
    plt.show()
