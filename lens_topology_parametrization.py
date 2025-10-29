from xmlrpc.client import Boolean

import jax
import jax.numpy as jnp
import numpy as np
from abc import ABC, abstractmethod


class TopologyParametrization(ABC):
    def __init__(self, n_geometrical_parameters: int):
        self.n_geometrical_parameters = n_geometrical_parameters

    def apply_symmetry(self, geometrical_parameters: jnp.ndarray) -> jnp.ndarray:
        return geometrical_parameters

    def extract_unique_parameters(self, full_geometrical_parameters: jnp.ndarray) -> jnp.ndarray:
        return full_geometrical_parameters

    @abstractmethod
    def _generate_filling_map(self, geometrical_parameters: jnp.ndarray, n_samples: int) -> jnp.ndarray:
        pass

    def __call__(self, args: jnp.ndarray, n_samples: int = 100, **kwargs):
        geometrical_parameters = self.apply_symmetry(args)
        filling_map = self._generate_filling_map(geometrical_parameters, n_samples, **kwargs)
        return filling_map


class CutoffTopologyParametrization(TopologyParametrization):
    @abstractmethod
    def _generate_pattern_cutoff_values(self, geometrical_parameters: jnp.ndarray, n_samples: int) -> jnp.ndarray:
        pass

    def _generate_filling_map(self, geometrical_parameters: jnp.ndarray, n_samples: int) -> jnp.ndarray:
        cutoff_values = self._generate_pattern_cutoff_values(geometrical_parameters, n_samples)

        cell_corner_values = jnp.stack([
            cutoff_values,
            jnp.roll(cutoff_values, -1, axis=0),
            jnp.roll(cutoff_values, -1, axis=1),
            jnp.roll(cutoff_values, -1, axis=(0, 1))
        ])
        max_corner_value = jnp.max(cell_corner_values, axis=0)
        min_corner_value = jnp.min(cell_corner_values, axis=0)
        dz = max_corner_value - min_corner_value
        filling_map = max_corner_value / (dz + 1e-6 * jnp.exp(- dz ** 2))
        filling_map = jnp.where(jnp.isnan(filling_map), jnp.sign(max_corner_value), filling_map)
        filling_map = jnp.clip(filling_map, 0, 1)

        return filling_map


class GridTopologyParametrization(TopologyParametrization):
    def __init__(self, grid_size: int, symmetry_type: str | None = None):
        if symmetry_type is None:
            unique_ids = np.arange(grid_size ** 2).reshape(grid_size, grid_size)
        elif symmetry_type == 'main_diagonal':
            big_enough_order_factor = 10 ** len(str(grid_size))
            i, j = np.indices((grid_size, grid_size))
            unique_ids = np.max([i, j], axis=0) * big_enough_order_factor + np.min([i, j], axis=0)
        elif symmetry_type == 'central':
            big_enough_order_factor = 10 ** (len(str(grid_size)) + 1)
            i, j = np.indices((grid_size, grid_size))
            center_i = center_j = (grid_size - 1) / 2
            di = np.abs(i - center_i)
            dj = np.abs(j - center_j)
            unique_ids = np.max([di, dj], axis=0) * big_enough_order_factor + np.min([di, dj], axis=0)
        else:
            raise ValueError("Unknown symmetry type. Allowed values: None, 'main_diagonal', 'central'")

        _, unique_parameter_indices, symmetry_indices  = np.unique(unique_ids, return_index=True, return_inverse=True)

        self.symmetry_indices = jnp.array(symmetry_indices)
        self.unique_parameter_indices = unique_parameter_indices
        n_geometrical_parameters = len(unique_parameter_indices)
        TopologyParametrization.__init__(self, n_geometrical_parameters)

    def apply_symmetry(self, geometrical_parameters: jnp.ndarray) -> jnp.ndarray:
        return geometrical_parameters[self.symmetry_indices]

    def extract_unique_parameters(self, full_geometrical_parameters: jnp.ndarray) -> jnp.ndarray:
        return full_geometrical_parameters[self.unique_parameter_indices]


class BicubicInterpolationTopologyParametrization(CutoffTopologyParametrization, GridTopologyParametrization):
    def __init__(self, grid_size: int, symmetry_type: str | None = None):
        GridTopologyParametrization.__init__(self, grid_size, symmetry_type)

    def _generate_pattern_cutoff_values(self, geometrical_parameters: jnp.ndarray, n_samples: int) -> jnp.ndarray:
        n_pad = 2
        n_primary_samples = geometrical_parameters.shape[0]
        n_padded_primary_samples = n_primary_samples + 2 * n_pad
        n_padded_interpolated_samples = round(n_samples * n_padded_primary_samples / n_primary_samples)
        interpolated_padded = jax.image.resize(
            jnp.pad(geometrical_parameters, n_pad, 'wrap'),
            shape=(n_padded_interpolated_samples,) * 2,
            method='bicubic'
        )
        n_pad_interpolated = (n_padded_interpolated_samples - n_samples) // 2
        interpolated = interpolated_padded[
                       n_pad_interpolated:n_pad_interpolated + n_samples,
                       n_pad_interpolated:n_pad_interpolated + n_samples]
        return interpolated


class FourierInterpolationTopologyParametrization(CutoffTopologyParametrization, GridTopologyParametrization):
    def __init__(self, grid_size: int, symmetry_type: str | None = None):
        GridTopologyParametrization.__init__(self, grid_size, symmetry_type)

    def _generate_pattern_cutoff_values(self, geometrical_parameters: jnp.ndarray, n_samples: int) -> jnp.ndarray:
        n_primary_samples = geometrical_parameters.shape[0]
        fourier_amps = jnp.fft.fftshift(jnp.fft.fft2(geometrical_parameters))

        shift_between_pixel_centers = (1 / n_primary_samples - 1 / n_samples) / 2
        single_axis_freq = jnp.fft.fftshift(jnp.fft.fftfreq(n_primary_samples, d=1 / n_primary_samples))
        i_freq, j_freq = jnp.meshgrid(single_axis_freq, single_axis_freq)
        fourier_amps *= jnp.exp(-1j * 2 * jnp.pi * shift_between_pixel_centers * (i_freq + j_freq))

        upscaled_fourier_amps = jnp.zeros((n_samples, n_samples), dtype=complex)
        half = n_primary_samples // 2
        start = n_samples // 2 - half
        end = start + n_primary_samples
        upscaled_fourier_amps = upscaled_fourier_amps.at[start:end, start:end].set(fourier_amps)
        upscaled_fourier_amps = jnp.fft.ifftshift(upscaled_fourier_amps)
        interpolated_values = jnp.fft.ifft2(upscaled_fourier_amps).real
        interpolated_values *= (n_samples / n_primary_samples) ** 2
        return interpolated_values


class FourierExpansionTopologyParametrization(CutoffTopologyParametrization):
    @staticmethod
    def generate_basis_indices(r):
        n_max = int(np.floor(r))

        basis_indices = []
        single_index_range = [0] + [s * i for i in range(1, n_max + 1) for s in [1, -1]]
        for n in single_index_range:
            for m in single_index_range:
                if n ** 2 + m ** 2 <= r ** 2:
                    basis_indices.append((n, m))
        return basis_indices

    @staticmethod
    def generate_primary_basis_indices(r, symmetry_type='central'):
        full_basis_indices = FourierExpansionTopologyParametrization.generate_basis_indices(r)
        symmetry_indices = np.zeros(len(full_basis_indices), dtype=int)
        primary_basis = []

        if symmetry_type == 'central':
            for i, (n, m) in enumerate(full_basis_indices):
                if n >= m >= 0:
                    primary_basis.append((n, m))
                    symmetry_indices[i] = len(primary_basis) - 1

            for i, (n, m) in enumerate(full_basis_indices):
                if not (n >= m >= 0):
                    for i1, (n1, m1) in enumerate(primary_basis):
                        if (abs(n1) == abs(n) and abs(m1) == abs(m)) or (abs(n1) == abs(m) and abs(m1) == abs(n)):
                            symmetry_indices[i] = i1
                            break
        elif symmetry_type == 'main_diagonal':
            symmetry_indices = np.zeros(len(full_basis_indices), dtype=int)
            primary_basis = []
            for i, (n, m) in enumerate(full_basis_indices):
                if n >= m:
                    primary_basis.append((n, m))
                    symmetry_indices[i] = len(primary_basis) - 1

            for i, (n, m) in enumerate(full_basis_indices):
                if n < m:
                    for i1, (n1, m1) in enumerate(primary_basis):
                        if n1 == m and m1 == n:
                            symmetry_indices[i] = i1
                            break
        else:
            raise ValueError('Unknown symmetry type')

        full_basis_indices = jnp.array(full_basis_indices)
        primary_basis = jnp.array(primary_basis)

        return full_basis_indices, primary_basis, symmetry_indices

    def __init__(self, r_max, symmetry_type='central'):
        (
            full_basis_indices, primary_basis, symmetry_indices
        ) = FourierExpansionTopologyParametrization.generate_primary_basis_indices(r_max, symmetry_type)
        self.full_basis_indices = full_basis_indices
        self.primary_basis = primary_basis
        self.symmetry_indices = symmetry_indices
        self.n_primary_parameters = len(primary_basis)
        TopologyParametrization.__init__(self, n_geometrical_parameters=2 * len(primary_basis) - 1)

    def apply_symmetry(self, geometrical_parameters: jnp.ndarray) -> jnp.ndarray:
        a_00 = geometrical_parameters[0]
        comples_geometrical_parameters = jnp.concatenate([
            jnp.atleast_1d(a_00),
            geometrical_parameters[1:self.n_primary_parameters]
            + 1j * geometrical_parameters[self.n_primary_parameters:]
        ])
        return comples_geometrical_parameters[self.symmetry_indices]

    def extract_unique_parameters(self, full_geometrical_parameters: jnp.ndarray) -> jnp.ndarray:
        raise NotImplemented

    def _generate_pattern_cutoff_values(self, geometrical_parameters: jnp.ndarray, n_samples: int) -> jnp.ndarray:
        a_00 = geometrical_parameters[0].real
        amps = geometrical_parameters.at[0].set(0)

        single_coordinate_samples = jnp.linspace(0, 1, n_samples, endpoint=False)
        x, y = jnp.meshgrid(single_coordinate_samples, single_coordinate_samples)
        n, m = self.full_basis_indices.T

        phase = n[None, None, :] * x[:, :, None] + m[None, None, :] * y[:, :, None]
        term = amps[None, None, :] * jnp.exp(1j * 2 * jnp.pi * phase)

        wave_values = jnp.sum(term, axis=-1)
        wave_values = wave_values.real

        filling_factor = (a_00 + 1) / 2
        max_wave_val = jnp.max(wave_values)
        min_wave_val = jnp.min(wave_values)
        wave_values -= min_wave_val * filling_factor + max_wave_val * (1 - filling_factor)

        return wave_values


if __name__ == '__main__':
    # topology_parametrization = BicubicInterpolationTopologyParametrization(grid_size=10, symmetry_type='central')
    topology_parametrization = FourierInterpolationTopologyParametrization(grid_size=15, symmetry_type='main_diagonal')
    # topology_parametrization = FourierExpansionTopologyParametrization(r_max=2, symmetry_type='main_diagonal')

    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')

    fig, ax = plt.subplots(5, 5)
    rng_key = jax.random.key(0)

    for ax_i in ax.flatten():
        rng_key, rng_subkey = jax.random.split(rng_key)
        x = jax.random.uniform(
            rng_subkey,
            shape=(topology_parametrization.n_geometrical_parameters,),
            minval=-1, maxval=1
        )
        y = topology_parametrization(x)
        # y = topology_parametrization._generate_pattern_cutoff_values(topology_parametrization.apply_symmetry(x), 100)
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(topology_parametrization.apply_symmetry(x))
        # ax[1].imshow(y)
        ax_i.imshow(y)
        ax_i.set_axis_off()

    plt.tight_layout()
    plt.show()
