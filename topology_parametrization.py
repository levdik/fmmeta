import jax
import jax.numpy as jnp
import numpy as np
from abc import ABC, abstractmethod


class TopologyParametrization(ABC):
    def __init__(self, n_geometrical_parameters: int, minval=-1., maxval=1.):
        self.n_geometrical_parameters = n_geometrical_parameters
        self.minval = minval
        self.maxval = maxval

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


class Cutoff(TopologyParametrization):
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


class Grid(TopologyParametrization):
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

        _, unique_parameter_indices, symmetry_indices = np.unique(unique_ids, return_index=True, return_inverse=True)

        self.grid_size = grid_size
        self.symmetry_indices = jnp.array(symmetry_indices)
        self.unique_parameter_indices = unique_parameter_indices
        n_geometrical_parameters = len(unique_parameter_indices)
        TopologyParametrization.__init__(self, n_geometrical_parameters)

    def apply_symmetry(self, geometrical_parameters: jnp.ndarray) -> jnp.ndarray:
        return geometrical_parameters[self.symmetry_indices]

    def extract_unique_parameters(self, full_geometrical_parameters: jnp.ndarray) -> jnp.ndarray:
        return full_geometrical_parameters[self.unique_parameter_indices]


class BicubicInterpolation(Cutoff, Grid):
    def __init__(self, grid_size: int, symmetry_type: str | None = None):
        Grid.__init__(self, grid_size, symmetry_type)

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


class FourierInterpolation(Cutoff, Grid):
    def __init__(self, grid_size: int, symmetry_type: str | None = None):
        Grid.__init__(self, grid_size, symmetry_type)

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


class FourierExpansion(Cutoff):
    @staticmethod
    def generate_basis_indices(r):
        n_max = int(np.floor(r))

        basis_indices = []
        single_index_range = [0] + [s * i for i in range(1, n_max + 1) for s in [1, -1]]
        for n in single_index_range:
            for m in single_index_range:
                if n ** 2 + m ** 2 <= r ** 2:
                    basis_indices.append((n, m))
        basis_indices = np.array(basis_indices)
        basis_indices = basis_indices[np.argsort(np.linalg.norm(basis_indices, axis=-1))]
        return basis_indices

    @staticmethod
    def generate_primary_basis_indices(r, symmetry_type='central'):
        full_basis_indices = FourierExpansion.generate_basis_indices(r)
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
        ) = FourierExpansion.generate_primary_basis_indices(r_max, symmetry_type)
        self.full_basis_indices = full_basis_indices
        self.primary_basis = primary_basis
        self.symmetry_indices = symmetry_indices
        self.n_primary_parameters = len(primary_basis)
        TopologyParametrization.__init__(self, n_geometrical_parameters=2 * len(primary_basis) - 1)

    @staticmethod
    def params_to_complex(geometrical_parameters):
        a_00 = geometrical_parameters[0]
        n_primary_parameters = geometrical_parameters.shape[-1] // 2 + 1
        complex_geometrical_parameters = jnp.concatenate([
            jnp.atleast_1d(a_00),
            geometrical_parameters[1:n_primary_parameters]
            + 1j * geometrical_parameters[n_primary_parameters:]
        ])
        return complex_geometrical_parameters

    @staticmethod
    def params_from_complex(complex_geometrical_parameters):
        geometrical_parameters = jnp.hstack([
            complex_geometrical_parameters[..., 0][..., None].astype(float),
            complex_geometrical_parameters[..., 1:].real,
            complex_geometrical_parameters[..., 1:].imag
        ])
        return geometrical_parameters

    def apply_symmetry(self, geometrical_parameters: jnp.ndarray) -> jnp.ndarray:
        a_00 = geometrical_parameters[0]
        complex_geometrical_parameters = jnp.concatenate([
            jnp.atleast_1d(a_00),
            geometrical_parameters[1:self.n_primary_parameters]
            + 1j * geometrical_parameters[self.n_primary_parameters:]
        ])
        return complex_geometrical_parameters[self.symmetry_indices]

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


class CrossPillarWithHole(Grid):
    def __init__(self, grid_size: int, symmetry_type: str | None = None):
        Grid.__init__(self, grid_size, symmetry_type)
        self.n_unique_pillars = self.n_geometrical_parameters
        self.n_geometrical_parameters = 4 * self.n_geometrical_parameters
        self.minval = 0

    def apply_symmetry(self, geometrical_parameters: jnp.ndarray) -> jnp.ndarray:
        geometrical_parameters = geometrical_parameters.reshape(self.n_unique_pillars, 4)
        a = geometrical_parameters[..., 0]
        b = geometrical_parameters[..., 1]
        ah = geometrical_parameters[..., 2]
        bh = geometrical_parameters[..., 3]
        geometrical_parameters = geometrical_parameters.at[..., 1].set(b * (1 - a) / 2)
        geometrical_parameters = geometrical_parameters.at[..., 2].set(ah * a)
        geometrical_parameters = geometrical_parameters.at[..., 3].set(bh * (a - ah * a) / 2)
        return Grid.apply_symmetry(self, geometrical_parameters)

    @staticmethod
    def _box_filling_map(
            relative_width: jnp.ndarray,
            relative_height: jnp.ndarray,
            n_pixels: int
    ) -> jnp.ndarray:
        assert jnp.shape(relative_width) == ()
        assert jnp.shape(relative_height) == ()

        single_coordinate_samples = jnp.linspace(-0.5, 0.5, n_pixels, endpoint=False) + 0.5 / n_pixels
        x_mesh, y_mesh = jnp.meshgrid(single_coordinate_samples, single_coordinate_samples)
        horizontal_side_y = relative_height / 2
        vertical_side_x = relative_width / 2
        signed_distance_to_closest_horizontal_side_px = (horizontal_side_y - jnp.abs(y_mesh)) * n_pixels
        signed_distance_to_closest_vertical_side_px = (vertical_side_x - jnp.abs(x_mesh)) * n_pixels
        vertical_filling = jnp.clip(signed_distance_to_closest_vertical_side_px + 0.5, 0., 1.)
        horizontal_filling = jnp.clip(signed_distance_to_closest_horizontal_side_px + 0.5, 0., 1.)
        box_filling = vertical_filling * horizontal_filling
        return box_filling

    @staticmethod
    def _cross_filling_map(
            relative_outer_side_width: jnp.ndarray,
            relative_corner_width: jnp.ndarray,
            n_pixels: int
    ) -> jnp.ndarray:
        assert jnp.shape(relative_outer_side_width) == ()
        assert jnp.shape(relative_corner_width) == ()

        cross_width = 2 * relative_corner_width + relative_outer_side_width
        horizontal_box_filling_map = CrossPillarWithHole._box_filling_map(
            relative_width=cross_width,
            relative_height=relative_outer_side_width,
            n_pixels=n_pixels
        )
        vertical_box_filling_map = horizontal_box_filling_map.T
        cross_filling_map = 1. - (1. - vertical_box_filling_map) * (1. - horizontal_box_filling_map)
        return cross_filling_map

    @staticmethod
    def _cross_with_hole_filling_map(
            relative_shape: jnp.ndarray,
            n_pixels: int
    ) -> jnp.ndarray:
        cross_filling = CrossPillarWithHole._cross_filling_map(
            relative_outer_side_width=relative_shape[0],
            relative_corner_width=relative_shape[1],
            n_pixels=n_pixels
        )
        hole_filling = 1. - CrossPillarWithHole._cross_filling_map(
            relative_outer_side_width=relative_shape[2],
            relative_corner_width=relative_shape[3],
            n_pixels=n_pixels
        )
        cross_with_hole_filling = cross_filling * hole_filling
        return cross_with_hole_filling

    def _generate_filling_map(self, geometrical_parameters: jnp.ndarray, n_samples: int) -> jnp.ndarray:
        n_pixels_per_pillar = round(n_samples / self.grid_size)
        geometrical_parameters = geometrical_parameters.reshape(-1, 4)
        filling_blocks_flat = jax.vmap(
            CrossPillarWithHole._cross_with_hole_filling_map,
            in_axes=(0, None)
        )(
            geometrical_parameters, n_pixels_per_pillar
        )
        filling_blocks = filling_blocks_flat.reshape(
            self.grid_size, self.grid_size, n_pixels_per_pillar, n_pixels_per_pillar
        )
        filling = filling_blocks.transpose(0, 2, 1, 3).reshape(
            self.grid_size * n_pixels_per_pillar, self.grid_size * n_pixels_per_pillar
        )
        filling = jax.image.resize(filling, shape=(n_samples, n_samples), method='linear')
        return filling


class SquarePillar(CrossPillarWithHole):
    def __init__(self, grid_size: int, symmetry_type: str | None = None):
        CrossPillarWithHole.__init__(self, grid_size, symmetry_type)
        self.n_geometrical_parameters = self.n_geometrical_parameters // 4

    def apply_symmetry(self, geometrical_parameters: jnp.ndarray) -> jnp.ndarray:
        geometrical_parameters = jnp.stack(
            [geometrical_parameters] + 3 * [jnp.zeros_like(geometrical_parameters)],
            axis=-1
        )
        return CrossPillarWithHole.apply_symmetry(self, geometrical_parameters)


class CrossPillar(CrossPillarWithHole):
    def __init__(self, grid_size: int, symmetry_type: str | None = None):
        CrossPillarWithHole.__init__(self, grid_size, symmetry_type)
        self.n_geometrical_parameters = self.n_geometrical_parameters // 2

    def apply_symmetry(self, geometrical_parameters: jnp.ndarray) -> jnp.ndarray:
        geometrical_parameters = jnp.stack(
            [geometrical_parameters[:self.n_unique_pillars], geometrical_parameters[self.n_unique_pillars:]]
            + 2 * [jnp.zeros(self.n_unique_pillars)],
            axis=-1
        )
        return CrossPillarWithHole.apply_symmetry(self, geometrical_parameters)


class GaussianField(Cutoff):
    def __init__(self, n_pixels, sigma, symmetry_type=None):
        self.n_pixels = n_pixels
        self.sigma = sigma
        self.symmetry_type = symmetry_type
        TopologyParametrization.__init__(self, n_geometrical_parameters=n_pixels ** 2)

        kx = jnp.fft.fftfreq(n_pixels)
        kx, ky = jnp.meshgrid(kx, kx)
        filter = jnp.exp(-0.25 * (sigma ** 2) * (kx ** 2 + ky ** 2))
        self.fourier_filter = filter

    def apply_symmetry(self, geometrical_parameters: jnp.ndarray) -> jnp.ndarray:
        x = geometrical_parameters.reshape(self.n_pixels, self.n_pixels)
        if self.symmetry_type is None:
            pass
        elif self.symmetry_type == 'central':
            x = (x + x.T) / 2
            x = (x + jnp.rot90(x, 2)) / 2
            x = (x + jnp.flip(x, 0)) / 2
        elif self.symmetry_type == 'main_diagonal':
            x = (x + x.T) / 2
        else:
            raise ValueError('Unknown symmetry type')
        return x

    def _generate_pattern_cutoff_values(self, geometrical_parameters: jnp.ndarray, n_samples: int) -> jnp.ndarray:
        filtered = jnp.fft.ifft2(self.fourier_filter * jnp.fft.fft2(geometrical_parameters)).real
        return filtered * jnp.sqrt(self.fourier_filter.size / jnp.sum(jnp.abs(self.fourier_filter) ** 2))
        # filtered = (filtered - jnp.min(filtered)) / (jnp.max(filtered) - jnp.min(filtered))
        # filtered = filtered * 2 - 1
        # return filtered

    def _generate_filling_map(
            self, geometrical_parameters: jnp.ndarray, n_samples: int, bin_strength: float = 1.
    ) -> jnp.ndarray:
        cutoff_values = self._generate_pattern_cutoff_values(geometrical_parameters, n_samples)
        binarised_with_edges = Cutoff._generate_filling_map(self, geometrical_parameters, n_samples)
        soft_binarized = bin_strength * binarised_with_edges + ((cutoff_values + 1) / 2) * (1 - bin_strength)
        soft_binarized = jax.image.resize(soft_binarized, (n_samples, n_samples), method='linear')
        return soft_binarized


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use('TkAgg')

    topology_parametrization = GaussianField(n_pixels=100, sigma=16, symmetry_type='main_diagonal')

    fig, ax = plt.subplots(5, 5)
    rng_key = jax.random.key(3)

    for ax_i in ax.flatten():
    # for ax_i in [ax]:
        rng_key, rng_subkey = jax.random.split(rng_key)
        x = jax.random.uniform(
            rng_subkey,
            shape=(topology_parametrization.n_geometrical_parameters,),
            minval=topology_parametrization.minval, maxval=topology_parametrization.maxval
        )

        rng_key, rng_subkey = jax.random.split(rng_key)
        filling_shift = jnp.clip(0.01 * jax.random.normal(rng_subkey), -1, 1)
        # filling_shift = jax.random.uniform(rng_subkey, minval=-1, maxval=1)
        x += filling_shift

        y = topology_parametrization(x, n_samples=128, bin_strength=1)
        print(np.min(y), np.max(y))
        # y = topology_parametrization._generate_pattern_cutoff_values(topology_parametrization.apply_symmetry(x), 100)
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(topology_parametrization.apply_symmetry(x))
        # ax[1].imshow(y)
        ax_i.imshow(y, vmin=0, vmax=1)
        ax_i.set_axis_off()

    plt.tight_layout()
    plt.show()
