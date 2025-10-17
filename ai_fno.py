import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax

from typing import Callable


class FourierLinear(nnx.Module):

    def __init__(
            self,
            n_in_channels: int,
            n_out_channels: int,
            n_pixels: int,
            mode_threshold: float,
            rngs: nnx.Rngs
    ):
        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels

        fourier_ix, fourier_iy = np.meshgrid(
            np.fft.rfftfreq(n_pixels, 1 / n_pixels),
            np.fft.fftfreq(n_pixels, 1 / n_pixels)
        )
        mode_mask = fourier_ix ** 2 + fourier_iy ** 2 <= mode_threshold ** 2
        # self.mode_mask = mode_mask
        self.idx = np.where(mode_mask)
        self.n_selected_modes = int(mode_mask.sum())

        init_w_scale = 1 / (n_in_channels * n_out_channels)
        init_w_kwargs = {
            'shape': (self.n_selected_modes, n_in_channels, n_out_channels),
            'minval': -init_w_scale, 'maxval': init_w_scale
        }
        self.w_re = nnx.Param(jax.random.uniform(rngs(), **init_w_kwargs))
        self.w_im = nnx.Param(jax.random.uniform(rngs(), **init_w_kwargs))

    def __call__(self, x: jax.Array) -> jax.Array:  # x.shape: (*n_batch, n_pixels_x, n_pixels_y, n_in_channels)
        x_fft = jnp.fft.rfft2(x, axes=(-3, -2))
        x_fft_selected = x_fft[..., self.idx[0], self.idx[1], :]

        w = self.w_re + 1j * self.w_im
        y_fft_selected = jnp.einsum('bfi, fio -> bfo', x_fft_selected, w)

        y_fft = jnp.zeros(x.shape[:-3] + (x_fft.shape[-3], x_fft.shape[-2], self.n_out_channels), dtype=complex)
        y_fft = y_fft.at[..., self.idx[0], self.idx[1], :].set(y_fft_selected)
        y = jnp.fft.irfft2(y_fft, axes=(-3, -2))

        return y


class FourierLayer(nnx.Module):

    def __init__(
            self,
            n_in_channels: int,
            n_out_channels: int,
            n_pixels: int,
            mode_threshold: float,
            activation_fn: Callable[jax.Array, jax.Array],
            rngs: nnx.Rngs
    ):
        self.activation_fn = activation_fn
        self.fourier_linear_block = FourierLinear(n_in_channels, n_out_channels, n_pixels, mode_threshold, rngs)
        self.bypass_convolution = nnx.Conv(n_in_channels, n_out_channels, kernel_size=1, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:  # x.shape: (*n_batch, n_pixels_x, n_pixels_y, n_in_channels)
        return self.activation_fn(
            self.fourier_linear_block(x)
            + self.bypass_convolution(x)
        )


class FourierNeuralOperator(nnx.Module):

    def __init__(
            self,
            n_in_channels: int,
            n_out_channels: int,
            hidden_n_channels: tuple[int],
            n_pixels: int,
            mode_threshold: float,
            activation_fn: Callable[jax.Array, jax.Array],
            rngs: nnx.Rngs
    ):
        self.lifting = nnx.Conv(n_in_channels, hidden_n_channels[0], kernel_size=1, rngs=rngs)
        self.fourier_layers = [
            FourierLayer(hidden_n_channels[i], hidden_n_channels[i + 1], n_pixels, mode_threshold, activation_fn, rngs)
            for i in range(len(hidden_n_channels) - 1)
        ]
        self.projection = nnx.Conv(hidden_n_channels[-1], n_out_channels, kernel_size=1, rngs=rngs)

        self.n_selected_modes = self.fourier_layers[0].fourier_linear_block.n_selected_modes

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.lifting(x)
        for fourier_layer in self.fourier_layers:
            x = fourier_layer(x)
        x = self.projection(x)
        return x


# class RealToComplexFNO(nnx.Module):
#     def __init__(
#             self, hidden_n_channels: tuple[int], n_pixels: int, mode_threshold: float,
#             activation_fn: Callable[jax.Array, jax.Array] = nnx.leaky_relu, rngs: nnx.Rngs
#     ):
#         self.fno = FourierNeuralOperator(
#             n_in_channels=1, n_out_channels=2,
#             hidden_n_channels, n_pixels, mode_threshold, activation_fn, rngs
#         )
#
#     def __call__(self, x: jax.Array) -> jax.Array:
#         x = fno(x)
#         x[..., 0] + 1j * x[..., 1]
#         y = jnp.fft.fft2(x)
#         # TODO: select or not propagating waves
#         return y


if __name__ == '__main__':
    fno = FourierNeuralOperator(
        n_in_channels=1,
        n_out_channels=2,
        hidden_n_channels=[64] * 3,
        n_pixels=128,
        mode_threshold=10,
        activation_fn=nnx.leaky_relu,
        rngs=nnx.Rngs(42)
    )

    print(fno.n_selected_modes)
    import numpy as np
    params = nnx.state(fno, nnx.Param)
    total_params = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))
    print(total_params)

    print(fno(jax.random.uniform(jax.random.key(43), (1, 128, 128, 1))).shape)
