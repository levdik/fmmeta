import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx, serialization

from field_postprocessing import extract_amps_from_fields


class SpectralConv2D(nnx.Module):
    def __init__(self, in_channels, out_channels, modes, rngs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        init_w_scale = 1 / (in_channels * out_channels)
        init_w_kwargs = {
            'shape': (2 * modes + 1, modes + 1, in_channels, out_channels),
            'minval': -init_w_scale, 'maxval': init_w_scale
        }
        self.w_re = nnx.Param(jax.random.uniform(rngs(), **init_w_kwargs))
        self.w_im = nnx.Param(jax.random.uniform(rngs(), **init_w_kwargs))

    def __call__(self, x):
        n_px = x.shape[-2]
        mode_start, mode_end = n_px // 2 - self.modes, n_px // 2 + self.modes + 1

        x_fft = jnp.fft.rfft2(x, axes=(-3, -2))
        x_fft_shifted = jnp.fft.fftshift(x_fft, axes=-3)
        x_fft_sliced = x_fft_shifted[..., mode_start:mode_end, :self.modes + 1, :]

        w = self.w_re + 1j * self.w_im
        y_fft_sliced = jnp.einsum('bnmi, nmio -> bnmo', x_fft_sliced, w)

        y_fft_shifted = jnp.zeros(x_fft.shape[:-1] + (self.out_channels,), dtype=complex)
        y_fft_shifted = y_fft_shifted.at[..., mode_start:mode_end, :self.modes + 1, :].set(y_fft_sliced)
        y_fft = jnp.fft.ifftshift(y_fft_shifted, axes=-3)
        y = jnp.fft.irfft2(y_fft, axes=(-3, -2))

        return y


class FNOLayer(nnx.Module):
    def __init__(self, in_channels, out_channels, modes, activation, rngs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.activation = activation

        self.spectral_conv = SpectralConv2D(in_channels, out_channels, modes, rngs=rngs)
        self.bypass_conv = nnx.Conv(in_channels, out_channels, kernel_size=1, rngs=rngs)

    def __call__(self, x):
        return self.activation(
            self.spectral_conv(x)
            + self.bypass_conv(x)
        )


class ScatteringFNO(nnx.Module):
    def __init__(self, dout, hidden_dims, modes, activation, rngs):
        self.din = 1
        self.dout = dout
        self.lifting = nnx.Conv(self.din, hidden_dims[0], kernel_size=1, rngs=rngs)
        self.fourier_layers = nnx.List([
            FNOLayer(hidden_dims[i], hidden_dims[i + 1], modes, activation, rngs)
            for i in range(len(hidden_dims) - 1)
        ])
        self.projection = nnx.Conv(hidden_dims[-1], 2 * self.dout, kernel_size=1, rngs=rngs)

    def __call__(self, x):
        x = self.lifting(x[..., None])
        for layer in self.fourier_layers:
            x = layer(x)
        x = self.projection(x)
        x = x[..., :self.dout] + 1j * x[..., self.dout:]
        return x


class NormalizedScatteringFNO(nnx.Module):
    def __init__(self, scattering_fno, basis_indices, relative_period):
        self.scattering_fno = scattering_fno
        self.basis_indices = basis_indices
        p = relative_period
        mode_n, mode_m = propagating_indices[:, :n_propagating_modes]
        kx = mode_n / p
        ky = mode_m / p
        self.kz = np.sqrt(1 - kx ** 2 - ky ** 2)

    def __call__(self, x):
        field_maps = self.scattering_fno(x)
        amps = extract_amps_from_fields(field_maps, self.basis_indices)
        total_amps_norm = jnp.sum(jnp.abs(amps) ** 2, axis=-1)
        total_energy = jnp.sum(total_amps_norm * self.kz[None, :], axis=-1)
        return amps / jnp.sqrt(total_energy)[:, None, None]


class ScatteringMLP(nnx.Module):
    def __init__(self, din, n_modes, n_out_channels, hidden_dims, rngs):
        self.din = din
        self.n_modes = n_modes
        self.n_out_channels = n_out_channels
        dims = [din] + hidden_dims + [n_modes * n_out_channels * 2]
        self.layers = nnx.List([
            nnx.Linear(dims[i], dims[i + 1], rngs=rngs)
            for i in range(len(dims) - 1)
        ])

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            x = nnx.gelu(x)
        x = x.reshape(-1, 2 * self.n_modes, self.n_out_channels)
        x = x[..., :self.n_modes, :] + 1j * x[..., self.n_modes:, :]
        return x


def save_model(model, filename):
    state = nnx.state(model)
    state_dict = state.to_pure_dict()
    with open(filename, 'wb') as f:
        f.write(serialization.to_bytes(state_dict))


def load_model(filename, model_type, **model_init_kwags):
    with open(filename, 'rb') as f:
        bytes_data = f.read()
    model = model_type(rngs=nnx.Rngs(0), **model_init_kwags)
    state_dict_template = nnx.state(model).to_pure_dict()
    loaded_state_dict = serialization.from_bytes(state_dict_template, bytes_data)
    nnx.update(model, loaded_state_dict)
    return model
