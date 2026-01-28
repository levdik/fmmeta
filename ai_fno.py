import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx


class SpectralConv2D(nnx.Module):
    def __init__(self, in_channels, out_channels, modes, rngs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.mode_transformation = nnx.Einsum(
            '...nmi, nmio -> ...nmo',
            kernel_shape=(modes, modes, in_channels, out_channels),
            bias_shape=(modes, modes, out_channels),
            param_dtype=jnp.complex64,
            rngs=rngs
        )

    def __call__(self, x):
        n_px = x.shape[-2]
        mode_start, mode_end = n_px // 2 - self.modes // 2, n_px // 2 + self.modes // 2

        x_fft = jnp.fft.fft2(x, axes=(-3, -2))
        x_fft_shifted = jnp.fft.fftshift(x_fft, axes=(-3, -2))
        x_fft_sliced = x_fft_shifted[..., mode_start:mode_end, mode_start:mode_end, :]

        y_fft_sliced = self.mode_transformation(x_fft_sliced)
        y_fft_shifted = jnp.zeros(x.shape[:-1] + (self.out_channels,), dtype=complex)
        y_fft_shifted = y_fft_shifted.at[..., mode_start:mode_end, mode_start:mode_end, :].set(y_fft_sliced)
        y_fft = jnp.fft.ifftshift(y_fft_shifted, axes=(-3, -2))
        y = jnp.fft.ifft2(y_fft, axes=(-3, -2)).real

        return y


class LinearAttention2D(nnx.Module):
    def __init__(self, channels: int, num_heads: int, rngs: nnx.Rngs):
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.q_proj = nnx.Linear(channels, channels, rngs=rngs)
        self.k_proj = nnx.Linear(channels, channels, rngs=rngs)
        self.v_proj = nnx.Linear(channels, channels, rngs=rngs)
        self.out_proj = nnx.Linear(channels, channels, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        B, H, W, C = x.shape
        N = H * W
        x_flat = x.reshape(B, N, C)

        q = self.q_proj(x_flat)
        k = self.k_proj(x_flat)
        v = self.v_proj(x_flat)

        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = jax.nn.elu(q) + 1.0
        k = jax.nn.elu(k) + 1.0

        context = jnp.einsum('bhnd, bhne -> bhde', k, v)

        k_sum = jnp.einsum('bhnd -> bhd', k)
        denom = jnp.einsum('bhnd, bhd -> bhn', q, k_sum)
        denom = jnp.expand_dims(denom, axis=-1) + 1e-6  # Stability epsilon

        num = jnp.einsum('bhnd, bhde -> bhne', q, context)

        out = num / denom

        out = out.transpose(0, 2, 1, 3).reshape(B, N, C)
        out = self.out_proj(out)

        return out.reshape(B, H, W, C)


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
        ) / 2


class LAFNOLayer(nnx.Module):
    def __init__(self, channels, modes, heads, activation, rngs):
        self.channels = channels
        self.modes = modes
        self.heads = heads
        self.activation = activation

        self.spectral_conv = SpectralConv2D(channels, channels, modes, rngs=rngs)
        self.linear_attention = LinearAttention2D(channels, num_heads=heads, rngs=rngs)
        self.bypass_conv = nnx.Conv(channels, channels, kernel_size=1, rngs=rngs)

    def __call__(self, x):
        return self.activation(
            self.spectral_conv(x)
            + self.linear_attention(x)
            + self.bypass_conv(x)
        ) / 3


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


class ScatteringLAFNO(nnx.Module):
    def __init__(self, dout, hidden_channels, n_hidden, modes, heads, activation, rngs):
        self.din = 1
        self.dout = dout
        self.lifting = nnx.Conv(self.din, hidden_channels, kernel_size=1, rngs=rngs)
        self.fourier_layers = nnx.List([
            LAFNOLayer(hidden_channels, modes, heads, activation, rngs)
            for _ in range(n_hidden)
        ])
        self.projection = nnx.Conv(hidden_channels, 2 * dout, kernel_size=1, rngs=rngs)

    def __call__(self, x):
        x = self.lifting(x[..., None])
        for layer in self.fourier_layers:
            x = layer(x)
        x = self.projection(x)
        x = x[..., :self.dout] + 1j * x[..., self.dout:]
        return x


if __name__ == '__main__':
    rngs = nnx.Rngs(42)
    # model = ScatteringFNO(dout=2, hidden_dims=[32] * 3, modes=16, activation=nnx.gelu, rngs=rngs)
    model = ScatteringLAFNO(dout=2, hidden_channels=32, n_hidden=3, modes=16, heads=2, activation=nnx.gelu, rngs=rngs)
    x = jax.random.uniform(rngs(), (10, 100, 100))
    y = model(x)
    print(y.shape)
