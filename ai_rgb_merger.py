import jax
import jax.numpy as jnp
from flax import nnx
import optax

from scattering_solver_factory import prepare_shapes_to_amplitudes_function


class MulticolorLensMerger(nnx.Module):
    def __init__(
            self,
            n_colors, n_pillars_per_side, hidden_layer_dims, rngs,
            n_pillar_params_in=1, n_pillar_params_out=2
    ):
        self.n_pillars_per_side = n_pillars_per_side
        self.n_pillar_params_in = n_pillar_params_in
        self.n_pillar_params_out = n_pillar_params_out
        self.n_colors = n_colors

        n_pillars = n_pillars_per_side ** 2
        output_dim = n_pillar_params_out * n_pillars
        input_dim = n_pillar_params_in * n_colors * n_pillars
        layer_dims = [input_dim] + hidden_layer_dims + [output_dim]

        self.layers = [
            nnx.Linear(layer_dims[i], layer_dims[i + 1], rngs=rngs)
            for i in range(len(layer_dims) - 1)
        ]

    def __call__(self, *multicolored_inputs):
        batch_size = jnp.atleast_2d(multicolored_inputs[0]).shape[0]
        multicolored_inputs = [
            multicolored_inputs[i].reshape(batch_size, -1)
            for i in range(self.n_colors)
        ]
        x = jnp.hstack(multicolored_inputs)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = nnx.leaky_relu(x)
            else:
                x = nnx.sigmoid(x)
        output_lens_params = x.reshape(
            batch_size, self.n_pillar_params_out, self.n_pillars_per_side, self.n_pillars_per_side)
        return jnp.squeeze(output_lens_params)


def train_multicolor_merging_model():
    wavelengths = [650, 550, 450]
    permittivity = 4.
    n_lens_subpixels = 8
    lens_subpixel_size = 300
    lens_thickness = 1000
    approximate_number_of_terms = 500

    model = MulticolorLensMerger(
        n_colors=len(wavelengths),
        n_pillars_per_side=n_lens_subpixels,
        hidden_layer_dims=[128, 128, 128],
        rngs=nnx.Rngs(0)
    )

    common_func_prep_kwargs = {
        'permittivity': permittivity,
        'lens_subpixel_size': lens_subpixel_size,
        'n_lens_subpixels': n_lens_subpixels,
        'lens_thickness': lens_thickness,
        'approximate_number_of_terms': approximate_number_of_terms,
        'include_reflection': True,
        'return_basis_indices': False
    }

    red_shapes_to_amps_function = prepare_shapes_to_amplitudes_function(
        wavelength=wavelengths[0], **common_func_prep_kwargs)
    green_shapes_to_amps_function = prepare_shapes_to_amplitudes_function(
        wavelength=wavelengths[1], **common_func_prep_kwargs)
    blue_shapes_to_amps_function = prepare_shapes_to_amplitudes_function(
        wavelength=wavelengths[2], **common_func_prep_kwargs)

    def ab_params_to_shapes(params):
        a, b = params * lens_subpixel_size
        ah = bh = jnp.zeros_like(a)
        shapes = jnp.stack([a, b, ah, bh], axis=-1)
        return shapes

    def a_params_to_shapes(params):
        a = params * lens_subpixel_size
        b = ah = bh = jnp.zeros_like(a)
        shapes = jnp.stack([a, b, ah, bh], axis=-1)
        return shapes

    def single_params_loss_function(red_params, green_params, blue_params, merged_rgb_params):
        red_shapes = a_params_to_shapes(red_params)
        green_shapes = a_params_to_shapes(green_params)
        blue_shapes = a_params_to_shapes(blue_params)
        merged_rgb_shapes = ab_params_to_shapes(merged_rgb_params)

        loss = (
            jnp.linalg.norm(
                red_shapes_to_amps_function(red_shapes) - red_shapes_to_amps_function(merged_rgb_shapes))
            + jnp.linalg.norm(
                green_shapes_to_amps_function(green_shapes) - green_shapes_to_amps_function(merged_rgb_shapes))
            + jnp.linalg.norm(
                blue_shapes_to_amps_function(blue_shapes) - blue_shapes_to_amps_function(merged_rgb_shapes))
        )
        return loss

    vectorized_params_loss_function = jax.vmap(single_params_loss_function)

    learning_rate = 1e-3
    batch_size = 10
    n_epochs = 3
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate))
    rng_key = jax.random.key(0)

    @nnx.jit
    def train_step(model, optimizer, *multicolored_params):
        def model_loss_function(model):
            merged_rgb_params = model(*multicolored_params)
            batch_losses = vectorized_params_loss_function(*multicolored_params, merged_rgb_params)
            loss = jnp.mean(batch_losses)
            return loss

        loss, grads = nnx.value_and_grad(model_loss_function)(model)
        optimizer.update(grads)
        return loss

    for epoch in range(n_epochs):
        rng_key, rng_base_params_subkey, rng_variation_magnitude_subkey = jax.random.split(rng_key, num=3)
        base_params = jax.random.uniform(rng_base_params_subkey, shape=(batch_size, n_lens_subpixels, n_lens_subpixels))
        variation_magnitude = jnp.abs(jax.random.normal(rng_variation_magnitude_subkey, shape=(batch_size,)))
        multicolored_params = []
        for i in range(len(wavelengths)):
            rng_key, rng_subkey = jax.random.split(rng_key)
            color_specific_params = (
                    base_params
                    + jax.random.uniform(rng_subkey, shape=base_params.shape, minval=0.5, maxval=0.5)
                    * 0.5 * variation_magnitude[:, jnp.newaxis, jnp.newaxis]
            )
            color_specific_params = jnp.clip(color_specific_params, 0., 1.)
            multicolored_params.append(color_specific_params)
        current_loss = train_step(model, optimizer, *multicolored_params)
        print(epoch, current_loss, sep='\t')


if __name__ == '__main__':
    jnp.set_printoptions(linewidth=1000)

    train_multicolor_merging_model()
