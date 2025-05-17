import jax
import jax.numpy as jnp


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


def _cross_filling_map(
        relative_outer_side_width: jnp.ndarray,
        relative_corner_width: jnp.ndarray,
        n_pixels: int
) -> jnp.ndarray:
    assert jnp.shape(relative_outer_side_width) == ()
    assert jnp.shape(relative_corner_width) == ()

    cross_width = 2 * relative_corner_width + relative_outer_side_width
    horizontal_box_filling_map = _box_filling_map(
        relative_width=cross_width,
        relative_height=relative_outer_side_width,
        n_pixels=n_pixels
    )
    vertical_box_filling_map = horizontal_box_filling_map.T
    cross_filling_map = 1. - (1. - vertical_box_filling_map) * (1. - horizontal_box_filling_map)
    return cross_filling_map


def _cross_with_hole_filling_map(
        relative_shape: jnp.ndarray,
        n_pixels: int
) -> jnp.ndarray:
    cross_filling = _cross_filling_map(
        relative_outer_side_width=relative_shape[0],
        relative_corner_width=relative_shape[1],
        n_pixels=n_pixels
    )
    hole_filling = 1. - _cross_filling_map(
        relative_outer_side_width=relative_shape[2],
        relative_corner_width=relative_shape[3],
        n_pixels=n_pixels
    )
    cross_with_hole_filling = cross_filling * hole_filling
    return cross_with_hole_filling


def _filling_to_permittivity(
        filling: jnp.ndarray,
        permittivity: float,
        permittivity_ambience: float
) -> jnp.ndarray:
    ref_ind_media = jnp.sqrt(permittivity)
    ref_ind_ambience = jnp.sqrt(permittivity_ambience)
    ref_ind_equivalent = filling * ref_ind_media + (1. - filling) * ref_ind_ambience
    permittivity_equivalent = ref_ind_equivalent ** 2
    return permittivity_equivalent


def generate_pillar_permittivity_map(
        shape: jnp.ndarray,
        map_size: float,
        permittivity: float,
        permittivity_ambience: float = 1.,
        n_pixels: int = 100
):
    relative_shape = shape / map_size
    pillar_filling = _cross_with_hole_filling_map(relative_shape, n_pixels)
    return _filling_to_permittivity(pillar_filling, permittivity, permittivity_ambience)


_vmap_generate_pillar_permittivity_map = jax.vmap(
    generate_pillar_permittivity_map,
    in_axes=(0, None, None, None, None)
)


def generate_lens_permittivity_map(
        shapes: jnp.ndarray,
        sub_pixel_size: float,
        n_lens_subpixels: int,
        permittivity_pillar,
        permittivity_ambience=1.,
        n_samples_per_subpixel=100
) -> jnp.ndarray:
    shapes = shapes.reshape(-1, 4)
    permittivity_map_blocks_flat = _vmap_generate_pillar_permittivity_map(
        shapes, sub_pixel_size, permittivity_pillar, permittivity_ambience, n_samples_per_subpixel
    )
    permittivity_map_blocks = permittivity_map_blocks_flat.reshape(
        n_lens_subpixels, n_lens_subpixels, n_samples_per_subpixel, n_samples_per_subpixel
    )
    lens_permittivity_map = permittivity_map_blocks.transpose(0, 2, 1, 3).reshape(
        n_lens_subpixels * n_samples_per_subpixel, n_lens_subpixels * n_samples_per_subpixel
    )
    return lens_permittivity_map


def generate_pillar_center_positions(
        lens_subpixel_size: float,
        n_lens_subpixels: int
):
    full_lens_size = n_lens_subpixels * lens_subpixel_size
    pillar_centers_x = jnp.linspace(
        -full_lens_size / 2, full_lens_size / 2,
        n_lens_subpixels, endpoint=False
    ) + lens_subpixel_size / 2
    pillar_centers = jnp.array([(x, y, 0) for x in pillar_centers_x for y in pillar_centers_x])

    return pillar_centers


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    permittivity_map = generate_lens_permittivity_map(
        shapes=jnp.array([
            [[100, 100, 0, 0], [200, 0, 0, 0]],
            [[300, 0, 200, 0], [200, 50, 20, 10]]
        ], dtype=float),
        sub_pixel_size=400.,
        n_lens_subpixels=2,
        permittivity_pillar=4.
    )

    plt.imshow(permittivity_map)
    plt.colorbar()
    plt.show()
