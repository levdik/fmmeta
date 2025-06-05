import numpy as np


def angle_to_standard_range(angle, convert_from_degrees=False):
    angle = np.array(angle)
    if convert_from_degrees:
        angle = angle * np.pi / 180
    return (angle - np.pi) % (2 * np.pi) - np.pi


def _generate_incidence_phase_profile(point, wavelength, inc_phi, inc_theta):
    k0 = 2 * np.pi / wavelength
    x = point[..., 0]
    y = point[..., 1]
    z = point[..., 2] if np.shape(point)[-1] > 2 else 0

    inc_path = np.sin(inc_theta) * (np.cos(inc_phi) * x + np.sin(inc_phi) * y) + np.cos(inc_theta) * z
    phase = k0 * inc_path
    return phase


def optical_path_to_focal_point(points, focal_point, xy_period=None):
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2] if np.shape(points)[-1] > 2 else 0

    if xy_period is None:
        wave_paths = np.sqrt((x - focal_point[0]) ** 2 + (y - focal_point[1]) ** 2 + (z - focal_point[2]) ** 2)
    else:
        min_x_distances = np.min(np.stack([
            np.abs(x - focal_point[0]),
            np.abs(x - focal_point[0] + xy_period),
            np.abs(x - focal_point[0] - xy_period)
        ]), axis=0)
        min_y_distances = np.min(np.stack([
            np.abs(y - focal_point[1]),
            np.abs(y - focal_point[1] + xy_period),
            np.abs(y - focal_point[1] - xy_period)
        ]), axis=0)
        z_distances = z - focal_point[2]
        wave_paths = np.sqrt(min_x_distances ** 2 + min_y_distances ** 2 + z_distances ** 2)
    return wave_paths


def generate_target_phase(points, focal_points, wavelength,
                          inc_theta=0, inc_phi=0,
                          xy_period=None):
    k0 = 2 * np.pi / wavelength

    points = np.atleast_2d(points)
    focal_points = np.atleast_2d(focal_points)

    optical_paths = np.full(len(points), np.inf)
    for focal_point in focal_points:
        optical_paths = np.minimum(
            optical_paths,
            optical_path_to_focal_point(points, focal_point, xy_period)
        )

    incidence_phase = _generate_incidence_phase_profile(points, wavelength, inc_phi=inc_phi, inc_theta=inc_theta)
    target_phase = angle_to_standard_range(incidence_phase - k0 * optical_paths)

    return target_phase
