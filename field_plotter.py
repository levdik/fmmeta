import numpy as np

from phase_profile_manager import angle_to_standard_range

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.axes_grid1 import make_axes_locatable

import warnings


def plot_field_amp_phase(*args, wavelength_nm=None, map_bounds=(0, 1, 0, 1), amp_min=0, amp_max=None):
    n_fields = len(args)
    fig, ax = plt.subplots(n_fields, 2, squeeze=False)
    for i, field in enumerate(args):
        plot_phase_map(
            fig, ax[i, 0], np.angle(field),
            map_bounds=map_bounds)
        plot_amplitude_map(
            fig, ax[i, 1], np.abs(field),
            map_bounds=map_bounds, wavelength_nm=wavelength_nm, vmin=amp_min, vmax=amp_max)


def plot_field_amp_phase_difference_db(field1, field2, map_bounds=(0, 1, 0, 1), **kwargs):
    fig, ax = plt.subplots(1, 2)
    phase_plot = ax[0].imshow(10 * np.log10(angle_to_standard_range(np.angle(field1) - np.angle(field2))),
                              extent=map_bounds, origin='lower', interpolation='nearest', cmap='inferno', **kwargs)
    amp_plot = ax[1].imshow(10 * np.log10(np.abs(np.abs(field1) - np.abs(field2))),
                            extent=map_bounds, origin='lower', interpolation='nearest', cmap='inferno', **kwargs)
    fig.colorbar(phase_plot)
    fig.colorbar(amp_plot)


def plot_phase_map(figure, axes, phase_map, map_bounds=(0, 1, 0, 1), cbar=True):
    if len(map_bounds) == 2:
        map_bounds = list(map_bounds) * 2

    phase_plot = axes.imshow(
        phase_map,
        extent=map_bounds,
        origin='lower',
        interpolation='nearest',
        vmin=-np.pi, vmax=np.pi,
        cmap='twilight'
    )
    if cbar:
        phase_cbar = figure.colorbar(phase_plot, label=f'Phase, rad', ticks=[-np.pi, 0, np.pi])
        phase_cbar.set_ticklabels(['-π', '0', 'π'])

    return phase_plot


def plot_amplitude_map(figure, axes, amplitude_map,
                       wavelength_nm=None, map_bounds=(0, 1, 0, 1),
                       vmin=0, vmax=None, cbar=True):
    if len(map_bounds) == 2:
        map_bounds = list(map_bounds) * 2

    if wavelength_nm is None:
        cmap = LinearSegmentedColormap.from_list('white_laser', ['black', 'white'])
    else:
        cmap = visible_amplitude_colormap(wavelength_nm)

    amp_plot = axes.imshow(
        amplitude_map,
        extent=map_bounds,
        origin='lower',
        interpolation='nearest',
        vmin=vmin, vmax=vmax,
        cmap=cmap
    )
    if cbar:
        figure.colorbar(amp_plot, label=f'Amplitude')

    return amp_plot


def visible_amplitude_colormap(wavelength_nm):
    cmap = LinearSegmentedColormap.from_list(
        'colored_laser',
        ['black', wavelength_to_color(wavelength_nm)]
    )
    return cmap


def wavelength_to_color(wavelength_nm):
    if np.any(wavelength_nm < 350) or np.any(wavelength_nm > 800):
        warnings.warn('Wavelength out of visible range, returning black.')

    wavelength_to_red_vertices = np.array([
        [350, 0], [380, 0.4], [400, 0.5], [440, 0], [510, 0], [580, 1], [700, 1], [800, 0]])
    wavelength_to_green_vertices = np.array([
        [350, 0], [440, 0], [490, 1], [580, 1], [645, 0], [800, 0]])
    wavelength_to_blue_vertices = np.array([
        [350, 0], [380, 0.4], [420, 1], [490, 1], [510, 0], [800, 0]])

    colors = np.vstack([
        np.interp(wavelength_nm, wavelength_to_red_vertices[:, 0], wavelength_to_red_vertices[:, 1]),
        np.interp(wavelength_nm, wavelength_to_green_vertices[:, 0], wavelength_to_green_vertices[:, 1]),
        np.interp(wavelength_nm, wavelength_to_blue_vertices[:, 0], wavelength_to_blue_vertices[:, 1])
    ]).T
    if len(np.shape(wavelength_nm)) == 0:
        colors = colors[0]
    return colors


def plot_complex_matrix_hsv(figure, axes, complex_matrix, cbar=True, map_bounds=(0, 1, 0, 1), amp_max=None):
    if len(map_bounds) == 2:
        map_bounds = list(map_bounds) * 2

    phase = np.angle(complex_matrix)
    amplitude = np.abs(complex_matrix)

    hue = (phase + np.pi) / (2 * np.pi)
    amplitude_normalized = amplitude / amplitude.max()

    hsv_image = np.zeros((*complex_matrix.shape, 3))
    hsv_image[..., 0] = hue
    hsv_image[..., 1] = 1.0
    hsv_image[..., 2] = amplitude_normalized

    rgb_image = hsv_to_rgb(hsv_image)

    img = axes.imshow(rgb_image, origin='lower', aspect='equal', extent=map_bounds, vmax=amp_max)

    if cbar:
        complex_plot_hsv_color_bars(figure, axes, amp_max if amp_max is not None else amplitude.max())

    return img


def complex_plot_hsv_color_bars(figure, axes, amp_max):
    divider = make_axes_locatable(axes)

    cax_amp = divider.append_axes("right", size="5%", pad=0.4)
    cbar_amp = figure.colorbar(
        plt.cm.ScalarMappable(cmap='Greys_r', norm=plt.Normalize(vmin=0, vmax=amp_max)),
        cax=cax_amp)
    cbar_amp.ax.xaxis.set_label_position('bottom')
    cbar_amp.ax.xaxis.set_ticks_position('none')
    cbar_amp.ax.set_xlabel("amp", rotation=0)

    cax_phase = divider.append_axes("right", size="5%", pad=0.33)
    phase_cmap = plt.get_cmap("hsv")
    cbar_phase = figure.colorbar(
        plt.cm.ScalarMappable(cmap=phase_cmap, norm=plt.Normalize(vmin=-np.pi, vmax=np.pi)),
        cax=cax_phase, orientation='vertical')
    cbar_phase.ax.xaxis.set_label_position('bottom')
    cbar_phase.ax.xaxis.set_ticks_position('none')
    cbar_phase.ax.set_xlabel("arg", rotation=0)
    cbar_phase.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cbar_phase.set_ticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
