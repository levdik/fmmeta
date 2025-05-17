import numpy as np


def find_best_fit_pillar(target_phases, lib_transmissions, return_distance=False):
    if len(np.shape(lib_transmissions)) == 1:
        lib_transmissions = np.reshape(lib_transmissions, [-1, 1])

    distance_list = []
    best_fit_index_list = []

    for target_phase in target_phases:
        distances = np.linalg.norm(np.abs(np.exp(1j * target_phase) - lib_transmissions), axis=-1)
        best_fit_index = np.argmin(distances)
        best_fit_index_list.append(best_fit_index)
        distance_list.append(distances[best_fit_index])

    if return_distance:
        return best_fit_index_list, distance_list
    return best_fit_index_list


def optimize_target_phase_shifts(target_phases, lib_transmissions):
    optimization_history = []

    def loss(shifts):
        best_fit_indices, distances = find_best_fit_pillar(
            target_phases + shifts, lib_transmissions, return_distance=True)
        loss_value = np.linalg.norm(distances) / len(target_phases)
        optimization_history.append((shifts, loss_value))
        return loss_value

    n_div = 30
    min_loss = np.inf
    previous_min_arg = [0, 0, 0]
    min_arg = None
    range_size = np.pi

    for _ in range(3):
        search_range = np.linspace(-range_size, range_size, n_div)
        for dx1 in previous_min_arg[0] + search_range:
            for dx2 in previous_min_arg[1] + search_range:
                for dx3 in previous_min_arg[2] + search_range:
                    current_loss = loss([dx1, dx2, dx3])
                    if current_loss < min_loss:
                        min_loss = current_loss
                        min_arg = [dx1, dx2, dx3]
        range_size /= n_div
        previous_min_arg = min_arg

    return min_arg


def generate_transmission_library():
    # TODO
    pass
