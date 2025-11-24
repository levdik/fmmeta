import numpy as np
import h5py


if __name__ == '__main__':
    topology_params = []
    field_amps = []

    dataset_name = 'green'

    for i in range(33):
        with h5py.File(f'temp_batches/{dataset_name}_{i}.hdf5') as f:
            topology_params.append(f['topology_params'][()])
            field_amps.append(f['field_amps'][()])

    topology_params = np.concatenate(topology_params, axis=0)
    field_amps = np.concatenate(field_amps, axis=0)

    print(type(topology_params))
    print(topology_params.shape)
    print(topology_params.dtype)

    print(type(field_amps))
    print(field_amps.shape)
    print(field_amps.dtype)

    with h5py.File(f'{dataset_name}.hdf5', 'a') as f:
        f.create_dataset('topology_params', data=topology_params)
        f.create_dataset('field_amps', data=field_amps)
