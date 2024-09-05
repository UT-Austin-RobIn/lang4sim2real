import argparse

import h5py


def downsample_substeps_in_grp(grp, substep_downsample):
    for key in grp.keys():
        if isinstance(grp[key], h5py.Dataset):
            # A numpy array-like thing with a shape
            # data is different shape, so need to delete and recreate dataset
            downsampled_data = grp[key][::substep_downsample][()]
            del grp[key]
            grp.create_dataset(key, data=downsampled_data)
        elif isinstance(grp[key], h5py.Group):
            downsample_substeps_in_grp(grp[key], substep_downsample)
        else:
            print(f"skipping downsampling for {key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, required=True)
    parser.add_argument("--substep-downsample", type=int, required=True)
    args = parser.parse_args()

    input(
        "This command will overwrite the existing dataset, are you sure? "
        "CTRL+C to quit.")
    with h5py.File(args.path, mode="r+") as h5fr:
        downsample_substeps_in_grp(h5fr['data'], args.substep_downsample)
        # relabel num_samples
        for task_idx in h5fr['data'].keys():
            for demo_key in h5fr[f'data/{task_idx}'].keys():
                h5fr[f'data/{task_idx}/{demo_key}'].attrs['num_samples'] = (
                    h5fr[f'data/{task_idx}/{demo_key}/actions'].shape[0])
