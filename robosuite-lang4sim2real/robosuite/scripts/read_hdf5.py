import argparse

import h5py
import nexusformat.nexus as nx
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to hdf5 file")
    parser.add_argument("--show-tree", action="store_true", default=False)
    args = parser.parse_args()

    np.set_printoptions(threshold=np.inf)

    if args.show_tree:
        f = nx.nxload(args.path)
        print(f.tree)

    with h5py.File(args.path, mode='r') as h5fr:
        for obj in h5fr.keys():
            # Example of loading attributes
            print(h5fr['data'].attrs['env_info'])
            import ipdb; ipdb.set_trace()
            for task_idx in h5fr['data'].keys():
                for demo_id in h5fr[f'data/{task_idx}'].keys():
                    speeds = h5fr[f'data/{task_idx}/{demo_id}/observations/ee_pos_speed_before_success']
                    rews = np.sum(h5fr[f'data/{task_idx}/{demo_id}/rewards'])
                    if rews <= 0.0:
                        print(f"task_idx: {task_idx}, demo_id: {demo_id} rews {rews}")
                        print(f"task_idx: {task_idx}, demo_id: {demo_id} speeds {speeds[-2:]}")
