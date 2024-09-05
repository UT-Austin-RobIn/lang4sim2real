import argparse
import h5py

def get_lang_list_for_task_idx(buffer_path, task_idx):
    candidate_buffer_paths = [
        buffer_path.replace("/mnt/hdd1/albert", "/scratch/cluster/albertyu"),
        buffer_path.replace("/mnt/hdd2/albert", "/scratch/cluster/albertyu"),
        buffer_path.replace("/scratch/cluster/albertyu", "/mnt/hdd1/albert"),
        buffer_path.replace("/scratch/cluster/albertyu", "/mnt/hdd2/albert"),
        buffer_path.replace("/scratch/cluster/albertyu/minibullet_datasets/realrobot_datasets/scripted_frka_pp_2023-12-05T18-11-58_postprocessed_substeps1.hdf5", "/scratch/cluster/albertyu/minibullet_datasets/realrobot_datasets/scripted_frka_pp_2023-12-05T18-11-58_postprocessed.hdf5"),
    ]
    for buffer_path in candidate_buffer_paths:
        try:
            with h5py.File(buffer_path, mode="r") as h5fr:
                print(f"found buffer at {buffer_path}")
                lang_list = h5fr[f'data/{task_idx}'].attrs['lang_list']
            break
        except:
            print(f"could not find buffer {buffer_path}")
            lang_list = []
    print("found lang_list", lang_list)
    return lang_list


def get_lang_lists(combined_buf_path):
    buf_idx_to_lang_list_map = {}
    with h5py.File(combined_buf_path, mode="r") as h5fr:
        for combined_buf_idx in h5fr['data'].keys():
            combined_buf_idx = int(combined_buf_idx)
            orig_buf_idx = h5fr['data'].attrs['curr_idx_to_orig_buffer_src_idx_arr_map'][combined_buf_idx]
            orig_buffer_path = h5fr['data'].attrs['orig_hdf5_list'][orig_buf_idx]
            orig_buf_task_idx = h5fr['data'].attrs['curr_to_old_task_idx_arr_map'][combined_buf_idx]
            lang_list = get_lang_list_for_task_idx(orig_buffer_path, orig_buf_task_idx)
            buf_idx_to_lang_list_map[combined_buf_idx] = lang_list
    print(buf_idx_to_lang_list_map)
    return buf_idx_to_lang_list_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, required=True)
    args = parser.parse_args()

    buf_idx_to_lang_list_map = get_lang_lists(args.path)

    input("This command will overwrite the existing dataset, are you sure? CTRL+C to quit.")
    with h5py.File(args.path, mode="r+") as h5fr:
        import ipdb; ipdb.set_trace()
        for task_idx in h5fr['data'].keys():
            task_idx = int(task_idx)
            h5fr[f'data/{task_idx}'].attrs['lang_list'] = buf_idx_to_lang_list_map[task_idx]
        # for task_idx in [3, 4]:
        #     for key in h5fr[f'data/{task_idx}'].keys():
        #         h5fr[f'data/{task_idx}/{key}'].attrs['num_samples'] = h5fr[f'data/{task_idx}/{key}/actions'].shape[0]
