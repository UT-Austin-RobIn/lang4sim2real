import argparse
import os

from colour import Color
import numpy as np
from sklearn.manifold import TSNE
import torch

from rlkit.util.roboverse_utils import get_buffer_size_multitask
from rlkit.torch.networks.cnn import ClipWrapper
from rlkit.util.experiment_script_utils import (
    create_task_indices, init_filled_buffer)
from rlkit.util.misc_functions import tile_embs_by_batch_size
import rlkit.util.pytorch_util as ptu
import roboverse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class DictArray():
    """
    Allows conversions between a dictionary of {key --> np.array}
    into a single np.array of concatenated values
    """
    def __init__(self, data_dict):
        self.dict = data_dict
        self.convert_to_single_array()

    def convert_to_single_array(self):
        self.arr = np.array([])  # single arr we are builidng.
        self.key_indices = {}
        sorted_keys = sorted(list(self.dict.keys()))
        for k in sorted_keys:
            start = self.arr.shape[0]
            end = start + self.dict[k].shape[0]
            self.key_indices[k] = (start, end)

            if self.arr.shape[0] == 0:
                self.arr = np.array(self.dict[k])
            else:
                self.arr = np.concatenate([self.arr, self.dict[k]], axis=0)

    def input_arr_to_dict(self, in_arr):
        assert self.arr is not None  # we already built self.arr
        assert in_arr.shape[0] == self.arr.shape[0]

        out_dict = {}
        for k in self.dict:
            start, end = self.key_indices[k]
            out_dict[k] = in_arr[start:end]
            assert out_dict[k].shape[0] == self.dict[k].shape[0]

        return out_dict

    def __repr__(self):
        key_indices_str = f"key_indices: {self.key_indices}"
        arr_str = f"arr: {self.arr}"
        return key_indices_str + "\n" + arr_str


def load_videos_and_lang_by_task_idx(
        args, buffer_datas, k, task_indices, task_encoder, max_path_len,
        visual_batch_process_fn=None):

    variant = dict(
        max_path_length=max_path_len,
        task_embedding="none",
        init_task_idx=None,
        num_trajs_per_task=args.num_trajs_per_task,
        clip_tokenize_scheme="clip",
        image_dim=(args.img_size, args.img_size, 3)
    )

    num_transitions = get_buffer_size_multitask(buffer_datas)
    max_replay_buffer_size = num_transitions + 10

    kwargs = {
        "task_str_format": args.task_str_type,
        "deterministic_target_obj_referent": (
            not args.random_target_obj_referent),
    }
    if args.img_obs_key == "image_hd":
        kwargs["observation_img_hd_dim"] = args.eval_image_dim
    env = roboverse.make(
        args.env, transpose_image=True, num_tasks=args.num_tasks, **kwargs)
    observation_keys = [args.img_obs_key, 'state']
    train_embeddings_map = None
    env_task_lang_list = env.get_task_language_list()
    task_strs = [env_task_lang_list[task_idx] for task_idx in task_indices]
    task_idx_to_str_list_map = dict(zip(task_indices, task_strs))
    print("task_idx_to_str_map", task_idx_to_str_list_map)

    internal_keys = []

    buf = init_filled_buffer(
        buffer_datas, variant, max_replay_buffer_size,
        env, task_indices, observation_keys, internal_keys, args.num_tasks,
        train_embeddings_map, success_only=True, video_encoder=None)

    kwargs = {}
    if task_encoder is not None:
        kwargs['frame_ranges'] = task_encoder.frame_ranges
    else:
        kwargs['frame_ranges'] = args.vid_enc_frame_ranges

    if visual_batch_process_fn is None:
        visual_batch_process_fn = task_encoder.process_visual_batch

    task_idx_to_demos_map = load_task_idx_to_demos_map(
        buf, task_indices, variant['num_trajs_per_task'],
        k, kwargs, visual_batch_process_fn)
    return task_idx_to_demos_map, task_idx_to_str_list_map


def load_task_idx_to_demos_map(
        buf, task_indices, num_trajs_per_task, k, kwargs,
        visual_batch_process_fn, ext):
    task_idx_to_demos_map = dict()
    unflatten_im = bool(ext == "npy")
    for task_idx in task_indices:
        demos = buf.random_trajectory(
            task=task_idx, batch_size=num_trajs_per_task,
            with_replacement=False, k=k, **kwargs)
        task_idx_to_demos_map[task_idx] = visual_batch_process_fn(
            demos, unflatten_im)
    return task_idx_to_demos_map


def load_model(args):
    model = torch.load(args.ckpt, map_location=torch.device('cuda'))
    video_encoder = model['trainer/task_encoder']
    video_encoder.encoder = video_encoder.encoder.eval()
    return video_encoder


def save_plot(out_dir, plot_name, epoch):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    epoch_str = str(epoch).zfill(4)
    save_path = os.path.join(
        out_dir, f'tsne_{plot_name}_epoch_{epoch_str}.png')
    print("save_path", save_path)
    plt.savefig(save_path, dpi=300)
    return save_path


def remove_outliers_from_dict(embeddings_tsne_by_task_id, z_thresh):
    all_embs_combined = np.concatenate(
        [embs for task_id, embs in embeddings_tsne_by_task_id.items()], axis=0)

    # IQR-based outlier filtering
    # q1, q3 = np.percentile(all_embs_combined, [25, 75], axis=0)
    # iqr = q3 - q1
    # iqr_multiplier = 1.4
    # lb = q1 - iqr_multiplier * iqr
    # ub = q3 + iqr_multiplier * iqr

    # Z-score based outlier filtering
    m = np.mean(all_embs_combined, axis=0)
    s = np.std(all_embs_combined, axis=0)
    lb = m - z_thresh * s
    ub = m + z_thresh * s

    num_outliers = 0
    outlier_removed_embeddings_tsne_by_task_id = {}
    for task_id, embeddings_tsne in embeddings_tsne_by_task_id.items():
        outlier_removed_embeddings_tsne = []
        for emb in embeddings_tsne:
            if np.all(emb >= lb) and np.all(emb <= ub):
                outlier_removed_embeddings_tsne.append(emb)
            else:
                num_outliers += 1
        if len(outlier_removed_embeddings_tsne) > 0:
            outlier_removed_embeddings_tsne_by_task_id[task_id] = np.array(
                outlier_removed_embeddings_tsne)
            # print(f"outlier_removed_embeddings_tsne_by_task_id[{task_id}].shape",
            #     outlier_removed_embeddings_tsne_by_task_id[task_id].shape)

    print("Num Outliers Removed:", num_outliers)
    return outlier_removed_embeddings_tsne_by_task_id


def get_z_embs(
        task_idx_to_demos_map, task_idx_to_str_list_map, task_encoder, env,
        modality_key="video"):
    z_embs_by_task_id = {}
    for task_id, demos_dict in task_idx_to_demos_map.items():
        # (n, 48, 96, 3) --> (n, 3, 48, 96)
        for key in demos_dict:
            if not torch.is_tensor(demos_dict[key]):
                demos_dict[key] = ptu.from_numpy(demos_dict[key])

        # Maybe add lang to encoder input batch
        if task_encoder.num_film_inputs > 0:
            traj_emb_targets = env.get_target_emb(task_id)
            traj_emb_targets = np.expand_dims(traj_emb_targets, axis=0)
            n = demos_dict['video'].shape[0]
            traj_emb_targets = tile_embs_by_batch_size(traj_emb_targets, n)
            assert all([
                traj_emb_targets.shape[0] == val.shape[0]
                for key, val in demos_dict.items()])
            demos_dict["lang"] = traj_emb_targets
        elif isinstance(task_encoder.encoder, ClipWrapper):
            # Language is passed into encoder but we're not
            # getting the language emb outputted
            task_lang_tokens = (
                task_encoder.encoder.get_task_lang_tokens_matrix([task_id]))
            demos_dict["clip_lang"] = task_lang_tokens

        emb_dict = task_encoder(demos_dict, train_mode=False)
        z_embs_by_task_id[task_id] = np.array(
            emb_dict[modality_key].detach().cpu())
    return z_embs_by_task_id


def get_tsne_from_z_embs(z_embs_by_task_id):
    z_dict_arr = DictArray(z_embs_by_task_id)
    merged_z_arr = z_dict_arr.arr
    embeddings_tsne = TSNE(
        n_components=2,
        learning_rate=200.0,
        init='random').fit_transform(merged_z_arr)
    print("embeddings_tsne.shape", embeddings_tsne.shape)
    embeddings_tsne_by_task_id = z_dict_arr.input_arr_to_dict(embeddings_tsne)
    return embeddings_tsne_by_task_id


def plot_tsne_embs(tsne_embs, axs, color, alpha=1.0, marker='o'):
    X = tsne_embs[:, 0]
    Y = tsne_embs[:, 1]
    axs.scatter(X, Y, color=color, alpha=alpha, marker=marker)


def plot_z_by_task_id(
        z_embs_by_task_id, train_task_indices, eval_task_indices,
        out_dir, plot_name, epoch, exp_title):

    # filter only tasks we want to plot
    z_embs_by_task_id_filtered = {}
    for task_id in z_embs_by_task_id:
        if task_id in eval_task_indices:
            z_embs_by_task_id_filtered[task_id] = z_embs_by_task_id[task_id]

    tsne_embs_by_task_id = get_tsne_from_z_embs(z_embs_by_task_id_filtered)

    plt.plot()
    num_r, num_c = 1, 1
    fig, axs = plt.subplots(num_r, num_c)
    fig.set_size_inches(10, 10)
    axs.set_title(f'Eval Clusters by Task ID: {exp_title}, Epoch {epoch}')

    tsne_embs_by_task_id = remove_outliers_from_dict(
        tsne_embs_by_task_id, z_thresh=3.0)

    # available markers: [
    # 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd',
    # 'P', 'X']
    markers = ['o', 'v', '^', '<', '>', '*', 'X']
    colors = list(
        Color("red").range_to(Color("purple"), len(tsne_embs_by_task_id)))
    for i, (task_id, embeddings_tsne) in enumerate(
            tsne_embs_by_task_id.items()):
        plot_tsne_embs(
            embeddings_tsne, axs, colors[i].get_rgb(), alpha=1.0,
            marker=markers[i % len(markers)])

    # Annotate after all dots are plotted
    for task_id, embeddings_tsne in tsne_embs_by_task_id.items():
        X = embeddings_tsne[:, 0]
        Y = embeddings_tsne[:, 1]
        i = 0
        axs.annotate(str(task_id), (X[i], Y[i]))

    return save_plot(out_dir, plot_name, epoch)


def plot_z_by_train_test(
        z_embs_by_task_id, train_task_indices, eval_task_indices,
        out_dir, plot_name, epoch, exp_title):

    # filter only tasks we want to plot
    z_embs_by_split = {"train": [], "eval": []}
    for task_id in z_embs_by_task_id:
        if task_id in train_task_indices:
            z_embs_by_split["train"].append(z_embs_by_task_id[task_id])
        elif task_id in eval_task_indices:
            z_embs_by_split["eval"].append(z_embs_by_task_id[task_id])

    z_embs_by_split_processed = {}
    for split in z_embs_by_split:
        if len(z_embs_by_split[split]) == 0:
            continue
        z_embs_by_split_processed[split] = np.concatenate(
            z_embs_by_split[split], axis=0)

    tsne_embs_by_split = get_tsne_from_z_embs(z_embs_by_split_processed)

    plt.plot()
    num_r, num_c = 1, 1
    fig, axs = plt.subplots(num_r, num_c)
    fig.set_size_inches(10, 10)
    axs.set_title(f'Train/Test Task Clusters: {exp_title}, Epoch {epoch}')

    tsne_embs_by_split = remove_outliers_from_dict(
        tsne_embs_by_split, z_thresh=3.0)

    colors = [Color("green"), Color("red")]
    for i, (split, embeddings_tsne) in enumerate(tsne_embs_by_split.items()):
        plot_tsne_embs(
            embeddings_tsne, axs, colors[i].get_rgb(), alpha=0.5, marker='o')
    plt.legend(list(tsne_embs_by_split.keys()))

    return save_plot(out_dir, plot_name, epoch)


def make_plots(
        train_task_indices, eval_task_indices, task_idx_to_demos_map,
        task_idx_to_str_list_map, task_encoder, env, epoch, output_fpath,
        modality_key):
    assert set(eval_task_indices).issubset(task_idx_to_demos_map.keys())
    plt.switch_backend('agg')
    print("eval_task_indices", eval_task_indices)
    z_embs_by_task_id = get_z_embs(
        task_idx_to_demos_map, task_idx_to_str_list_map, task_encoder, env,
        modality_key)
    plot_z_by_task_id(
        z_embs_by_task_id, train_task_indices, eval_task_indices,
        output_fpath, f"task_ids_{modality_key}", epoch, modality_key)
    plot_z_by_train_test(
        z_embs_by_task_id, train_task_indices, eval_task_indices,
        output_fpath, f"train_test_{modality_key}", epoch, modality_key)


def plot_tsne_embs_by_stage(
        traj_embs, traj_lang_idxs, out_dir, plot_name,
        idx_annotation_interval=10, plot_cluster_means=False,
        title='Trajectory image embs by stage',
        label_scheme="traj_ts"):
    def save_plot(out_dir, plot_name):
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        save_path = os.path.join(
            out_dir, f'tsne_{plot_name}.png')
        print("save_path", save_path)
        plt.savefig(save_path, dpi=300)
        return save_path

    assert traj_embs.shape[0] == traj_lang_idxs.shape[0]
    tsne_embs = TSNE(
        n_components=2,
        learning_rate=200.0,
        init='random').fit_transform(traj_embs)

    plt.plot()
    plt.clf()
    num_r, num_c = 1, 1
    fig, axs = plt.subplots(num_r, num_c)
    fig.set_size_inches(10, 10)
    axs.set_title(title)

    markers = ['o', 'v', '^', '<', '>', '*', 'X']
    colors = list(
        Color("red").range_to(Color("purple"), len(set(traj_lang_idxs))))

    X = tsne_embs[:, 0]
    Y = tsne_embs[:, 1]
    alpha = 1.0

    for i, lang_stage_num in enumerate(sorted(list(set(traj_lang_idxs)))):
        indices_of_lang_stage_num = np.where(
            traj_lang_idxs == lang_stage_num)[0]
        stage_ts_start = np.min(indices_of_lang_stage_num)
        stage_ts_end = np.max(indices_of_lang_stage_num)
        color = colors[i % len(set(traj_lang_idxs))].get_rgb()
        if label_scheme == "traj_ts":
            label = (
                f"stage {lang_stage_num} ({stage_ts_start}, {stage_ts_end})")
        elif label_scheme == "task":
            label = f"task {lang_stage_num}"
        axs.scatter(
            X[indices_of_lang_stage_num], Y[indices_of_lang_stage_num],
            color=color, alpha=alpha, marker=markers[0],
            label=label)
        if plot_cluster_means:
            X_mean = np.mean(X[indices_of_lang_stage_num])
            Y_mean = np.mean(Y[indices_of_lang_stage_num])
            axs.scatter(
                X_mean, Y_mean, color=color, alpha=0.7, marker=markers[2],
                s=150)

    # Annotate after all dots are plotted
    if label_scheme == "traj_ts":
        for i in range(0, X.shape[0], idx_annotation_interval):
            stage_num = traj_lang_idxs[i]
            color = colors[stage_num % len(set(traj_lang_idxs))].get_rgb()
            axs.annotate(str(i), (X[i], Y[i]), color=color)

    axs.legend()

    save_plot(out_dir, plot_name)
    plt.clf()
    return tsne_embs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer", type=str, required=True)
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--num-tasks", type=int, required=True)
    parser.add_argument(
        "--eval-task-idx-intervals", nargs="+", type=str, default=[])
    parser.add_argument("--output-fpath", type=str, required=True)
    parser.add_argument(
        "--img-obs-key", type=str, default="image",
        choices=['image', 'image_hd'])
    parser.add_argument("--img-size", type=int, default=48)
    parser.add_argument("--eval-image-dim", type=int, default=None)
    parser.add_argument("--task-str-type", type=str, default="pick_place_obj")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--num-trajs-per-task", type=int, default=10)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--clip-ckpt", type=str, default=None)
    parser.add_argument("--max-path-len", type=int, default=30)
    args = parser.parse_args()

    task_encoder = load_model(args)
    task_idx_to_demos_map, task_idx_to_str_list_map = (
        load_videos_and_lang_by_task_idx(
            args, task_encoder, max_path_len=args.max_path_len))
    train_task_indices, eval_task_indices = create_task_indices(
        args.eval_task_idx_intervals, args.num_tasks)
    make_plots(
        train_task_indices, eval_task_indices, task_idx_to_demos_map,
        task_idx_to_str_list_map, task_encoder, 0, args.output_fpath)
