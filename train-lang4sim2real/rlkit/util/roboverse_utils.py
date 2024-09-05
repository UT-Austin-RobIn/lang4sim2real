import os
import os.path as osp

import numpy as np
import psutil
import torch

from rlkit.core import logger


def process_image_arr_from_buffer(image):
    if len(image.shape) == 3:
        image = np.transpose(image, [2, 0, 1])
        image = image.flatten()
    elif len(image.shape) == 4:
        image = np.transpose(image, [0, 3, 1, 2])
        image = image.reshape(image.shape[0], -1)
    else:
        assert len(image.shape) < 3

    if np.mean(image) > 5:
        image = image / 255.0
    return image


def process_torch_image_arr_from_buffer(image):
    if len(image.shape) == 4:
        image = torch.permute(image, (0, 3, 1, 2))
        image = image.reshape(image.shape[0], -1)
    else:
        assert len(image.shape) < 3

    if torch.mean(image) > 5:
        image = image / 255.0
    return image


def process_keys(observations, observation_keys):
    output = []
    for i in range(len(observations)):
        observation = dict()
        for key in observation_keys:
            if key in ['image', 'image_hd']:
                image = observations[i][key]
                observation[key] = process_image_arr_from_buffer(image)
            else:
                observation[key] = np.array(observations[i][key])

        output.append(observation)
    return output


def add_multitask_data_to_multitask_buffer_v2(
        data, replay_buffer, observation_keys, num_tasks, success_only=False):
    num_paths = 0
    for j in range(len(data)):
        assert len(data[j]['actions']) == len(data[j]['observations']) == len(
            data[j]['next_observations'])

        task_idx = data[j]['env_infos'][0]['task_idx']

        if success_only and data[j]['rewards'][-1] == 0:
            continue

        for i in range(len(data[j]['observations'])):
            data[j]['observations'][i]['one_hot_task_id'] = np.array(
                [0] * num_tasks)
            data[j]['observations'][i]['one_hot_task_id'][task_idx] = 1
            data[j]['next_observations'][i]['one_hot_task_id'] = np.array(
                [0] * num_tasks)
            data[j]['next_observations'][i]['one_hot_task_id'][task_idx] = 1

        path = dict(
            rewards=[np.asarray([r]) for r in data[j]['rewards']],
            actions=data[j]['actions'],
            terminals=[np.asarray([t]) for t in data[j]['terminals']],
            observations=process_keys(
                data[j]['observations'], observation_keys),
            next_observations=process_keys(
                data[j]['next_observations'], observation_keys),
        )

        replay_buffer.task_buffers[task_idx].add_path(path)
        num_paths += 1
    print("Total number of paths in multitask buffer:", num_paths)


def add_multitask_data_to_multitask_buffer_v3(
        data, replay_buffer, observation_keys, num_tasks, embeddings,
        success_only=False):
    num_paths = 0
    for j in range(len(data)):
        assert len(data[j]['actions']) == len(data[j]['observations']) == len(
            data[j]['next_observations'])

        task_idx = data[j]['env_infos'][0]['task_idx']

        if embeddings is not None:
            for i in range(len(data[j]['observations'])):
                data[j]['observations'][i]['lang_embedding'] = (
                    embeddings[task_idx])
                data[j]['next_observations'][i]['lang_embedding'] = (
                    embeddings[task_idx])

        if success_only and data[j]['rewards'][-1] == 0:
            continue

        path = dict(
            rewards=[np.asarray([r]) for r in data[j]['rewards']],
            actions=data[j]['actions'],
            terminals=[np.asarray([t]) for t in data[j]['terminals']],
            observations=process_keys(
                data[j]['observations'], observation_keys),
            next_observations=process_keys(
                data[j]['next_observations'], observation_keys),
        )

        replay_buffer.task_buffers[task_idx].add_path(path)
        num_paths += 1
    print("Total number of paths in multitask buffer:", num_paths)


def add_data_to_multitask_buffer_v3(
        data, replay_buffer, observation_keys, num_tasks, success_only=False):
    num_paths = 0
    for j in range(len(data)):
        assert len(data[j]['actions']) == len(data[j]['observations']) == len(
            data[j]['next_observations'])

        task_idx = data[j]['env_infos'][0]['task_idx']

        if success_only and data[j]['rewards'][-1] == 0:
            continue

        path = dict(
            rewards=[np.asarray([r]) for r in data[j]['rewards']],
            actions=data[j]['actions'],
            terminals=[np.asarray([t]) for t in data[j]['terminals']],
            observations=process_keys(
                data[j]['observations'], observation_keys),
            next_observations=process_keys(
                data[j]['next_observations'], observation_keys),
        )

        replay_buffer.task_buffers[task_idx].add_path(path)
        num_paths += 1
    print("Total number of paths in multitask buffer:", num_paths)


class VideoSaveFunctionBullet:
    def __init__(self, variant):
        self.logdir = logger.get_snapshot_dir()
        self.dump_video_kwargs = variant.get("dump_video_kwargs", dict())
        self.save_period = self.dump_video_kwargs.pop('save_video_period', 5)
        self.eval_image_dim = variant['eval_image_dim']

    def __call__(self, algo, epoch):
        if epoch % self.save_period == 0 or epoch == algo.num_epochs:
            video_dir = osp.join(self.logdir,
                                 'videos_eval/{epoch}/'.format(epoch=epoch))
            eval_paths = algo.eval_data_collector.get_epoch_paths()
            dump_video_basic(video_dir, eval_paths, self.eval_image_dim)


class PlotSaveFunctionBullet:
    def __init__(self, variant, eval_env):
        from rlkit.util.visualization_utils import (
            make_plots, load_task_idx_to_demos_map)
        self.make_plot_fn = make_plots
        self.logdir = logger.get_snapshot_dir()
        self.plot_kwargs = variant.get("plot_kwargs", dict())
        self.save_period = self.plot_kwargs.pop('plot_period', 5)
        self.task_encoder = variant['trainer_kwargs']['task_encoder']
        self.task_idx_to_str_list_map = variant['buffer_task_strs']
        self.task_idx_to_str_list_map.update(variant['eval_task_strs'])
        self.target_buffer = variant['target_buffer_obj']
        kwargs = {
            'frame_ranges': variant['vid_enc_frame_ranges'],
        }
        self.task_idx_to_demos_map = load_task_idx_to_demos_map(
            self.target_buffer,
            variant['target_buffer_task_idxs'],
            variant['eval_video_batch_size'], self.task_encoder.k,
            kwargs, self.task_encoder.process_visual_batch,
            variant['buffer_ext_dict']['target'])
        self.train_task_indices = variant['train_task_indices']
        self.eval_task_indices = variant['eval_task_indices']
        self.visual_modalities = variant['vid_enc_visual_modalities']
        self.eval_env = eval_env

    def __call__(self, algo, epoch):
        if epoch % self.save_period == 0 or epoch == algo.num_epochs:
            save_dir = osp.join(self.logdir, 'plots')
            for mod in self.visual_modalities:
                self.make_plot_fn(
                    self.train_task_indices, self.eval_task_indices,
                    self.task_idx_to_demos_map, self.task_idx_to_str_list_map,
                    self.task_encoder, self.eval_env, epoch, save_dir, mod)


def dump_video_basic(video_dir, paths, eval_image_dim):
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    import cv2
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    for i, path in enumerate(paths):
        video = path['next_observations']
        frame_list = []
        for frame in video:
            if eval_image_dim is not None:
                frame_list.append(frame['image_hd'])
            else:
                frame_list.append(frame['image'])
        frame_list = np.asarray(frame_list)
        video_len = frame_list.shape[0]
        n_channels = 3
        imsize = int(np.sqrt(frame_list.shape[1] / n_channels))
        assert imsize*imsize*n_channels == frame_list.shape[1]

        video = frame_list.reshape(video_len, n_channels, imsize, imsize)
        video = np.transpose(video, (0, 2, 3, 1))
        video = (video*255.0).astype(np.uint8)
        final_rew_str = str(path['rewards'][-1, 0]).replace(".", "_")
        filename = osp.join(video_dir, f'{i}.fr{final_rew_str}.mp4')
        FPS = float(np.ceil(video_len/3.0))
        writer = cv2.VideoWriter(filename, fourcc, FPS, (imsize, imsize))

        # Create final-reward-based dir to save video again in reward directory
        video_dir_prefix_strs = os.path.dirname(video_dir).split("/")[:-1]
        epoch_num = os.path.dirname(video_dir).split("/")[-1]
        rew_video_dir = os.path.join(
            "/", *video_dir_prefix_strs, f"fr{final_rew_str}")
        if not os.path.exists(rew_video_dir):
            os.makedirs(rew_video_dir)
        rew_dir_filename = osp.join(rew_video_dir, f'epoch{epoch_num}_{i}.mp4')

        for j in range(video.shape[0]):
            writer.write(cv2.cvtColor(video[j], cv2.COLOR_RGB2BGR))
        writer = None
        command = f"cp {filename} {rew_dir_filename}"
        os.system(command)


def get_buffer_size_multitask(
        data_list, success_only=False, output_type="max", add_offset=10,
        ext=""):
    if ext == "hdf5":
        return None
    elif ext == "npy":
        assert output_type in ["max", "dict"]
        num_transitions = {}
        assert isinstance(data_list, list)
        for data in data_list:
            for i in range(len(data)):
                task_id = data[i]['env_infos'][0]['task_idx']
                if task_id not in num_transitions.keys():
                    num_transitions[task_id] = add_offset
                if success_only and data[i]['rewards'][-1] == 0:
                    continue
                num_transitions[task_id] += len(data[i]['observations'])

        if output_type == "max":
            return max(num_transitions.values())
        elif output_type == "dict":
            return num_transitions
    else:
        raise NotImplementedError


def measure_ram_used():
    """Return RAM usage in Mb"""
    process = psutil.Process(os.getpid())
    return int(process.memory_info().rss / 1000000)
