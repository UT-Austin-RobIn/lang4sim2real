from collections import Counter

import numpy as np
from PIL import Image, ImageDraw
import torch
from tqdm import tqdm

from rlkit.util.core import torch_ify
from rlkit.torch.networks.cnn import ClipWrapper
from rlkit.util.losses import (
    bcz_cosine_distance_loss_fn,
    clip_contrastive_loss_fn,
    metabatch_cross_ent_fn,
    multi_emb_loss_fn,
)

VIDEO_ENCODER_LOSS_TYPE_TO_FN_MAP = {
    "cosine_dist": bcz_cosine_distance_loss_fn,
    "contrastive": clip_contrastive_loss_fn,
    "cross_ent": metabatch_cross_ent_fn,
}


class BCZVideoEncoder:
    def __init__(
            self, encoder_net, frame_ranges,
            mosaic_rc=(1, 2), loss_type="cosine_dist",
            loss_kwargs={}, image_size=(48, 48, 3),
            use_cached_embs=False, buffer_ext_dict={}):
        self.encoder = encoder_net
        self.m = mosaic_rc[0]
        self.n = mosaic_rc[1]
        self.k = np.prod(mosaic_rc)  # num frames to downsample the video to.
        self.image_size = image_size
        self.num_film_inputs = encoder_net.num_film_inputs
        self.loss_kwargs = loss_kwargs
        self.frame_ranges = frame_ranges
        self.set_loss_criterion(loss_type, loss_kwargs)
        self.use_cached_embs = use_cached_embs
        self.buffer_ext_dict = buffer_ext_dict
        self.i_map = Counter()

    def __call__(self, batch, train_mode=True):
        if isinstance(self.encoder, ClipWrapper):
            video_embs, lang_embs, logit_scale = self.encoder(
                batch['video'], batch['clip_lang'], train_mode=train_mode)
        elif self.num_film_inputs == 0:
            video_embs = self.encoder(batch['video'], train_mode=train_mode)
        elif self.num_film_inputs == 1:
            video_embs = self.encoder(
                batch['video'], train_mode=train_mode,
                film_inputs=[batch['lang']])
        else:
            raise NotImplementedError

        out_dict = {"video": video_embs}

        if isinstance(self.encoder, ClipWrapper):
            out_dict["lang"] = lang_embs
            out_dict["logit_scale"] = logit_scale

        return out_dict

    def get_cached_embs_from_task_ids(self, task_ids, num_embs_per_task):
        """
        Only use for frozen models.
        Output should have the same format as __call__()"""
        embs = []
        for task_id in task_ids:
            task_id = int(task_id)
            possible_embs_for_task_id = self.cached_embs[task_id]
            chosen_emb_indices = np.random.choice(
                range(possible_embs_for_task_id.shape[0]), num_embs_per_task)
            task_embs = possible_embs_for_task_id[chosen_emb_indices]
            embs.append(task_embs)
        embs_array = torch.cat(embs, axis=0)
        return {"video": embs_array, "lang": None}

    def create_cached_embs(
            self, train_buffer, target_buffer,
            train_task_indices, target_task_indices, num_embs_per_task,
            max_path_length):
        assert isinstance(self.encoder, ClipWrapper)
        orig_num_embs_per_task = num_embs_per_task
        self.cached_embs = {}
        all_task_indices = sorted(train_task_indices + target_task_indices)
        for task_idx in tqdm(all_task_indices):
            if task_idx in train_task_indices:
                buf = train_buffer
                ext = self.buffer_ext_dict['train']
            elif task_idx in target_task_indices:
                buf = target_buffer
                ext = self.buffer_ext_dict['target']

            if ext == "hdf5":
                num_paths_available = buf.num_steps_can_sample(task_idx)
            elif ext == "npy":
                num_paths_available = (
                    buf.task_buffers[task_idx]._top // max_path_length)
            else:
                raise NotImplementedError
            num_embs_per_task = min(
                orig_num_embs_per_task, num_paths_available)

            task_indices = [task_idx]
            # Get batch of positive videos
            task_encoder_batch_dict = (
                buf.sample_bcz_video_batch_of_trajectories(
                    task_indices, num_embs_per_task,
                    with_replacement=False, k=self.k,
                    frame_ranges=self.frame_ranges,
                ))

            # Process task_encoder_batch_dict
            for key, val in task_encoder_batch_dict.items():
                torch_val = torch_ify(val)
                # task_encoder_batch_dict[key].shape:
                # (meta_bs, bs, 3, im_size * num_rows, im_size * num_cols)
                task_encoder_batch_dict[key] = torch.cat(
                    [torch_val[i] for i in range(torch_val.shape[0])], dim=0)
                # task_encoder_batch_dict[key].shape:
                # (meta_bs * bs, 3, im_size * num_rows, im_size * num_cols)

            task_encoder_batch_dict["clip_lang"] = (
                self.encoder.get_task_lang_tokens_matrix(task_indices))
            traj_emb_preds_dict = self.__call__(
                task_encoder_batch_dict, train_mode=False)
            task_idx_embs = traj_emb_preds_dict['video']
            self.cached_embs[task_idx] = task_idx_embs

        print("Num cached embs by task idx:",
              dict([(i, x.shape[0]) for i, x in self.cached_embs.items()]))

    def set_loss_criterion(self, loss_type, loss_kwargs):
        if loss_type is None:
            return

        self.loss_criterion = VIDEO_ENCODER_LOSS_TYPE_TO_FN_MAP[loss_type]
        if len(loss_kwargs) > 0:
            self.loss_criterion = self.loss_criterion(**loss_kwargs)
        else:
            # instantiate the class for cosine_dist
            self.loss_criterion = self.loss_criterion()

        if loss_type == "cosine_dist":
            self.val_loss_criterion = self.loss_criterion
        elif loss_type in ["contrastive", "cross_ent"]:
            self.val_loss_criterion = (
                VIDEO_ENCODER_LOSS_TYPE_TO_FN_MAP["contrastive"])
            self.val_loss_kwargs = dict(loss_kwargs)
            if "meta_batch_size" in self.val_loss_kwargs:
                self.val_loss_kwargs.pop("meta_batch_size")
            self.val_loss_kwargs['temp'] = 1.0
            self.val_loss_criterion = self.val_loss_criterion(
                **self.val_loss_kwargs)
        else:
            raise NotImplementedError

        self.loss_criterion = multi_emb_loss_fn(self.loss_criterion)
        self.val_loss_criterion = multi_emb_loss_fn(self.val_loss_criterion)

    def save_image(self, im_arr, fname):
        im = Image.fromarray(im_arr)
        im.save(fname)

    def create_text_im_arr(self, text):
        im = Image.new('RGB', (48, 10), color=(0, 0, 0))
        d = ImageDraw.Draw(im)
        d.text((0, 0), text, fill=(255, 255, 255))
        return np.array(im)

    def process_batch_by_key(self, batch, obs_key, unflatten_im=True):
        """traj is a dict : key --> (bsz, value_dim)"""
        """Assumes batch[obs_key] is AFTER randomly selecting the frames"""

        # (B, self.k, 6912) --> (B, self.k, 48, 48, 3)
        if unflatten_im:
            im_size_flat = np.prod(self.image_size)
            image_batch = batch[obs_key][:, :, :im_size_flat]
            # np.save("20220330_imb.npy", image_batch)

            B, H, _ = image_batch.shape
            h, w, c = self.image_size
            image_batch = np.reshape(
                image_batch, (B, H, c, h, w)).transpose(0, 1, 3, 4, 2)
        else:
            image_batch = batch[obs_key]
            B, H, h, w, c = image_batch.shape

        # The randomization stuff is moved into the obs_dict_replay_buffer
        # random_trajectory(...) function

        # (B, self.k, 48, 48, 3) --> (B, m * 48, n * 48, 3)
        image_batch_mosaic = np.zeros_like(image_batch).reshape(
            B, self.m * h, self.n * w, c)
        for r in range(self.m):
            for c in range(self.n):
                image_batch_mosaic[
                    :, h * r : h * (r + 1):, w * c : w * (c + 1), :] = (
                        image_batch[:, self.m * r + c, :, :])

        image_batch = image_batch_mosaic

        # (B, m * 48, n * 48, 3) --> (B, 3, m * 48, n * 48)
        # [to pass into resnet]
        image_batch = np.transpose(image_batch, (0, 3, 1, 2))
        return image_batch

    def process_visual_batch(self, batch, unflatten_im=True):
        video = self.process_batch_by_key(batch, "observations", unflatten_im)
        visual_batch_dict = {"video": video}
        return visual_batch_dict
