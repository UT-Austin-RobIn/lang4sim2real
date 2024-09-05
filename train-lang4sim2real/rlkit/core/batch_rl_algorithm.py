from collections import OrderedDict

import numpy as np
import torch


from rlkit.core.timer import timer
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.env_wrappers.embedding_wrappers import (
    TrainableEmbeddingWrapper,
    VideoTargetEmbeddingWrapper,
)
from rlkit.torch.networks.cnn import ClipWrapper
import rlkit.util.pytorch_util as ptu
from rlkit.util.core import np_to_pytorch_batch


class BatchRLAlgorithm(BaseRLAlgorithm):
    def __init__(
            self,
            batch_size,
            max_path_length,
            num_eval_steps_per_epoch,
            num_trains_per_train_loop,
            multi_task=False,
            meta_batch_size=4,
            train_tasks=0,  # or a list of task idxs.
            train_target_task_indices=[],
            eval_tasks=0,  # or a list of task idxs.
            train_task_sample_probs=None,
            biased_sampling=False,
            replay_buffer_positive=None,
            buffer_embeddings=None,
            multiple_strs_per_task=False,
            task_embedding=None,
            video_batch_size=None,
            task_emb_noise_std=0.0,
            *args,
            **kwargs
    ):

        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.multi_task = multi_task
        self.meta_batch_size = meta_batch_size
        self.train_tasks = train_tasks
        self.train_target_task_indices = train_target_task_indices
        self.eval_tasks = eval_tasks
        self.train_task_sample_probs = train_task_sample_probs
        self.biased_sampling = biased_sampling
        self.task_embedding = task_embedding
        if self.task_embedding in ['demo', 'demo_lang', 'mcil']:
            assert replay_buffer_positive is not None
            self.replay_buffer_positive = replay_buffer_positive
            self.buffer_embeddings = buffer_embeddings
            self.multiple_strs_per_task = multiple_strs_per_task
            self.task_emb_noise_std = task_emb_noise_std
            # self.eval_embeddings = eval_embeddings
            if video_batch_size is not None:
                assert self.batch_size % video_batch_size == 0
                self.video_batch_size = video_batch_size
            else:
                self.video_batch_size = batch_size

    def _train(self):
        done = (self.epoch == self.num_epochs)
        if done:
            return OrderedDict(), done

        timer.start_timer('evaluation sampling')
        if isinstance(self.eval_env, VideoTargetEmbeddingWrapper):
            # Calculate all the resnet embeddings once per epoch.
            with torch.no_grad():
                self.eval_env.update_video_and_target_embeddings()

                if self.task_embedding != "mcil":
                    # Compute Val loss
                    video_emb_matrix, target_emb_matrix, logit_scale = (
                        self.eval_env.get_video_target_embs_as_matrix())

                    target_emb_matrix = target_emb_matrix[:, 0, :]
                    visual_embs_list = [video_emb_matrix.cuda()]
                    logit_scale_list = [logit_scale]
                    val_task_enc_loss = (
                        self.trainer.task_encoder.val_loss_criterion.calc(
                            visual_embs_list, target_emb_matrix.cuda(),
                            logit_scale_list))

                    self.trainer.eval_statistics.update(
                        {'Val Task encoder loss': ptu.get_numpy(
                            val_task_enc_loss)}
                    )
        elif isinstance(self.eval_env, TrainableEmbeddingWrapper):
            self.eval_env.update_embeddings()

        if (self.epoch % self._eval_epoch_freq == 0
                and self.num_eval_steps_per_epoch > 0):
            if self.multi_task:
                for i in self.eval_tasks:
                    self.eval_data_collector.collect_new_paths(
                        self.max_path_length,
                        self.num_eval_steps_per_epoch,
                        discard_incomplete_paths=True,
                        multi_task=True,
                        task_index=i,
                    )
            else:
                self.eval_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_eval_steps_per_epoch,
                    discard_incomplete_paths=True,
                )

            # print("ending eval")
        timer.stop_timer('evaluation sampling')

        timer.start_timer('training', unique=False)
        for _ in range(self.num_trains_per_train_loop):
            if not self.multi_task:
                train_data = self.replay_buffer.random_batch(
                    self.batch_size,
                    biased_sampling=self.biased_sampling)
            else:
                task_indices = np.random.choice(
                    self.train_tasks, self.meta_batch_size,
                    replace=True,
                    p=self.train_task_sample_probs,
                )
                if len(self.train_target_task_indices) == 0:
                    assert len(set(task_indices).intersection(
                               set(self.eval_tasks))) == 0, (
                        "Cannot mix train and eval tasks")
                else:
                    trask_indices_from_eval_tasks = (
                        set(task_indices).intersection(
                            set(self.eval_tasks)))
                    assert trask_indices_from_eval_tasks.issubset(
                        self.train_target_task_indices), (
                        "Cannot train on eval tasks if they were not "
                        "from train-target-buffer")

                # remove img-lang stuff from batch if
                # img-lang-dist-learning not active
                self.replay_buffer.img_lang_dist_learning = bool(
                    self.trainer.alpha > 0.0)

                train_data = self.replay_buffer.sample_batch(
                    task_indices,
                    self.batch_size,
                )
                train_data['task_indices'] = task_indices
            if self.task_embedding in ['demo', 'demo_lang', 'mcil']:
                if self.trainer.task_encoder.use_cached_embs:
                    pass
                else:
                    # Get batch of positive videos
                    task_encoder_batch_dict = (
                        self.replay_buffer_positive.sample_bcz_video_batch_of_trajectories(
                            task_indices, self.video_batch_size,
                            with_replacement=True,
                            k=self.trainer.task_encoder.k,
                            frame_ranges=(
                                self.trainer.task_encoder.frame_ranges),
                        )
                    )
                    if self.video_batch_size < self.batch_size:
                        num_tiles = (
                            self.batch_size // self.video_batch_size)
                        task_encoder_batch_dict['video'] = np.tile(
                            task_encoder_batch_dict['video'],
                            (1, num_tiles, 1, 1, 1))
                    train_data['task_encoder_batch_dict'] = (
                        task_encoder_batch_dict)

                # Get batch of target embeddings
                if len(self.buffer_embeddings) > 0:
                    if not self.multiple_strs_per_task:
                        train_data['target_traj_embs'] = np.array([
                            self.buffer_embeddings[task_idx]
                            for task_idx in task_indices])
                    else:
                        # Choose randomly amongst multiple embs.
                        target_embs = []
                        for task_idx in task_indices:
                            buffer_embedding_choices = (
                                self.buffer_embeddings[task_idx])
                            num_choices = buffer_embedding_choices.shape[0]
                            assert 1 <= num_choices <= 7  # Hardcoded range
                            chosen_idx = np.random.randint(num_choices)
                            target_embs.append(
                                buffer_embedding_choices[chosen_idx])
                        train_data['target_traj_embs'] = np.array(
                            target_embs)
                else:
                    assert (isinstance(
                        self.trainer.task_encoder.encoder, ClipWrapper)
                        and not self.trainer.task_encoder.use_cached_embs)

                if self.task_emb_noise_std > 0.0:
                    # Add noise
                    noise = np.random.normal(
                        scale=self.task_emb_noise_std,
                        size=train_data['target_traj_embs'].shape)
                    train_data['target_traj_embs'] += noise

                    # Re-normalize
                    norms = np.linalg.norm(
                        train_data['target_traj_embs'], axis=-1)[:, None]
                    train_data['target_traj_embs'] = (
                        train_data['target_traj_embs'] / norms)
            self.trainer.train(train_data, epoch_num=self.epoch)
        timer.stop_timer('training')

        timer.start_timer('validation')
        if self.replay_buffer.train_val_split:
            task_indices = np.random.choice(
                self.train_tasks, self.meta_batch_size,
                replace=True,
                p=self.train_task_sample_probs,
            )
            val_data = self.replay_buffer.sample_batch(
                task_indices,
                self.batch_size,
                validation=True,
                img_lang_dist_learning=(
                    self.trainer.img_lang_dist_learning
                    and bool(self.trainer.alpha > 0.0))
            )
            self.trainer.val_stats(np_to_pytorch_batch(val_data))
        timer.stop_timer('validation')

        log_stats = self._get_diagnostics()
        return log_stats, False
