import gym
import numpy as np
import torch

from rlkit.core.roboverse_serializable import Serializable
from rlkit.torch.networks.cnn import ClipWrapper
from rlkit.util.core import torch_ify
from rlkit.util.misc_functions import tile_embs_by_batch_size
import rlkit.util.pytorch_util as ptu


class EmbeddingWrapperOffline(gym.Env, Serializable):

    def __init__(self, env, embeddings, emb_obs_key):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.embeddings = embeddings
        assert isinstance(emb_obs_key, str)
        self.emb_obs_key = emb_obs_key
        self.latent_dim = len(self.embeddings[list(self.embeddings.keys())[0]])
        self.num_tasks = env.num_tasks

    def is_reset_task(self):
        return self.env.is_reset_task()

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        obs.update({self.emb_obs_key: self.embeddings[self.env.task_idx]})
        return obs, rew, done, info

    def reset(self):
        obs = self.env.reset()
        obs.update({self.emb_obs_key: self.embeddings[self.env.task_idx]})
        return obs

    def reset_task(self, task_idx):
        self.env.reset_task(task_idx)


class EmbeddingWrapper(gym.Env, Serializable):
    def __init__(self, env, embeddings, emb_obs_key, eval_task_indices):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.embeddings = embeddings
        self.latent_dim = len(self.embeddings[0])
        self.num_tasks = env.num_tasks
        assert isinstance(emb_obs_key, str)
        self.emb_obs_key = emb_obs_key
        self.eval_task_indices = eval_task_indices

    def is_reset_task(self):
        return self.env.is_reset_task()

    def get_task_emb_dict(self, task_idx):
        return {self.emb_obs_key: self.embeddings[task_idx]}

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        obs.update(self.get_task_emb_dict(self.env.task_idx))
        return obs, rew, done, info

    def reset(self):
        obs = self.env.reset()
        obs.update(self.get_task_emb_dict(self.env.task_idx))
        return obs

    def reset_task(self, task_idx):
        self.env.reset_task(task_idx)

    def get_observation(self):
        return self.env.get_observation()

    def reset_robot_only(self):
        return self.env.reset_robot_only()

    def get_new_task_idx(self):
        '''
        Propose a new task based on whether the last trajectory succeeded.
        '''
        info = self.env.get_info()
        if info['reset_success_target']:
            new_task_idx = self.env.task_idx - self.env.num_tasks
        elif info['place_success_target']:
            new_task_idx = self.env.task_idx + self.env.num_tasks
        else:
            new_task_idx = self.env.task_idx
        return new_task_idx

    def create_emb_dict_by_task_idx(self, embeddings_dict, modality):
        assert modality in ["video", "lang"]
        if modality in embeddings_dict:
            # Average video embeddings over each task
            embeddings_by_task = torch.split(
                embeddings_dict[modality].detach(), self.eval_video_batch_size)
            avg_embeddings_by_task = [
                torch.mean(embs, dim=0).cpu().numpy()
                for embs in embeddings_by_task]
            return dict(
                [(task_idx, avg_embeddings_by_task[i])
                 for i, task_idx in enumerate(self.eval_task_indices)])
        else:
            return None


class TrainableEmbeddingWrapper(EmbeddingWrapper):
    """
    Allows the (language) embedding model to be trainable;
    runs it each time.
    Does not support multimodal embeddings.
    """
    def __init__(
            self, env, lang_enc, emb_obs_key, eval_video_batch_size,
            eval_task_indices, eval_task_idx_to_tokens_map=None):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.lang_enc = lang_enc
        self.num_tasks = env.num_tasks
        assert isinstance(emb_obs_key, str)
        self.emb_obs_key = emb_obs_key
        self.eval_video_batch_size = eval_video_batch_size
        self.eval_task_idx_to_tokens_map = eval_task_idx_to_tokens_map
        self.eval_task_indices = eval_task_indices
        if self.eval_task_idx_to_tokens_map is not None:
            self.task_lang_tokens_matrix = np.concatenate(
                [self.eval_task_idx_to_tokens_map[task_idx][None]
                    for task_idx in self.eval_task_indices],
                axis=0)
            self.task_lang_tokens_matrix = ptu.from_numpy(
                self.task_lang_tokens_matrix).long()
        else:
            self.task_lang_tokens_matrix = None

    def update_embeddings(self):
        if isinstance(self.lang_enc, ClipWrapper):
            # This case is not actually being used.
            assert self.task_lang_tokens_matrix is None
            task_lang_tokens = self.lang_enc.get_task_lang_tokens_matrix(
                self.eval_task_indices)
            task_lang_tokens_tiled = tile_embs_by_batch_size(
                task_lang_tokens, self.eval_video_batch_size)
            lang_enc_batch_dict = {"clip_lang": task_lang_tokens_tiled}
            image_features, text_features = self.lang_enc(
                lang_enc_batch_dict, train_mode=False)
            embeddings_dict = dict(
                video=image_features,
                lang=text_features,
            )
            self.embeddings = self.create_emb_dict_by_task_idx(
                embeddings_dict, "lang")
        else:
            assert self.task_lang_tokens_matrix is not None
            with torch.no_grad():
                lang_embs = self.lang_enc(self.task_lang_tokens_matrix)
            lang_embs_tiled = tile_embs_by_batch_size(
                lang_embs, self.eval_video_batch_size)
            embeddings_dict = {"lang": lang_embs_tiled}
            self.embeddings = self.create_emb_dict_by_task_idx(
                embeddings_dict, "lang")


class VideoTargetEmbeddingWrapper(EmbeddingWrapper):
    def __init__(
            self, env, task_encoder, target_embeddings,
            eval_task_indices, eval_video_batch_size,
            task_embedding, path_len, target_emb_net=None,
            emb_obs_keys=[], policy_num_film_inputs=0,
            task_emb_input_mode="concat_to_img_embs",
            lang_enc=None):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

        self.replay_buffer = None
        # Success-only target buffer containing only eval_task_indices tasks
        self.task_encoder = task_encoder
        self.target_embeddings = target_embeddings  # Dict();
        # before transformation by target_emb_net
        self.target_embeddings_transformed = dict()
        self.target_emb_net = target_emb_net
        self.eval_task_indices = eval_task_indices
        self.eval_video_batch_size = eval_video_batch_size
        self.video_embeddings = None # Dict()
        self.task_embedding = task_embedding
        self.path_len = path_len
        self.emb_obs_keys = emb_obs_keys
        self.policy_num_film_inputs = policy_num_film_inputs
        self.task_emb_input_mode = task_emb_input_mode
        self.lang_enc = lang_enc
        if self.lang_enc is not None:
            # In cases where lang_enc != None, self.target_embeddings
            # is interpreted as tokens instead of embs.
            # But after the first update_target_embeddings(), 
            # self.target_embeddings becomes the actual language embs of the
            # lang_enc
            self.task_lang_tokens_matrix = self.get_target_embs_as_matrix()
        self.logit_scale = None

        if (isinstance(self.task_encoder.encoder, ClipWrapper)
                and not self.task_encoder.use_cached_embs):
            assert target_embeddings == dict()
            # They will be calculated by making a pass through CLIP
        assert sorted(self.eval_task_indices) == self.eval_task_indices

        self.modality_to_emb_obs_key_map = (
            self.create_modality_to_emb_obs_key_map())

    def create_modality_to_emb_obs_key_map(self):
        if self.task_embedding == "mcil":
            modality_to_emb_obs_key_map = {}
            for modality in ["video", "lang"]:
                modality_to_emb_obs_key_map[modality] = f"{modality}_embedding"
            assert (set(modality_to_emb_obs_key_map.values())
                    == set(self.emb_obs_keys))
        elif ((self.policy_num_film_inputs in [0, 1]) and
              (self.task_emb_input_mode in ["concat_to_img_embs", "film"])):
            assert len(self.emb_obs_keys) == 1
            modality_to_emb_obs_key_map = {"task": self.emb_obs_keys[0]}
        elif (self.policy_num_film_inputs in [2, 3]
              or self.task_emb_input_mode in [
                "film_video_concat_lang", "film_lang_concat_video"]):
            if self.task_embedding == "demo_lang":
                modalities = ["video", "lang"]
            else:
                modalities = ["video"]

            modality_to_emb_obs_key_map = {}
            for modality in modalities:
                modality_to_emb_obs_key_map[modality] = f"{modality}_embedding"
            assert (set(modality_to_emb_obs_key_map.values())
                    == set(self.emb_obs_keys))
        else:
            raise NotImplementedError
        return modality_to_emb_obs_key_map

    def update_video_and_target_embeddings(self):
        if self.task_encoder.use_cached_embs:
            embs_dict = self.task_encoder.get_cached_embs_from_task_ids(
                self.eval_task_indices, self.eval_video_batch_size)
            self.video_embeddings = self.create_emb_dict_by_task_idx(
                embs_dict, "video")
            self.update_target_embeddings_transformed()
        else:
            self.update_eval_video_task_embeddings()
            self.update_target_embeddings()
            self.update_target_embeddings_transformed()

    def update_eval_video_task_embeddings(self):
        # This should only be called once at the end of each epoch
        assert self.replay_buffer is not None
        # Run through the ResNet video encoder
        task_encoder_batch_dict = (
            self.replay_buffer.sample_bcz_video_batch_of_trajectories(
                self.eval_task_indices, self.eval_video_batch_size,
                with_replacement=False, k=self.task_encoder.k,
                frame_ranges=self.task_encoder.frame_ranges,)
        )
        for key, val in task_encoder_batch_dict.items():
            torch_val = torch_ify(val)
            task_encoder_batch_dict[key] = torch.cat(
                [torch_val[i] for i in range(torch_val.shape[0])], dim=0)

        if self.task_encoder.num_film_inputs > 0:
            # When putting language into the video encoder
            assert not isinstance(self.task_encoder.encoder, ClipWrapper)
            traj_emb_targets = np.array([
                self.get_target_emb(task_idx)
                for task_idx in self.eval_task_indices])
            traj_emb_targets = tile_embs_by_batch_size(
                traj_emb_targets, self.eval_video_batch_size)
            assert all([
                traj_emb_targets.shape[0] == val.shape[0]
                for key, val in task_encoder_batch_dict.items()])
            task_encoder_batch_dict["lang"] = traj_emb_targets

        if isinstance(self.task_encoder.encoder, ClipWrapper):
            task_lang_tokens = (
                self.task_encoder.encoder.get_task_lang_tokens_matrix(
                    self.eval_task_indices)
            )
            task_lang_tokens_tiled = tile_embs_by_batch_size(
                task_lang_tokens, self.eval_video_batch_size)
            task_encoder_batch_dict["clip_lang"] = task_lang_tokens_tiled

        embeddings_dict = self.task_encoder(
            task_encoder_batch_dict, train_mode=False)
        self.video_embeddings = self.create_emb_dict_by_task_idx(
            embeddings_dict, "video")
        if isinstance(self.task_encoder.encoder, ClipWrapper):
            # Update language embeddings by updating self.target_embeddings.
            self.target_embeddings = self.create_emb_dict_by_task_idx(
                embeddings_dict, "lang")
            self.logit_scale = embeddings_dict["logit_scale"]

    def update_target_embeddings(self):
        if self.lang_enc is None:
            return

        with torch.no_grad():
            lang_embs = self.lang_enc(self.task_lang_tokens_matrix)
        lang_embs_tiled = tile_embs_by_batch_size(
            lang_embs, self.eval_video_batch_size)
        embeddings_dict = {"lang": lang_embs_tiled}
        self.target_embeddings = self.create_emb_dict_by_task_idx(
            embeddings_dict, "lang")

    def update_target_embeddings_transformed(self):
        if self.target_emb_net is None:
            return

        target_emb_matrix = self.get_target_embs_as_matrix()
        target_emb_transformed_matrix = self.target_emb_net(
            ptu.tensor(target_emb_matrix))

        sorted_task_idxs = sorted(list(self.target_embeddings.keys()))
        for i, task_idx in enumerate(sorted_task_idxs):
            self.target_embeddings_transformed[task_idx] = np.array(
                target_emb_transformed_matrix[i].cpu())

    def get_target_emb(self, task_idx):
        if self.target_emb_net is not None:
            target_emb = self.target_embeddings_transformed[task_idx]
        else:
            target_emb = self.target_embeddings[task_idx]
        return target_emb

    def get_task_emb_dict(self, task_idx):
        video_emb = self.video_embeddings[task_idx]

        if self.task_embedding == "demo_lang":
            target_emb = self.get_target_emb(task_idx)

            if (self.policy_num_film_inputs in [0, 1] and
                    self.task_emb_input_mode in [
                        "concat_to_img_embs", "film"]):
                if self.task_encoder.num_film_inputs == 0:
                    emb = np.concatenate([video_emb, target_emb], axis=0)
                elif self.task_encoder.num_film_inputs == 1:
                    emb = video_emb
                else:
                    raise NotImplementedError
                emb_dict = {self.modality_to_emb_obs_key_map['task']: emb}
            elif (self.policy_num_film_inputs == 2
                  or self.task_emb_input_mode in [
                    "film_video_concat_lang", "film_lang_concat_video"]):
                emb_dict = {
                    self.modality_to_emb_obs_key_map['video']: video_emb,
                    self.modality_to_emb_obs_key_map['lang']: target_emb,
                }
            elif self.policy_num_film_inputs == 3:
                emb_dict = {
                    self.modality_to_emb_obs_key_map['video']: video_emb,
                    self.modality_to_emb_obs_key_map['lang']: target_emb,
                }
            else:
                raise NotImplementedError
        elif self.task_embedding == "mcil":
            target_emb = self.get_target_emb(task_idx)
            emb_dict = {
                self.modality_to_emb_obs_key_map['video']: video_emb,
                self.modality_to_emb_obs_key_map['lang']: target_emb,
            }
        else:
            assert self.task_embedding in ["demo"]
            emb = video_emb
            emb_dict = {self.modality_to_emb_obs_key_map['task']: emb}

        if self.task_emb_input_mode != "concat_to_img_embs":
            # Film stuff needs emb to be converted.
            # convert each emb from (size,) --> (1, size)
            for key in emb_dict:
                if len(emb_dict[key].shape) == 1:
                    emb_dict[key] = np.expand_dims(emb_dict[key], axis=0)

        return emb_dict

    def get_target_embs_as_matrix(self):
        target_emb_list = []
        sorted_task_idxs = sorted(list(self.target_embeddings.keys()))
        for task_idx in sorted_task_idxs:
            target_emb_list.append(self.target_embeddings[task_idx])
        target_emb_matrix = ptu.tensor(target_emb_list)
        return target_emb_matrix

    def get_video_target_embs_as_matrix(self):
        """
        For validation loss calculations
        """
        video_emb_list = []
        target_emb_list = []
        sorted_task_idxs = sorted(list(self.target_embeddings.keys()))
        for task_idx in sorted_task_idxs:
            video_emb_list.append(self.video_embeddings[task_idx])
            target_emb = self.get_target_emb(task_idx)
            target_emb_list.append(target_emb)

        video_emb_matrix = torch.tensor(video_emb_list)
        target_emb_matrix = torch.tensor(target_emb_list)

        return video_emb_matrix, target_emb_matrix, self.logit_scale

    def set_replay_buffer(self, replay_buffer):
        self.replay_buffer = replay_buffer

    def get_new_task_idx(self):
        raise NotImplementedError
