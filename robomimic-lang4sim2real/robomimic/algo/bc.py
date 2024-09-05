"""
Implementation of Behavioral Cloning (BC).
"""
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import robomimic.models.base_nets as BaseNets
import robomimic.models.obs_nets as ObsNets
import robomimic.models.policy_nets as PolicyNets
import robomimic.models.task_encoder_nets as TaskEncoderNets
import robomimic.models.vae_nets as VAENets
import robomimic.utils.loss_utils as LossUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo
from robomimic.models.language_models import LM_STR_TO_FN_CLASS_MAP

@register_algo_factory_func("bc")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """

    # note: we need the check below because some configs import BCConfig and exclude
    # some of these options
    gaussian_enabled = ("gaussian" in algo_config and algo_config.gaussian.enabled)
    gmm_enabled = ("gmm" in algo_config and algo_config.gmm.enabled)
    vae_enabled = ("vae" in algo_config and algo_config.vae.enabled)

    if algo_config.verb_obj_enc.enabled:
        algo_kwargs = {
            "sample_size": algo_config.verb_obj_enc.sample_size,
            "task_emb_mode": algo_config.verb_obj_enc.task_emb_mode,
            "task_enc_contr_temp": algo_config.verb_obj_enc.task_enc_contr_temp,
            "task_enc_weight": algo_config.verb_obj_enc.task_enc_weight,
            "task_enc_loss": algo_config.verb_obj_enc.loss,
        }
        return BC_Verb_Obj, algo_kwargs

    if algo_config.rnn.enabled:
        if gmm_enabled:
            return BC_RNN_GMM, {}
        return BC_RNN, {}
    assert sum([gaussian_enabled, gmm_enabled, vae_enabled]) <= 1
    if gaussian_enabled:
        return BC_Gaussian, {}
    if gmm_enabled:
        return BC_GMM, {}
    if vae_enabled:
        return BC_VAE, {}
    algo_kwargs = {
        "num_tasks": 12, # TODO: remove this hardcoding
        "gt_task_emb": algo_config.gt_task_emb,
    }
    return BC, algo_kwargs


class BC(PolicyAlgo):
    """
    Normal BC training.
    """
    def __init__(self, num_tasks=0, gt_task_emb="no_task_emb", *args, **kwargs):
        self.num_tasks = num_tasks
        self.gt_task_emb = gt_task_emb
        super().__init__(*args, **kwargs)

    def _create_shapes(self, obs_keys, obs_key_shapes):
        super()._create_shapes(obs_keys, obs_key_shapes)
        if self.gt_task_emb == "onehot_taskid":
            assert "task_id" in self.obs_shapes
            self.obs_shapes["task_id"] = self.num_tasks
            # override; by default task_id is already decomposed into verb and obj onehots.

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.ActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )
        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training 
        """
        input_batch = dict()
        if self.gt_task_emb == "onehot_taskid":
            # Replace original task_ids with vector length num_tasks
            assert "task_id" in batch["obs"].keys()
            batch["obs"]["task_id"] = batch.get("onehot_task_ids", None)
        input_batch["obs"] = {k: batch["obs"][k][:, 0, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"][:, 0, :]
        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(BC, self).train_on_batch(batch, epoch, validate=validate)
            predictions = self._forward_training(batch)
            losses = self._compute_losses(predictions, batch)

            info["predictions"] = TensorUtils.detach(predictions)
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                step_info = self._train_step(losses)
                info.update(step_info)

        return info

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        predictions = OrderedDict()
        actions = self.nets["policy"](obs_dict=batch["obs"], goal_dict=batch["goal_obs"])
        predictions["actions"] = actions
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """
        losses = OrderedDict()
        a_target = batch["actions"]
        actions = predictions["actions"]
        losses["l2_loss"] = nn.MSELoss()(actions, a_target)
        losses["l1_loss"] = nn.SmoothL1Loss()(actions, a_target)
        # cosine direction loss on eef delta position
        losses["cos_loss"] = LossUtils.cosine_loss(actions[..., :3], a_target[..., :3])

        action_losses = [
            self.algo_config.loss.l2_weight * losses["l2_loss"],
            self.algo_config.loss.l1_weight * losses["l1_loss"],
            self.algo_config.loss.cos_weight * losses["cos_loss"],
        ]
        action_loss = sum(action_losses)
        losses["action_loss"] = action_loss
        return losses

    def _train_step(self, losses):
        """
        Internal helper function for BC algo class. Perform backpropagation on the
        loss tensors in @losses to update networks.

        Args:
            losses (dict): dictionary of losses computed over the batch, from @_compute_losses
        """

        # gradient step
        info = OrderedDict()
        policy_grad_norms = TorchUtils.backprop_for_loss(
            net=self.nets["policy"],
            optim=self.optimizers["policy"],
            loss=losses["action_loss"],
        )
        info["policy_grad_norms"] = policy_grad_norms
        return info

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(BC, self).log_info(info)
        log["Loss"] = info["losses"]["action_loss"].item()
        if "l2_loss" in info["losses"]:
            log["L2_Loss"] = info["losses"]["l2_loss"].item()
        if "l1_loss" in info["losses"]:
            log["L1_Loss"] = info["losses"]["l1_loss"].item()
        if "cos_loss" in info["losses"]:
            log["Cosine_Loss"] = info["losses"]["cos_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training
        return self.nets["policy"](obs_dict, goal_dict=goal_dict)


class BC_Verb_Obj(BC):
    def __init__(
            self, sample_size, task_emb_mode,
            verb_id_to_verb_str_map=None,
            obj_id_to_obj_str_map=None,
            task_id_to_instruction_map=None,
            task_enc_contr_temp=1.0,
            task_enc_weight=1.0,
            task_enc_loss="contrastive",
            *args, **kwargs):
        self.sample_size = sample_size
        self.task_emb_mode = task_emb_mode
        assert self.task_emb_mode in ["onehot", "lang"]
        self.num_objs = 5 #TODO: Make this not hardcoded

        if self.task_emb_mode == "onehot":
            self.verb_enc_output_dim = 4 #TODO: Make this not hardcoded
            self.obj_enc_output_dim = self.num_objs

            self.fc_layer_dims = [256, 128, 64]
            self.task_enc_final_activ = "logsoftmax"
        elif self.task_emb_mode == "lang":
            lang_model_class = LM_STR_TO_FN_CLASS_MAP["minilm"]
            self.lang_model = lang_model_class()
            self.verb_enc_output_dim = self.lang_model.out_dim
            self.obj_enc_output_dim = self.lang_model.out_dim

            self.verb_id_to_verb_str_map = verb_id_to_verb_str_map
            self.obj_id_to_obj_str_map = obj_id_to_obj_str_map
            self.task_id_to_instruction_map = task_id_to_instruction_map

            self.fc_layer_dims = [512, 512, 512]
            self.task_enc_final_activ = "l2norm"

            self.task_enc_contr_temp = task_enc_contr_temp

            assert task_enc_loss in ["contrastive", "cosdist"]
            self.task_enc_loss = task_enc_loss

            self.create_id_to_lang_emb_maps()

        self.task_enc_weight = task_enc_weight
        return super().__init__(*args, **kwargs)

    def _create_shapes(self, obs_keys, obs_key_shapes):
        super()._create_shapes(obs_keys, obs_key_shapes)
        # Needed to actually pass verb and obj_embs into the policy
        self.obs_shapes.pop("task_id") # We don't want to input task_id ground truth into policy
        self.obs_shapes["verb_embs"] = self.verb_enc_output_dim
        self.obs_shapes["obj_embs"] = self.obj_enc_output_dim

        if self.task_emb_mode == "lang":
            # Also add language embeddings to obs
            self.obs_shapes["lang_embs"] = self.lang_model.out_dim

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        super()._create_networks()
        self.nets["verb_enc"] = TaskEncoderNets.TrajMLPClassifier(
            out_dim=self.verb_enc_output_dim,
            fc_layer_dims=self.fc_layer_dims,
            token_dim=5,
            sample_size=self.sample_size,
            final_activ=self.task_enc_final_activ)
        self.nets["obj_enc"] = TaskEncoderNets.TrajMLPClassifier(
            out_dim=self.obj_enc_output_dim,
            fc_layer_dims=self.fc_layer_dims,
            token_dim=self.num_objs * 3,
            sample_size=self.sample_size,
            final_activ=self.task_enc_final_activ)
        self.nets = self.nets.float().to(self.device)

    def _create_optimizers(self):
        """
        Create joint optimizer for all nets.
        """
        self.optimizers = dict()
        self.lr_schedulers = dict()
        self.optimizers["joint"] = TorchUtils.optimizer_from_optim_params(
            net_optim_params=self.optim_params['policy'], net=list(self.nets.values()))
        self.lr_schedulers["joint"] = TorchUtils.lr_scheduler_from_optim_params(
            net_optim_params=self.optim_params["policy"], net=list(self.nets.values()),
            optimizer=self.optimizers["joint"])

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, 0, :] for k in batch["obs"]} # TODO: double check this concats everything we want.
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"][:, 0, :]

        input_batch["verb_enc_obs"] = batch.get("verb_enc_obs", None)
        input_batch["verb_ids"] = batch.get("verb_ids", None)
        input_batch["obj_enc_obs"] = batch.get("obj_enc_obs", None)
        input_batch["obj_ids"] = batch.get("obj_ids", None)

        if self.task_emb_mode == "lang":
            input_batch["task_ids"] = batch.get("task_ids", None)

        if input_batch["verb_enc_obs"] is not None:
            input_batch["verb_enc_obs"] = torch.cat(list(input_batch["verb_enc_obs"].values()), dim=-1)
        if input_batch["obj_enc_obs"] is not None:
            input_batch["obj_enc_obs"] = torch.cat(list(input_batch["obj_enc_obs"].values()), dim=-1)

        # print("input_batch['verb_enc_obs']", input_batch["verb_enc_obs"].shape)
        # print("input_batch['obj_enc_obs']", input_batch["obj_enc_obs"].shape)
        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def create_id_to_lang_emb_maps(self):
        def create_id_to_lang_emb_matrix(id_to_str_map):
            # Order each map in ascending order of id
            id_str_pair_list = list(id_to_str_map.items())
            id_str_pair_list = sorted(id_str_pair_list, key=lambda x: x[0])

            # Create list of strings
            str_list = [x[1] for x in id_str_pair_list]

            # Get embs
            emb_matrix = self.lang_model(str_list)

            # Create id_to_lang_emb_map
            id_lang_emb_matrix = torch.zeros((len(str_list), self.lang_model.out_dim)) # rows are id
            for i, (id_, _) in enumerate(id_str_pair_list):
                id_lang_emb_matrix[id_] = emb_matrix[i]

            return id_lang_emb_matrix.cuda()

        self.verb_id_to_lang_emb_matrix = create_id_to_lang_emb_matrix(self.verb_id_to_verb_str_map)
        self.obj_id_to_lang_emb_matrix = create_id_to_lang_emb_matrix(self.obj_id_to_obj_str_map)
        self.task_id_to_lang_emb_matrix = create_id_to_lang_emb_matrix(self.task_id_to_instruction_map)

    def _enc_forward(self, batch):
        embs_dict = dict()
        embs_dict['verb_embs'] = self.nets["verb_enc"](batch["verb_enc_obs"])
        embs_dict['obj_embs'] = self.nets["obj_enc"](batch["obj_enc_obs"])
        if self.task_emb_mode == "lang":
            embs_dict['lang_embs'] = self.task_id_to_lang_emb_matrix[batch["task_ids"].long()].detach()
        return embs_dict

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        predictions = OrderedDict()
        embs_dict = self._enc_forward(batch)
        predictions.update(embs_dict)

        if self.task_emb_mode == "onehot":
            # Softmax the embs_dict before passing to policy.
            embs_dict['verb_embs'] = torch.exp(embs_dict['verb_embs'])
            embs_dict['obj_embs'] = torch.exp(embs_dict['obj_embs'])
        batch['obs'].update(embs_dict)

        actions = self.nets["policy"](obs_dict=batch["obs"], goal_dict=batch["goal_obs"])
        predictions["actions"] = actions
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """
        losses = super()._compute_losses(predictions, batch)

        if self.task_emb_mode == "onehot":
            losses['verb_enc_loss'] = nn.CrossEntropyLoss()(
                predictions['verb_embs'], batch['verb_ids'].long())
            losses['obj_enc_loss'] = nn.CrossEntropyLoss()(
                predictions['obj_embs'], batch['obj_ids'].long())
        elif self.task_emb_mode == "lang":
            if self.task_enc_loss == "contrastive":
                # (bsz, lang_dim) (lang_dim, num_classes)
                # Use a semi-contrastive loss (fixed number of targets (language embs))
                verb_emb_dot_lang_verb_embs = predictions['verb_embs'] @ self.verb_id_to_lang_emb_matrix.T.detach()
                verb_logits = verb_emb_dot_lang_verb_embs / self.task_enc_contr_temp
                losses['verb_enc_loss'] = nn.CrossEntropyLoss()(
                    verb_logits, batch['verb_ids'].long())

                obj_emb_dot_lang_obj_embs = predictions['obj_embs'] @ self.obj_id_to_lang_emb_matrix.T.detach()
                obj_logits = obj_emb_dot_lang_obj_embs / self.task_enc_contr_temp
                losses['obj_enc_loss'] = nn.CrossEntropyLoss()(
                    obj_logits, batch['obj_ids'].long())
            elif self.task_enc_loss == "cosdist":
                # Use cosine distance loss
                verb_emb_targets = self.verb_id_to_lang_emb_matrix[batch['verb_ids'].long()] # (bsz, 384)
                verb_preds_dot_targets = predictions['verb_embs'] @ verb_emb_targets.T # (bsz, bsz)
                losses['verb_enc_loss'] = 1 - torch.diagonal(verb_preds_dot_targets)
                losses['verb_enc_loss'] = torch.mean(losses['verb_enc_loss'])

                obj_emb_targets = self.obj_id_to_lang_emb_matrix[batch['obj_ids'].long()] # (bsz, 384)
                obj_preds_dot_targets = predictions['obj_embs'] @ obj_emb_targets.T # (bsz, bsz)
                losses['obj_enc_loss'] = 1 - torch.diagonal(obj_preds_dot_targets)
                losses['obj_enc_loss'] = torch.mean(losses['obj_enc_loss'])
            else:
                raise NotImplementedError

        losses['joint_loss'] = (
            (1.0 * losses['action_loss']) +
            (self.task_enc_weight * losses['verb_enc_loss']) +
            (self.task_enc_weight * losses['obj_enc_loss'])
        )
        return losses

    def _train_step(self, losses):
        """
        Internal helper function for BC algo class. Perform backpropagation on the
        loss tensors in @losses to update networks.

        Args:
            losses (dict): dictionary of losses computed over the batch, from @_compute_losses
        """

        # gradient step
        info = OrderedDict()
        grad_norms = TorchUtils.backprop_for_loss(
            net=list(self.nets.values()),
            optim=self.optimizers['joint'],
            loss=losses['joint_loss'],
        )
        info['grad_norms'] = grad_norms
        return info

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(BC_Verb_Obj, self).log_info(info)
        if "joint_loss" in info["losses"]:
            log["Action_Loss"] = log["Loss"]
            log["Loss"] = info["losses"]["joint_loss"].item()
        if "verb_enc_loss" in info["losses"]:
            log["Verb_Enc_Loss"] = info["losses"]["verb_enc_loss"].item()
        if "obj_enc_loss" in info["losses"]:
            log["Obj_Enc_Loss"] = info["losses"]["obj_enc_loss"].item()
        return log

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training
        embs_dict = self._enc_forward(obs_dict)
        if self.task_emb_mode == "onehot":
            embs_dict['verb_embs'] = torch.exp(embs_dict['verb_embs'])
            embs_dict['obj_embs'] = torch.exp(embs_dict['obj_embs'])
        obs_dict.update(embs_dict)
        return self.nets["policy"](obs_dict, goal_dict=goal_dict)


class BC_Gaussian(BC):
    """
    BC training with a Gaussian policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gaussian.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.GaussianActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            fixed_std=self.algo_config.gaussian.fixed_std,
            init_std=self.algo_config.gaussian.init_std,
            std_limits=(self.algo_config.gaussian.min_std, 7.5),
            std_activation=self.algo_config.gaussian.std_activation,
            low_noise_eval=self.algo_config.gaussian.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )

        self.nets = self.nets.float().to(self.device)

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        dists = self.nets["policy"].forward_train(
            obs_dict=batch["obs"], 
            goal_dict=batch["goal_obs"],
        )

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(dists.batch_shape) == 1
        log_probs = dists.log_prob(batch["actions"])

        predictions = OrderedDict(
            log_probs=log_probs,
        )
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # loss is just negative log-likelihood of action targets
        action_loss = -predictions["log_probs"].mean()
        return OrderedDict(
            log_probs=-action_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Log_Likelihood"] = info["losses"]["log_probs"].item() 
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log


class BC_GMM(BC_Gaussian):
    """
    BC training with a Gaussian Mixture Model policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gmm.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.GMMActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            num_modes=self.algo_config.gmm.num_modes,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )

        self.nets = self.nets.float().to(self.device)


class BC_VAE(BC):
    """
    BC training with a VAE policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.VAEActor(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            device=self.device,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **VAENets.vae_args_from_config(self.algo_config.vae),
        )
        
        self.nets = self.nets.float().to(self.device)

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Update from superclass to set categorical temperature, for categorical VAEs.
        """
        if self.algo_config.vae.prior.use_categorical:
            temperature = self.algo_config.vae.prior.categorical_init_temp - epoch * self.algo_config.vae.prior.categorical_temp_anneal_step
            temperature = max(temperature, self.algo_config.vae.prior.categorical_min_temp)
            self.nets["policy"].set_gumbel_temperature(temperature)
        return super(BC_VAE, self).train_on_batch(batch, epoch, validate=validate)

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        vae_inputs = dict(
            actions=batch["actions"],
            obs_dict=batch["obs"],
            goal_dict=batch["goal_obs"],
            freeze_encoder=batch.get("freeze_encoder", False),
        )

        vae_outputs = self.nets["policy"].forward_train(**vae_inputs)
        predictions = OrderedDict(
            actions=vae_outputs["decoder_outputs"],
            kl_loss=vae_outputs["kl_loss"],
            reconstruction_loss=vae_outputs["reconstruction_loss"],
            encoder_z=vae_outputs["encoder_z"],
        )
        if not self.algo_config.vae.prior.use_categorical:
            with torch.no_grad():
                encoder_variance = torch.exp(vae_outputs["encoder_params"]["logvar"])
            predictions["encoder_variance"] = encoder_variance
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # total loss is sum of reconstruction and KL, weighted by beta
        kl_loss = predictions["kl_loss"]
        recons_loss = predictions["reconstruction_loss"]
        action_loss = recons_loss + self.algo_config.vae.kl_weight * kl_loss
        return OrderedDict(
            recons_loss=recons_loss,
            kl_loss=kl_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["KL_Loss"] = info["losses"]["kl_loss"].item()
        log["Reconstruction_Loss"] = info["losses"]["recons_loss"].item()
        if self.algo_config.vae.prior.use_categorical:
            log["Gumbel_Temperature"] = self.nets["policy"].get_gumbel_temperature()
        else:
            log["Encoder_Variance"] = info["predictions"]["encoder_variance"].mean().item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log


class BC_RNN(BC):
    """
    BC training with an RNN policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.RNNActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.rnn_args_from_config(self.algo_config.rnn),
        )

        self._rnn_hidden_state = None
        self._rnn_horizon = self.algo_config.rnn.horizon
        self._rnn_counter = 0
        self._rnn_is_open_loop = self.algo_config.rnn.get("open_loop", False)

        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        input_batch = dict()
        input_batch["obs"] = batch["obs"]
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"]

        if self._rnn_is_open_loop:
            # replace the observation sequence with one that only consists of the first observation.
            # This way, all actions are predicted "open-loop" after the first observation, based
            # on the rnn hidden state.
            n_steps = batch["actions"].shape[1]
            obs_seq_start = TensorUtils.index_at_time(batch["obs"], ind=0)
            input_batch["obs"] = TensorUtils.unsqueeze_expand_at(obs_seq_start, size=n_steps, dim=1)

        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training

        if self._rnn_hidden_state is None or self._rnn_counter % self._rnn_horizon == 0:
            batch_size = list(obs_dict.values())[0].shape[0]
            self._rnn_hidden_state = self.nets["policy"].get_rnn_init_state(batch_size=batch_size, device=self.device)

            if self._rnn_is_open_loop:
                # remember the initial observation, and use it instead of the current observation
                # for open-loop action sequence prediction
                self._open_loop_obs = TensorUtils.clone(TensorUtils.detach(obs_dict))

        obs_to_use = obs_dict
        if self._rnn_is_open_loop:
            # replace current obs with last recorded obs
            obs_to_use = self._open_loop_obs

        self._rnn_counter += 1
        action, self._rnn_hidden_state = self.nets["policy"].forward_step(
            obs_to_use, goal_dict=goal_dict, rnn_state=self._rnn_hidden_state)
        return action

    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        self._rnn_hidden_state = None
        self._rnn_counter = 0


class BC_RNN_GMM(BC_RNN):
    """
    BC training with an RNN GMM policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gmm.enabled
        assert self.algo_config.rnn.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.RNNGMMActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            num_modes=self.algo_config.gmm.num_modes,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.rnn_args_from_config(self.algo_config.rnn),
        )

        self._rnn_hidden_state = None
        self._rnn_horizon = self.algo_config.rnn.horizon
        self._rnn_counter = 0
        self._rnn_is_open_loop = self.algo_config.rnn.get("open_loop", False)

        self.nets = self.nets.float().to(self.device)

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        dists = self.nets["policy"].forward_train(
            obs_dict=batch["obs"], 
            goal_dict=batch["goal_obs"],
        )

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(dists.batch_shape) == 2 # [B, T]
        log_probs = dists.log_prob(batch["actions"])

        predictions = OrderedDict(
            log_probs=log_probs,
        )
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # loss is just negative log-likelihood of action targets
        action_loss = -predictions["log_probs"].mean()
        return OrderedDict(
            log_probs=-action_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Log_Likelihood"] = info["losses"]["log_probs"].item() 
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log
