import argparse

import torch

from rlkit.torch.networks.cnn import ClipWrapper
from rlkit.torch.policies import GaussianStandaloneCNNPolicy, MakeDeterministic

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--type", choices=["clip", "r3m"], default=None)
    args = parser.parse_args()

    # policy_params = {
    #     'max_log_std': 0,
    #     'min_log_std': -6,
    #     'obs_dim': None,
    #     'std_architecture': 'values',
    #     'freeze_policy_cnn': True,
    #     'lang_emb_obs_is_tokens': False,
    #     'gripper_policy_arch': 'ac_dim',
    #     'gripper_loss_type': None,
    #     'output_dist_learning_cnn_embs_in_aux': False,
    #     'aug_transforms': ['pad_crop'],
    #     'image_augmentation_padding': 4,
    #     'image_size': (128, 128, 3),
    #     'batchnorm': False,
    #     'hidden_sizes': [1024, 512, 256],
    #     'std': None,
    #     'output_activation': lambda x: x,
    #     'cnn_output_stage': '',
    #     'use_spatial_softmax': False,
    #     'state_obs_dim': 22,
    #     'emb_obs_keys': ['lang_embedding'],
    #     'aux_tasks': [],
    #     'aux_obs_bounds': {},
    #     'aux_to_feed_fc': 'none',
    #     'obs_key_to_dim_map': {
    #         'image': 49152,
    #         'lang_embedding': 384,
    #         'state': 22},
    #     'observation_keys': ['image', 'lang_embedding', 'state'],
    #     'action_dim': 7,
    # }
    policy_params = {
        'max_log_std': 0,
        'min_log_std': -6,
        'obs_dim': None,
        'std_architecture': 'values',
        'freeze_policy_cnn': True,
        'lang_emb_obs_is_tokens': False,
        'gripper_policy_arch': 'ac_dim',
        'gripper_loss_type': None,
        'output_dist_learning_cnn_embs_in_aux': False,
        'aug_transforms': ['pad_crop'],
        'image_augmentation_padding': 4,
        'image_size': (128, 128, 3),
        'batchnorm': False,
        'hidden_sizes': [1024, 512, 256],
        'std': None,
        'output_activation': lambda x: x,
        'cnn_output_stage': '',
        'use_spatial_softmax': False,
        'state_obs_dim': 22,
        'emb_obs_keys': ['lang_embedding'],
        'aux_tasks': [],
        'aux_obs_bounds': {},
        'aux_to_feed_fc': 'none',
        'obs_key_to_dim_map': {
            'image': 49152,
            'lang_embedding': 384,
            'state': 22},
        'observation_keys': ['image', 'lang_embedding', 'state'],
        'film_emb_dim_list': [],
        'action_dim': 7,
    }
    cnn_params = {
        'clip_checkpoint': '/scratch/cluster/albertyu/dev/open_clip/logs/lr=1e-05_wd=0.1_agg=True_model=ViT-B/32_batchsize=128_workers=8_date=2022-09-23-02-43-39/checkpoints/epoch_1.pt',
        'freeze': True,
        'image_shape': (128, 128, 3),
        'tokenize_scheme': 'clip',
        'image_augmentation_padding': 0,
        'task_lang_list': [],
        'added_fc_input_size': 22
    }

    if args.type == "r3m":
        from r3m import load_r3m
        cnn = load_r3m("resnet18")
        cnn.eval()
        # ckpt = torch.load(args.ckpt)
        # out = ckpt['evaluation/policy']._action_distribution_generator(torch.zeros(1, 49152 + 406))
        policy_params['cnn_out_dim'] = cnn.module.outdim
        cnn_params = {'added_fc_input_size': 406}
        policy = GaussianStandaloneCNNPolicy(
            cnn=cnn,
            **policy_params,
            **cnn_params)
    elif args.type == "clip":
        policy_params["added_fc_input_size"] = cnn_params.pop(
            "added_fc_input_size")
        cnn = ClipWrapper(**cnn_params)
        policy_params['cnn_out_dim'] = cnn.visual_outdim
        policy = GaussianStandaloneCNNPolicy(
            cnn=cnn,
            **policy_params)
    else:
        raise NotImplementedError
    policy.gripper_idx = 5  # eval_env.env.gripper_idx
    policy = MakeDeterministic(policy)
    state_dict = torch.load(args.ckpt, map_location="cuda:0")
    import ipdb; ipdb.set_trace()
    policy.load_state_dict(state_dict['evaluation/policy'], strict=False)
    device = torch.device("cuda")
    policy.to(device)
    import ipdb; ipdb.set_trace()
    policy.get_action(torch.zeros((49152 + 406)).cuda())
