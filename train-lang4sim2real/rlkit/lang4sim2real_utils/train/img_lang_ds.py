import collections
import os
import random

import h5py
import numpy as np
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm

from rlkit.torch.pretrained_models.language_models import (
    LM_STR_TO_FN_CLASS_MAP)
import rlkit.util.experiment_script_utils as exp_utils
from rlkit.util.visualization_utils import plot_tsne_embs_by_stage


class ImgLangDataset(Dataset):
    def __init__(
            self, img_dir, l2_unit_normalize, hdf5_kwargs={},
            two_stage_pp=0, override_lang_params={}, domain=1, rephrasing_csv="",
            env_name="", realrobot_target_obj="", out_dir="",
            mean_rephrase_emb=False, shuffle_demos=False, use_pred_stages=False):
        self.hdf5_kwargs = hdf5_kwargs
        emb_model_class = LM_STR_TO_FN_CLASS_MAP["minilm"]
        self.lang_enc = emb_model_class(l2_unit_normalize=l2_unit_normalize, gpu=0)
        self.two_stage_pp = two_stage_pp
        self.domain = domain
        self.override_lang_params = override_lang_params
        self.rephrase_lang = rephrasing_csv != ""
        self.env_name = env_name
        self.out_dir = out_dir
        self.mean_rephrase_emb = mean_rephrase_emb
        self.shuffle_demos = shuffle_demos
        self.use_pred_stages = use_pred_stages
        if self.rephrase_lang:
            assert self.two_stage_pp == 0, (
                "Rephrasing support only when not doing <7 stage ablations.")
            assert env_name in [
                "PPObjToPotToStove", "PPObjToPotToStove_ang1_fr5damp50",
                "frka_obj_bowl_plate", "WrapUnattachedWire", "frka_wirewrap"]
            self.idx_to_rephrasing_map, self.template_to_rephrasings_map = (
                self.create_rephrasing_list(rephrasing_csv))
            # We need the env to get the obj and cont names to get the template strings.
            variant = {
                'image_dim': 128,
                'eval_image_dim': 128,
                'state_mode': 1,
                'env_type': exp_utils.get_env_type(env_name),
                # real robot env.
                'realrobot_target_obj': realrobot_target_obj,  # not used in lang rephrasings
                'num_tasks': 2,
                'pass_in_ww_env_args': False,
            }
            if env_name in ["WrapUnattachedWire", "frka_wirewrap"]:
                variant.update(dict(
                    pass_in_ww_env_args=True,
                    realrobot_obj_set=1,  # cable and spool; not used in lang rephrasings anyway.
                ))
            self.env = exp_utils.init_env(env_name, variant, env_idx=None)
        else:
            self.num_rephrasings = 0
        (self.imgs, self.langs, self.unique_lang_idxs,
            self.lang_str_list_in_idx_order,
            self.unique_lang_idx_to_lang_emb_matrix,
            self.task_idx_to_num_stages_map) = self.load_imgs_langs(img_dir)

    def __len__(self):
        return self.langs.shape[0]

    def __getitem__(self, idx):
        x, y, y_unique_idx = (
            self.imgs[idx], self.langs[idx], self.unique_lang_idxs[idx])

        if self.rephrase_lang:
            # override the lang emb.
            rephrasing_lang_embs = self.unique_lang_idx_to_lang_emb_matrix[y_unique_idx]  # (self.num_rephrasings + 1, 384)
            if self.mean_rephrase_emb:
                # Take a mean over all rephrasing lang embs.
                chosen_rephrasing_lang_emb = torch.mean(rephrasing_lang_embs, dim=0)
            else:
                rephrase_idx = np.random.choice(rephrasing_lang_embs.shape[0])
                chosen_rephrasing_lang_emb = rephrasing_lang_embs[rephrase_idx]
            y = chosen_rephrasing_lang_emb

        x = x.astype(np.float32).transpose(2, 0, 1) / 255.
        x = torch.tensor(x)
        return x, y, y_unique_idx

    def create_rephrasing_list(self, rephrasing_csv):
        df = pd.read_csv(rephrasing_csv)
        template_to_rephrasings_map = {}  # maps from template string to list of rephrasings
        fn_list_idx_to_rephrasing_map = {}  # maps from fn list idx to list of rephrasings

        # Get distinct values of fn_list_idx
        fn_list_idxs = df['fn_list_idx'].unique()

        max_num_rephrasings = 0

        for fn_list_idx in fn_list_idxs:
            rephrasings = df.loc[df['fn_list_idx'] == fn_list_idx]['rephrased_lang'].tolist()
            fn_list_idx_to_rephrasing_map[fn_list_idx] = rephrasings

            orig_lang = list(set(df.loc[df['fn_list_idx'] == fn_list_idx]['orig_lang'].tolist()))
            assert len(orig_lang) == 1
            orig_lang = orig_lang[0]
            template_to_rephrasings_map[orig_lang] = rephrasings

            if len(rephrasings) > max_num_rephrasings:
                print("len(rephrasings)", len(rephrasings))
                assert max_num_rephrasings == 0, "All num rephrasings should be the same."
                max_num_rephrasings = max(max_num_rephrasings, len(rephrasings))

        self.num_rephrasings = max_num_rephrasings
        return fn_list_idx_to_rephrasing_map, template_to_rephrasings_map

    def load_imgs_langs(self, path):
        if os.path.splitext(path)[-1] == ".hdf5":
            out = self.load_imgs_langs_from_hdf5(path)
        elif os.path.splitext(path)[-1] == "":
            out = self.load_imgs_langs_from_dir(path)
        else:
            raise NotImplementedError
        return out

    def load_imgs_langs_from_hdf5(self, hdf5_path):
        imgs = []
        langs = []
        unique_lang_idxs = []
        # --This is now a list of idxs corresponding to
        #   the idx in lang_str_rephrase_list_in_idx_order,
        #   which is different from the idx in
        #  lang_str_list_in_idx_order when self.rephrase_lang = True.
        # unique_lang_to_idx_map = {}
        lang_str_list_in_idx_order = []
        lang_str_rephrase_list_in_idx_order = []
        # --list of lists. for self.rephrase_lang = False,
        #   each element is a len-1 list.
        #   for self.rephrase_lang = True, each element is a
        #   len-(self.num_rephrasings + 1) list, where the
        #   0th element is the template string.

        # Set unique_lang_idxs
        task_idx_to_lang_idx_offset = {}
        task_idx_to_num_stages_map = {}

        with h5py.File(hdf5_path, 'r', swmr=True, libver='latest') as f:
            max_demos_per_task = self.hdf5_kwargs.get(
                'max_demos_per_task', np.inf)
            task_idx_to_num_demos_loaded = collections.Counter()
            ds_task_idxs = [int(i) for i in list(f['data'].keys())]
            task_idxs_to_load = self.hdf5_kwargs.get(
                'task_indices', ds_task_idxs)
            for task_idx in task_idxs_to_load:
                if self.two_stage_pp==0:
                    task_lang_desc_list = (
                        f[f'data/{task_idx}'].attrs['lang_list'])
                    if task_idx not in task_idx_to_num_stages_map:
                        task_idx_to_num_stages_map[task_idx] = len(task_lang_desc_list)
                    task_idx_to_lang_idx_offset[task_idx] = len(lang_str_rephrase_list_in_idx_order)
                    if len(self.override_lang_params) > 0:
                        task_lang_desc_list = self.override_lang_params["fn"](
                            **self.override_lang_params["fn_kwargs"])
                    for stage_num, lang_desc in enumerate(task_lang_desc_list):
                        # assert lang_desc not in unique_lang_to_idx_map
                        # If this assert fails, then we might need to revisit
                        # `maybe_set_lang_desc_attrs` in
                        # rlkit/data_management/dataset_from_hdf5.py
                        # Update 2023/12/13: I think it's fine to comment it out.
                        # Ok for merged phase 1-2 emb mat to have multiple of the same langs.
                        # unique_lang_to_idx_map[lang_desc] = len(unique_lang_to_idx_map)
                        if not self.rephrase_lang:
                            lang_str_list_in_idx_order.append(lang_desc)
                            lang_str_rephrase_list_in_idx_order.append([lang_desc])
                        elif self.env_name in [
                                "PPObjToPotToStove", "PPObjToPotToStove_ang1_fr5damp50",
                                "frka_obj_bowl_plate"]:
                            # Extract template from the element in lang_list
                            cont_names = self.env.step_kwargs['cont_names']
                            obj_names = self.env.step_kwargs['obj_names']
                            stages_per_step = int(len(task_lang_desc_list) // len(cont_names))
                            step_idx = int(stage_num // stages_per_step)
                            # We assumes that each step has the same number of stages.
                            target_obj_name = f[f'data/{task_idx}/demo_0'].attrs['target_obj'].lower()
                            obj_name = obj_names[step_idx].replace(
                                "<target_obj>", target_obj_name).replace("_", " ")
                            cont_name = cont_names[step_idx].replace("_", " ")
                            template = lang_desc.replace(obj_name, "{obj_name}").replace(cont_name, "{cont_name}")
                            print("fixed template", template)
                            rephrasings = self.template_to_rephrasings_map[template]

                            # substitute {obj_name}, {cont_name} in rephrasings
                            rephrasings = [
                                r.format(obj_name=obj_name, cont_name=cont_name)
                                for r in rephrasings]
                            lang_str_list_in_idx_order.extend([lang_desc] + rephrasings)
                            lang_str_rephrase_list_in_idx_order.append([lang_desc] + rephrasings)
                        elif self.env_name in ["WrapUnattachedWire", "frka_wirewrap"]:
                            # TODO: remove hardcoded obj names. For instance, this only works for p1 != p3 case.
                            if self.env_name == "WrapUnattachedWire":
                                lang_kwargs = dict(
                                    grasp_obj_name="last bead",
                                    flex_wraparound_obj_name="beads",
                                    central_obj_name="cylinder",
                                )
                            elif self.env_name == "frka_wirewrap":
                                lang_kwargs = dict(
                                    grasp_obj_name="bridge",
                                    flex_wraparound_obj_name="ethernet cable",
                                    central_obj_name="3d printer spool",
                                )
                            template = lang_desc.replace(
                                lang_kwargs['grasp_obj_name'], "{grasp_obj_name}").replace(
                                lang_kwargs['flex_wraparound_obj_name'], "{flex_wraparound_obj_name}").replace(
                                lang_kwargs['central_obj_name'], "{central_obj_name}")
                            print("fixed template", template)
                            rephrasings = self.template_to_rephrasings_map[template]
                            rephrasings = [
                                r.format(**lang_kwargs)
                                for r in rephrasings]
                            lang_str_list_in_idx_order.extend([lang_desc] + rephrasings)
                            lang_str_rephrase_list_in_idx_order.append([lang_desc] + rephrasings)
                        else:
                            raise NotImplementedError
                else:
                    if self.two_stage_pp != -1:
                        task_idx_to_lang_idx_offset[task_idx] = self.two_stage_pp * task_idx
                        task_idx_to_num_stages_map[task_idx] = self.two_stage_pp
                    else:
                        task_idx_to_lang_idx_offset[task_idx] = 0
                        task_idx_to_num_stages_map[task_idx] = 2
                # import ipdb; ipdb.set_trace()
                demo_ids = list(f[f'data/{task_idx}'].keys())
                if self.shuffle_demos:
                    random.shuffle(demo_ids)
                for demo_id in tqdm(demo_ids):
                    # Break out if enough demos have been added.
                    if task_idx_to_num_demos_loaded[task_idx] >= max_demos_per_task:
                        print(
                            f"task_idx {task_idx}, loaded demo ids:",
                            demo_ids[:max_demos_per_task])
                        break
                    imgs.append(
                        f[f'data/{task_idx}/{demo_id}/observations/image'][()])
                    # langs.append([f[f'data/{task_idx}/lang_idx']])
                    if self.two_stage_pp == 0:
                        if self.use_pred_stages:
                            stage_num_key = "pred_stage_num"
                        else:
                            stage_num_key = "lang_stage_num"
                        unique_lang_idxs.append(
                            task_idx_to_lang_idx_offset[task_idx]
                            + f[f'data/{task_idx}/{demo_id}/{stage_num_key}'][()])
                    elif self.two_stage_pp == 4:
                        original_lang_idx = f[f'data/{task_idx}/{demo_id}/lang_stage_num'][()]
                        mask = original_lang_idx % 7 >= 3
                        stage = np.zeros_like(original_lang_idx)
                        stage[mask] = 1
                        unique_lang_idxs.append(
                            task_idx_to_lang_idx_offset[task_idx] + stage + (2 * (original_lang_idx // 7)))
                    elif self.two_stage_pp == 8:
                        original_lang_idx = f[f'data/{task_idx}/{demo_id}/lang_stage_num'][()]
                        stage = np.zeros_like(original_lang_idx)
                        stage[(0 <= original_lang_idx) & (original_lang_idx <= 1)] = 0
                        stage[(2 == original_lang_idx)] = 1
                        stage[(3 <= original_lang_idx) & (original_lang_idx <= 5)] = 2
                        stage[(6 == original_lang_idx)] = 3
                        stage[(7 <= original_lang_idx) & (original_lang_idx <= 8)] = 4
                        stage[(9 == original_lang_idx)] = 5
                        stage[(10 <= original_lang_idx) & (original_lang_idx <= 12)] = 6
                        stage[(13 == original_lang_idx)] = 7
                        assert task_idx_to_lang_idx_offset[task_idx] % 8 == 0
                        unique_lang_idxs.append(
                            task_idx_to_lang_idx_offset[task_idx] + stage)
                    elif self.two_stage_pp == 6:
                        #ww
                        original_lang_idx = f[f'data/{task_idx}/{demo_id}/lang_stage_num'][()]
                        stage = np.zeros_like(original_lang_idx)
                        stage[(0 <= original_lang_idx) & (original_lang_idx <= 1)] = 0
                        stage[(2 <= original_lang_idx) & (original_lang_idx <= 3)] = 1
                        stage[(4 <= original_lang_idx) & (original_lang_idx <= 8)] = 2
                        stage[(9 == original_lang_idx) & (original_lang_idx <= 13)] = 3
                        stage[(14 == original_lang_idx)] = 4
                        stage[(15 == original_lang_idx)] = 5
                        assert task_idx_to_lang_idx_offset[task_idx] % 6 == 0
                        unique_lang_idxs.append(
                            task_idx_to_lang_idx_offset[task_idx] + stage)
                    elif self.two_stage_pp==2:
                        original_lang_idx = f[f'data/{task_idx}/{demo_id}/lang_stage_num'][()]
                        mask = original_lang_idx >= 7
                        stage = np.zeros_like(original_lang_idx)
                        stage[mask] = 1
                        unique_lang_idxs.append(
                            task_idx_to_lang_idx_offset[task_idx] + stage + ((original_lang_idx // 14)))
                    elif self.two_stage_pp==1:
                        original_lang_idx = f[f'data/{task_idx}/{demo_id}/lang_stage_num'][()]
                        stage = np.zeros_like(original_lang_idx) # all stages are 0
                        unique_lang_idxs.append(stage) #always append zeros
                    elif self.two_stage_pp== -1: #wire wrap
                        # import ipdb; ipdb.set_trace()
                        original_lang_idx = f[f'data/{task_idx}/{demo_id}/lang_stage_num'][()]
                        mask = original_lang_idx == 14
                        stage = np.zeros_like(original_lang_idx) # all stages are 0
                        stage[mask] = 1
                        unique_lang_idxs.append(
                            task_idx_to_lang_idx_offset[task_idx] + stage)
                    elif self.two_stage_pp==-2:
                        original_lang_idx = f[f'data/{task_idx}/{demo_id}/lang_stage_num'][()]
                        stage = np.zeros_like(original_lang_idx) # all stages are 0
                        if self.domain == 1:
                            unique_lang_idxs.append(stage) #always append zeros
                        else:
                            unique_lang_idxs.append(stage + 1) #always append ones
                    task_idx_to_num_demos_loaded[task_idx] += 1
            # import ipdb; ipdb.set_trace()
            domain_obj_list = ["milk small", "bread", "can", "cereal small"] if self.domain == 1 else ["bridge"]
            domain_container_1 = "pot" if self.domain == 1 else "clear container"
            domain_container_2 = "stove" if self.domain == 1 else "plate"
            if self.two_stage_pp == 4:
                lang_str_list_in_idx_order = []
                for obj in domain_obj_list:
                    lang_str_list_in_idx_order.append(f"reaching toward {obj}")
                    lang_str_list_in_idx_order.append(f"holding {obj}, moving toward {domain_container_1}")
                    lang_str_list_in_idx_order.append(f"reaching toward {domain_container_1}")
                    lang_str_list_in_idx_order.append(f"holding {domain_container_1}, reaching toward {domain_container_2}")
            elif self.two_stage_pp == 2:
                lang_str_list_in_idx_order = []
                for obj in domain_obj_list:
                    lang_str_list_in_idx_order.append(f"picking {obj} and putting in {domain_container_1}")
                    lang_str_list_in_idx_order.append(f"picking {domain_container_1} and putting in {domain_container_2}")
            elif self.two_stage_pp == 8:
                lang_str_list_in_idx_order = []
                for obj in domain_obj_list:
                    lang_str_list_in_idx_order.append(f'gripper open, reaching for {obj}, out of {domain_container_1}')
                    lang_str_list_in_idx_order.append(f'gripper closing, with {obj}, out of {domain_container_1}')
                    lang_str_list_in_idx_order.append(f'gripper closed, moving up with {obj}')
                    lang_str_list_in_idx_order.append(f'gripper open, dropped {obj}, in {domain_container_1}')

                    lang_str_list_in_idx_order.append(f'gripper open, reaching for {domain_container_1}, out of {domain_container_2}')
                    lang_str_list_in_idx_order.append(f'gripper closing, with {domain_container_1}, out of {domain_container_2}')
                    lang_str_list_in_idx_order.append(f'gripper closed, moving up with {domain_container_1}')
                    lang_str_list_in_idx_order.append(f'gripper open, dropped {domain_container_1}, in {domain_container_2}')
            elif self.two_stage_pp == 6:
                lang_str_list_in_idx_order = []
                if self.domain == 1: 
                    flex_obj = "beads"
                    grasp_obj = "last bead"
                    center = "cylinder"
                else:
                    flex_obj = "ethernet cable"
                    grasp_obj = "bridge"
                    center = "3d printer"

                for obj in domain_obj_list:
                    lang_str_list_in_idx_order.append(f'gripper open, reaching for {grasp_obj}')
                    lang_str_list_in_idx_order.append(f'gripper closing and lifting {grasp_obj}')
                    lang_str_list_in_idx_order.append(f'counter-clockwise')
                    lang_str_list_in_idx_order.append(f'clockwise')

                    lang_str_list_in_idx_order.append(f'gripper open, above {center} with {flex_obj} fully wrapped')
                    lang_str_list_in_idx_order.append(f'gripper open, above {center} with {flex_obj} fully unwrapped')
            elif self.two_stage_pp == 1:
                lang_str_list_in_idx_order = ["gripper open, reaching for bread, out of pot"]
            elif self.two_stage_pp == -1:#wire wrap
                if self.domain == 1:
                    lang_str_list_in_idx_order = ["picking and wrapping beads around cylinder", "beads fully wrapped"]
                else:
                    lang_str_list_in_idx_order = ["picking and wrapping ethernet cable around 3d printer spool", "ethernet cable fully wrapped"]
            elif self.two_stage_pp == -2:
                lang_str_list_in_idx_order = ["", ""]

        # import ipdb; ipdb.set_trace()
        print("task_idx to num_demos loaded", task_idx_to_num_demos_loaded)
        imgs = np.concatenate(imgs, axis=0)  # uint8 (dataset_size, img_h, img_w, 3)
        unique_lang_idxs = np.concatenate(unique_lang_idxs, axis=0) # (dataset_size,)
        unique_lang_idx_to_lang_emb_matrix = self.lang_enc(lang_str_list_in_idx_order)  # (len(lang_str_list_in_idx_order), 384)
        langs = unique_lang_idx_to_lang_emb_matrix[unique_lang_idxs]  # float (dataset_size, 384)

        if self.two_stage_pp == -2:
            # import ipdb; ipdb.set_trace()
            torch.manual_seed(1)
            d1_emb = torch.randn((384)).to("cuda:0")
            d1_emb = d1_emb / torch.norm(d1_emb)
            d2_emb =  torch.randn((384)).to("cuda:0")
            d2_emb = d2_emb / torch.norm(d2_emb)
            unique_lang_idx_to_lang_emb_matrix = torch.zeros((2, 384)).to("cuda:0") # (len(lang_str_list_in_idx_order), 384)
            unique_lang_idx_to_lang_emb_matrix[0] = d1_emb
            unique_lang_idx_to_lang_emb_matrix[1] = d2_emb
            langs = unique_lang_idx_to_lang_emb_matrix[unique_lang_idxs]
            torch.seed()

        if self.rephrase_lang:
            # reshape unique_lang_idx_to_lang_emb_matrix
            # into (len(lang_str_list_in_idx_order), 1 + self.num_rephrasings, 384)
            rephrase_list_shape = np.array(lang_str_rephrase_list_in_idx_order).shape
            unique_lang_idx_to_lang_emb_matrix = (
                unique_lang_idx_to_lang_emb_matrix.reshape(
                    rephrase_list_shape[0], rephrase_list_shape[1], 384))
            # langs = unique_lang_idx_to_lang_emb_matrix[unique_lang_idxs]
            langs = np.zeros(langs.shape[0])  # will be overriden in __getitem__
            # float (dataset_size, self.num_rephrasings + 1, 384)
            self.plot_rephrasing_lang_embs(
                unique_lang_idx_to_lang_emb_matrix,
                task_idx_to_num_stages_map[0])

        return (
            imgs, langs, unique_lang_idxs, lang_str_list_in_idx_order,
            unique_lang_idx_to_lang_emb_matrix, task_idx_to_num_stages_map)

    def plot_rephrasing_lang_embs(
            self, unique_lang_idx_to_lang_emb_matrix, num_stages):
        # Plot only lang for task 0
        num_langs_per_stage = self.num_rephrasings + 1
        stage_idxs = np.array(
            [[i] * num_langs_per_stage
            for i in range(num_stages)])  # (num_stages, self.num_rephrasings + 1)
        stage_idxs = np.concatenate(list(stage_idxs), axis=0)  # ((self.num_rephrasings + 1) * num_stages)
        # Assumes task 0 langs are all at the top of `unique_lang_idx_to_lang_emb_matrix`.
        lang_embs = torch.cat(
            list(unique_lang_idx_to_lang_emb_matrix[:num_stages]),
            dim=0)
        lang_embs = lang_embs.cpu().detach().numpy()
        tsne_embs = plot_tsne_embs_by_stage(
            lang_embs, stage_idxs, self.out_dir, "rephrasing_embs",
            idx_annotation_interval=num_langs_per_stage,
            plot_cluster_means=True)
        return tsne_embs

    # Commenting out for readability; should still work.
    # def load_imgs_langs_from_dir(self, img_dir):
    #     labels_df = pd.read_csv(os.path.join(img_dir, "labels.csv"))
    #     imgs = []
    #     langs = []
    #     unique_lang_idxs = []
    #     unique_lang_to_idx_map = {}
    #     lang_str_list_in_idx_order = []
    #     for i, row in tqdm(labels_df.iterrows()):
    #         img_fpath = os.path.join(img_dir, row['img_fname'])
    #         img = np.array(Image.open(img_fpath))
    #         imgs.append(img)

    #         langs.append(row['lang'])

    #         if row['lang'] not in unique_lang_to_idx_map:
    #             unique_lang_to_idx_map[row['lang']] = len(unique_lang_to_idx_map)
    #             lang_str_list_in_idx_order.append(row['lang'])

    #         unique_lang_idxs.append(unique_lang_to_idx_map[row['lang']])

    #     imgs = np.array(imgs)  # uint8 (dataset_size, img_h, img_w, 3)
    #     langs = self.lang_enc(langs)  # float (dataset_size, 384)

    #     unique_lang_idx_to_lang_emb_matrix = self.lang_enc(lang_str_list_in_idx_order)
    #     return (
    #         imgs, langs, unique_lang_idxs, unique_lang_to_idx_map,
    #         unique_lang_idx_to_lang_emb_matrix)


class ImgLangTimeContrDataset(Dataset):
    def __init__(self, img_dir, l2_unit_normalize, hdf5_kwargs={}):
        self.hdf5_kwargs = hdf5_kwargs
        emb_model_class = LM_STR_TO_FN_CLASS_MAP["minilm"]
        self.lang_enc = emb_model_class(l2_unit_normalize=l2_unit_normalize, gpu=0)
        (self.imgs, self.langs, self.unique_lang_idxs,
            self.lang_str_list_in_idx_order,
            self.unique_lang_idx_to_lang_emb_matrix) = self.load_imgs_langs_from_hdf5(img_dir)

    def __len__(self):
        return self.langs.shape[0]

    # def __iter__(self):
    #     while True:
    #         yield self._sample()

    def __getitem__(self, idx):
        # idx = np.random.randint(0, len(self))
        img_traj, lang_traj, lang_unique_idx = (
            self.imgs[idx], self.langs[idx], self.unique_lang_idxs[idx])

        t1 = np.random.randint(1, img_traj.shape[0] - 1)
        t0 = np.random.randint(0, t1)
        t2 = np.random.randint(t1, img_traj.shape[0])

        # print(t0, t1, t2)
        # t0, t1, t2 = 10, 100, 199
        x = img_traj[[t0, t1, t2]]
        y = lang_traj[[t0, t1, t2]]
        y_unique_idx = lang_unique_idx[[t0, t1, t2]]

        x = x.astype(np.float32).transpose(0, 3, 1, 2) / 255.
        x = torch.tensor(x)
        return x, y, y_unique_idx

    def load_imgs_langs_from_hdf5(self, hdf5_path):
        imgs = []
        langs = []
        unique_lang_idxs = []
        # unique_lang_to_idx_map = {}
        lang_str_list_in_idx_order = []
        # Set unique_lang_idxs
        task_idx_to_lang_idx_offset = {}

        with h5py.File(hdf5_path, 'r', swmr=True, libver='latest') as f:
            max_demos_per_task = self.hdf5_kwargs.get(
                'max_demos_per_task', np.inf)
            task_idx_to_num_demos_loaded = collections.Counter()
            ds_task_idxs = [int(i) for i in list(f['data'].keys())]
            task_idxs_to_load = self.hdf5_kwargs.get(
                'task_indices', ds_task_idxs)
            for task_idx in task_idxs_to_load:
                task_lang_desc_list = (
                    f[f'data/{task_idx}'].attrs['lang_list'])
                task_idx_to_lang_idx_offset[task_idx] = len(lang_str_list_in_idx_order)

                for lang_desc in task_lang_desc_list:
                    # assert lang_desc not in unique_lang_to_idx_map
                    # See comment above for this same line.
                    # unique_lang_to_idx_map[lang_desc] = len(unique_lang_to_idx_map)
                    lang_str_list_in_idx_order.append(lang_desc)
                for demo_id in tqdm(f[f'data/{task_idx}'].keys()):
                    # Break out if enough demos have been added.
                    if task_idx_to_num_demos_loaded[task_idx] >= max_demos_per_task:
                        break
                    imgs.append(
                        f[f'data/{task_idx}/{demo_id}/observations/image'][()])
                    # langs.append([f[f'data/{task_idx}/lang_idx']])
                    unique_lang_idxs.append(
                        task_idx_to_lang_idx_offset[task_idx]
                        + f[f'data/{task_idx}/{demo_id}/lang_stage_num'][()])
                    task_idx_to_num_demos_loaded[task_idx] += 1

        print("task_idx to num_demos loaded", task_idx_to_num_demos_loaded)
        imgs = np.array(imgs)  # uint8 (num_trajs [over all tasks], max_path_len, img_h, img_w, 3)
        unique_lang_idxs = np.array(unique_lang_idxs) # (num_trajs [over all tasks], max_path_len)
        unique_lang_idx_to_lang_emb_matrix = self.lang_enc(lang_str_list_in_idx_order)
        langs = np.array([
            unique_lang_idx_to_lang_emb_matrix[unique_lang_idxs[i]]
            for i in range(unique_lang_idxs.shape[0])])  # float (num_trajs, max_path_len, 384)

        return (
            imgs, langs, unique_lang_idxs, lang_str_list_in_idx_order,
            unique_lang_idx_to_lang_emb_matrix)


if __name__ == "__main__":
    pass
