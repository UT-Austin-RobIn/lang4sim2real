import argparse
import itertools
import os

import h5py
import numpy as np
import torch
from bleurt_pytorch import (
    BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer)

from rlkit.lang4sim2real_utils.lang_templates import ww_lang_by_stages_debug
from rlkit.lang4sim2real_utils.train.train_policy_cnn_lang4sim2real import (
    get_timestamp_str)
from rlkit.lang4sim2real_utils.train.img_lang_ds import ImgLangDataset
from rlkit.plot.plot_llm_dist import plot_diff_matrix
import rlkit.util.experiment_script_utils as exp_utils


class BleurtScorer:
    def __init__(self, args, norm_range=()):
        self.config = BleurtConfig.from_pretrained(
            'lucadiliello/BLEURT-20-D12')
        self.model = BleurtForSequenceClassification.from_pretrained(
            'lucadiliello/BLEURT-20-D12')
        self.tokenizer = BleurtTokenizer.from_pretrained(
            'lucadiliello/BLEURT-20-D12')

        # get attrs for saving npy
        with h5py.File(
                args.dom1_img_dir, 'r', swmr=True, libver='latest') as f:
            self.dom1 = f['data'].attrs['env']
        self.dom1_task_idxs = "_".join(args.dom1_task_idxs)
        with h5py.File(
                args.dom2_img_dir, 'r', swmr=True, libver='latest') as f:
            self.dom2 = f['data'].attrs['env']
        self.dom2_task_idxs = "_".join(args.dom2_task_idxs)

        self.norm_range = norm_range

    def compute_bleurt(self, references, candidates):
        assert len(references) == len(candidates)
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(
                references, candidates, padding='longest', return_tensors='pt')
            scores = self.model(**inputs).logits.flatten().tolist()
        scores = np.array(scores)
        if len(self.norm_range) == 2:
            assert self.norm_range[0] < self.norm_range[1]
            # scores = (scores - np.mean(scores)) / (np.std(scores))
            scores = (
                (scores - np.min(scores)) / (np.max(scores) - np.min(scores)))
            mean_adjustment = np.mean(self.norm_range) - 0.5
            _range = self.norm_range[1] - self.norm_range[0]
            scores = _range * (scores + mean_adjustment)
            # scale [0, 1] --> [self.norm_range[0], self.norm_range[1]]
        return scores

    def compute_cart_prod_bleurt(self, lang1, lang2):
        # if len(lang1) = m, len(lang2) = n, then scores
        # will be an array of shape (m, n)
        prod = list(itertools.product(lang1, lang2))
        references = [x[0] for x in prod]
        candidates = [x[1] for x in prod]
        scores = self.compute_bleurt(references, candidates)
        scores = scores.reshape(len(lang1), len(lang2))
        target_diff_fname = (
            f"{get_timestamp_str()}_{self.dom1}_{self.dom1_task_idxs}_"
            f"{self.dom2}_{self.dom2_task_idxs}.npy")
        np.save(target_diff_fname, scores)
        print(f"scores, shape {scores.shape}\n", scores)
        print("Saved diff mat to", target_diff_fname)
        return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dom1-img-dir", type=str)
    parser.add_argument("--dom2-img-dir", type=str)
    parser.add_argument("--dom1-task-idxs", type=str, nargs="+", default=[])
    parser.add_argument("--dom2-task-idxs", type=str, nargs="+", default=[])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--override-lang", action="store_true", default=False)
    args = parser.parse_args()

    bs = BleurtScorer(args, norm_range=(0, 1))

    hdf5_kwargs_dom1 = {"max_demos_per_task": 2}
    if os.path.splitext(args.dom1_img_dir)[-1] == ".hdf5":
        hdf5_kwargs_dom1['task_indices'] = (
            exp_utils.create_task_indices_from_task_int_str_list(
                args.dom1_task_idxs, np.inf))

    hdf5_kwargs_dom2 = {"max_demos_per_task": 2}
    if os.path.splitext(args.dom2_img_dir)[-1] == ".hdf5":
        hdf5_kwargs_dom2['task_indices'] = (
            exp_utils.create_task_indices_from_task_int_str_list(
                args.dom2_task_idxs, np.inf))

    if args.override_lang:
        ds1_override_lang_params = {
            "fn": ww_lang_by_stages_debug,
            "fn_kwargs": dict(
                grasp_obj_name="last bead",
                flex_wraparound_obj_name="beads",
                central_obj_name="cylinder"),
        }
        ds2_override_lang_params = {
            "fn": ww_lang_by_stages_debug,
            "fn_kwargs": dict(
                grasp_obj_name="eu white plug",
                flex_wraparound_obj_name="cable",
                central_obj_name="blender"),
        }
        print("ds1_override_lang_params", ds1_override_lang_params)
        print("ds2_override_lang_params", ds2_override_lang_params)
        input("Overriding language stages found in buffer. continue?")
    else:
        ds1_override_lang_params = {}
        ds2_override_lang_params = {}

    ds1 = ImgLangDataset(
        args.dom1_img_dir,
        l2_unit_normalize=True,
        hdf5_kwargs=hdf5_kwargs_dom1,
        override_lang_params=ds1_override_lang_params)
    ds2 = ImgLangDataset(
        args.dom2_img_dir,
        l2_unit_normalize=True,
        hdf5_kwargs=hdf5_kwargs_dom2,
        override_lang_params=ds2_override_lang_params)

    dom1_langs = ds1.lang_str_list_in_idx_order
    # dom1_langs = [x.replace(" small", "") for x in dom1_langs]
    dom2_langs = ds2.lang_str_list_in_idx_order
    # dom2_langs = [x.replace("bridge", "block") for x in dom2_langs]
    print("dom1_langs", dom1_langs)
    print("dom2_langs", dom2_langs)
    print(f"dom1 x dom2 size: {len(dom1_langs)} x {len(dom2_langs)}")

    scores = bs.compute_cart_prod_bleurt(dom1_langs, dom2_langs)
    plot_fname = f"{get_timestamp_str()}.png"
    plot_diff_matrix(
        scores, plot_fname, dom1_langs, dom2_langs, "Bleurt raw scores")
    print("saved plot to", plot_fname)
