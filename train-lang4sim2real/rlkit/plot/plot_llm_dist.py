import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import softmax
from tqdm import tqdm

from rlkit.lang4sim2real_utils.train.train_policy_cnn_lang4sim2real import (
    pairwise_diffs_matrix)
from rlkit.torch.pretrained_models.language_models import (
    LM_STR_TO_FN_CLASS_MAP)


def encode_lang(img_dir, lang_enc):
    labels_df = pd.read_csv(os.path.join(img_dir, "labels.csv"))

    task_tuple_to_lang_map = {}
    for i, row in tqdm(labels_df.iterrows()):
        row_lang = row['lang']
        if row_lang not in task_tuple_to_lang_map.values():
            print(row['img_fname'])
            fname = row['img_fname'].split("/")[-1].split(".")[0].split("_")
            task_idx, ts = fname[0], fname[-1]
            task_tuple_to_lang_map[(int(task_idx), int(ts))] = row_lang

    # sort the language by traj order and task order.
    # [[task 0 ordered traj lang lists], [task 1 ordered traj lang list], ...]
    lang_lists_by_task = []
    sorted_task_idxs = sorted(list(set([
        task_idx for (task_idx, _), _ in task_tuple_to_lang_map.items()])))
    for task_idx in sorted_task_idxs:
        sorted_ts = sorted(list(set([
            ts for task_id, ts in task_tuple_to_lang_map
            if task_id == task_idx])))
        task_lang_list = []
        for ts in sorted_ts:
            lang = task_tuple_to_lang_map[(task_idx, ts)]
            task_lang_list.append(lang)
        lang_lists_by_task.append(task_lang_list)

    flattened_langs = [l for sublist in lang_lists_by_task for l in sublist]
    langs = lang_enc(flattened_langs)  # float (dataset_size, 386)

    out_dict = dict(
        lang_lists_by_task=lang_lists_by_task,
        flattened_langs=flattened_langs,
        lang_embs=langs,
    )
    return out_dict


def box_around_cells(x, y, xlen, ylen, ax=None, **kwargs):
    rect = plt.Rectangle((x-.5, y-.5), xlen, ylen, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect


def plot(args):
    emb_model_class = LM_STR_TO_FN_CLASS_MAP["minilm"]
    dist_fn_to_norm_bool_map = {
        "l2": False,
        "dotprod": True,
    }
    lang_enc = emb_model_class(
        l2_unit_normalize=dist_fn_to_norm_bool_map[args.dist_fn], gpu=0)
    dom1_lang_dict = encode_lang(args.dom1_img_dir, lang_enc)
    dom2_lang_dict = encode_lang(args.dom2_img_dir, lang_enc)
    diff_matrix = pairwise_diffs_matrix(
        dom1_lang_dict['lang_embs'], dom2_lang_dict['lang_embs'], args.dist_fn)
    diff_matrix = np.array(diff_matrix.detach().cpu())
    # adding softmax
    if args.softmax_temp > 0.0:
        coeff = -1 if args.dist_fn == "l2" else 1
        diff_matrix = softmax(
            coeff * (diff_matrix / args.softmax_temp), axis=-1)
    print("diff_matrix", diff_matrix)
    title_str = f"Cross Domain LLM Embed {args.dist_fn} distances."
    if args.softmax_temp:
        title_str += f" Softmax temp: {args.softmax_temp}."
    plot_diff_matrix(
        diff_matrix, "20230923_llmdist.png",
        dom1_lang_dict['flattened_langs'],
        dom2_lang_dict['flattened_langs'],
        title_str,
        dom1_lang_dict['lang_lists_by_task'],
        dom2_lang_dict['lang_lists_by_task'])


def plot_diff_matrix(
        diff_matrix, fname, lang1, lang2, title_str,
        dom1_lang_lists_by_task=None, dom2_lang_lists_by_task=None):
    plt.switch_backend('agg')
    plt.figure(figsize=(0.4 * len(lang1), 0.4 * len(lang2)))
    plt.matshow(diff_matrix, cmap='viridis_r', fignum=1)
    fontsize = 10
    plt.xticks(
        ticks=np.arange(diff_matrix.shape[1]),
        labels=lang2,
        rotation=90, fontsize=fontsize)
    plt.yticks(
        ticks=np.arange(diff_matrix.shape[0]),
        labels=lang1, fontsize=fontsize)

    # add value text to each cell
    for r in range(diff_matrix.shape[0]):
        for c in range(diff_matrix.shape[1]):
            cell_text = str(round(diff_matrix[r, c], 2))
            # r, c --> y, x in matplotlib coord frame.
            plt.text(
                c, r, cell_text, va='center', ha='center',
                color='paleturquoise', fontsize=0.8 * fontsize)

    if (dom1_lang_lists_by_task is not None
            and dom2_lang_lists_by_task is not None):
        # draw boxes around each task
        x, y = 0, 0
        for i, task_lang_list2 in enumerate(dom2_lang_lists_by_task):
            xlen = len(task_lang_list2)
            for j, task_lang_list1 in enumerate(dom1_lang_lists_by_task):
                ylen = len(task_lang_list1)
                box_around_cells(x, y, xlen, ylen, color="cyan", linewidth=3)
                y += ylen
            x += xlen

    plt.colorbar(label="Value")
    plt.xlabel("Domain 2 strings")
    plt.ylabel("Domain 1 strings")
    plt.title(title_str)
    plt.savefig(fname, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dom1-img-dir", type=str, required=True)
    parser.add_argument("--dom2-img-dir", type=str, required=True)
    parser.add_argument("--softmax-temp", type=float, default=0.0)
    parser.add_argument(
        "--dist-fn", type=str, choices=["l2", "dotprod"], default="l2")
    args = parser.parse_args()
    # plot(args)
    plot_diff_matrix(
        np.random.rand(56, 56), "20231121.png", ['a'] * 56, ['b'] * 56, "hi",
        dom1_lang_lists_by_task=None, dom2_lang_lists_by_task=None)
