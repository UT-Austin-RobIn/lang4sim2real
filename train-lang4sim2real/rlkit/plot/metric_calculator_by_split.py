import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from deepdiff.diff import DeepDiff
from rlkit.lang4sim2real_utils.train.train_policy_cnn_lang4sim2real import (
    get_timestamp_str)
from rlkit.plot.plot_utils import (
    get_all_csvs_under, downsample)


def flatten_dict(d):
    flat_params = dict()
    for k, v in d.items():
        if isinstance(v, dict):
            v = flatten_dict(v)
            for subk, subv in flatten_dict(v).items():
                flat_params[k + "." + subk] = subv
        else:
            flat_params[k] = v
    return flat_params


class ExpInfo:
    def __init__(self, csv_file):
        self.exp_dir = os.path.dirname(csv_file)
        self.csv_file = csv_file
        if self.num_csv_file_lines() == 0:
            self.df = None
        else:
            self.df = downsample(pd.read_csv(self.csv_file), freq=10)
        self.json = os.path.join(self.exp_dir, "variant.json")
        with open(self.json, 'r') as f:
            self.params = flatten_dict(json.load(f))

    def num_csv_file_lines(self):
        with open(self.csv_file) as f:
            return len(f.readlines())


class MetricCalculator:
    def __init__(
            self,
            exp_path_pattern,
            args,
            csv_key="eval/env_infos/final/reward Mean"):
        self.exp_dir_to_expinfo_map = self.get_exp_info(
            exp_path_pattern)
        self.csv_key = csv_key
        self.min_epoch_cutoff = args.min_epoch_cutoff // 10
        self.max_epoch_cutoff = 600 // 10
        self.min_num_seeds = 2

    def get_exp_info(self, paths):
        exp_dir_to_expinfo_map = {}
        for path in paths:
            csv_files = get_all_csvs_under(path)
            for csv_file in csv_files:
                expinfo = ExpInfo(csv_file)
                if expinfo.df is None:
                    continue
                exp_dir_to_expinfo_map[expinfo.exp_dir] = expinfo
        return exp_dir_to_expinfo_map

    def extract_split_keys(self):
        def strip_diff_key(s):
            s = s.replace("root['", "").replace("']", "")
            if "[" in s:
                s = s[:s.index("[")]
            return s

        allpairs = itertools.product(
            self.exp_dir_to_expinfo_map.keys(),
            self.exp_dir_to_expinfo_map.keys())

        split_keys = set()
        for exp1, exp2 in tqdm(allpairs):
            exp1_params = self.exp_dir_to_expinfo_map[exp1].params
            exp2_params = self.exp_dir_to_expinfo_map[exp2].params
            diff = DeepDiff(exp1_params, exp2_params, report_repetition=True)
            if exp1_params != exp2_params:
                diff_keys = [
                    strip_diff_key(s)
                    for s in diff['values_changed'].keys()]
                diff_keys = set(diff_keys) - set(
                    ['exp_name', 'seed', 'trial_name', 'unique_id'])
                split_keys.update(diff_keys)
        return list(split_keys)

    def split_by_param(self, split_keys):
        assert isinstance(split_keys, list)
        split_val_to_dfs_map = {}
        for exp_dir, expinfo in self.exp_dir_to_expinfo_map.items():
            # Get all values for split_keys
            split_keyvals = []
            for split_key in split_keys:
                split_val = (
                    "None" if split_key not in expinfo.params
                    else expinfo.params[split_key])
                split_keyvals.append((split_key, split_val))
            out_map_key = str(tuple(split_keyvals))
            if out_map_key in split_val_to_dfs_map:
                split_val_to_dfs_map[out_map_key].append(expinfo.df)
            else:
                split_val_to_dfs_map[out_map_key] = [expinfo.df]

        df_dict = {
             "Mean: " + self.csv_key: [],
             "Std: " + self.csv_key: [],
             "epochs": [],
        }
        for out_map_key, dfs in split_val_to_dfs_map.items():
            df_lens = [len(df) for df in dfs]
            if len(dfs) < self.min_num_seeds:
                continue
            furthest_calculable_epoch = min(
                sorted(df_lens)[-self.min_num_seeds], self.max_epoch_cutoff)
            if furthest_calculable_epoch < self.min_epoch_cutoff:
                continue
            split_setting_tuple = eval(out_map_key)
            for split_key, split_val in split_setting_tuple:
                if split_key not in df_dict:
                    df_dict[split_key] = [split_val]
                else:
                    df_dict[split_key].append(split_val)
            metric_cols = pd.concat(
                [pd.DataFrame(data=df[self.csv_key][
                    self.min_epoch_cutoff:furthest_calculable_epoch])
                    for df in dfs], axis=1)
            means_over_dfs = np.mean(metric_cols, axis=1).to_numpy()
            stds_over_dfs = np.std(metric_cols, axis=1).to_numpy()
            df_dict["Mean: " + self.csv_key].append(np.mean(means_over_dfs))
            df_dict["Std: " + self.csv_key].append(np.mean(stds_over_dfs))
            df_dict["epochs"].append([
                x * 10 for x in sorted(df_lens) if x >= self.min_epoch_cutoff])
        df = pd.DataFrame.from_dict(df_dict)
        df.to_csv(f'eval_metrics_{get_timestamp_str()}.csv')
        pd.set_option('display.max_colwidth', None)
        print(df)
        return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-epoch-cutoff", type=int, default=300)
    parser.add_argument("exp_paths", nargs="*", type=str)
    args = parser.parse_args(sys.argv[1:])

    mc = MetricCalculator(args.exp_paths, args)
    # split_keys = mc.extract_split_keys()
    split_keys = ['eval_task_indices', 'policy_cnn_ckpt']
    split_val_to_dfs_map = mc.split_by_param(split_keys)
