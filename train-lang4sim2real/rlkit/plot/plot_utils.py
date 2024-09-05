import os
import pandas as pd


def get_all_csvs_under(path):
    # print(path)
    """path can end with an asterisk"""

    if type(path) == list:
        csv_files = []
        for p in path:
            command = "ls {}/*.csv".format(p)
            csv_file = os.popen(command).read().split("\n")[:-1]
            csv_files.extend(csv_file)
    else:
        command = "ls {}/*.csv".format(path)
        csv_files = os.popen(command).read().split("\n")[:-1]

    return csv_files


def get_dfs_from_paths_under(path, num_epochs_thresh, downsample_freq=1):
    csv_files = get_all_csvs_under(path)
    dframes = []
    for i, csv_file in enumerate(csv_files):
        try:
            csv_data = pd.read_csv(csv_file)
            if len(csv_data) > num_epochs_thresh:
                csv_data = downsample(csv_data, downsample_freq)
                dframes.append(csv_data)
        except:
            # print("csv_file blank", csv_file)
            pass
    if len(dframes) < 3:
        print(f"Warning: Less than 3 dframes under {path}")
    return dframes


def downsample(df, freq):
    """
    For ex: we eval every freq epochs. So drop rows where
    epochs % freq != 0.
    """
    df = df[df['epoch'] % freq == 0]
    return df
