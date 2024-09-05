import argparse
import numpy as np
import os

# Used to concatenate .mp4's with format epoch{epoch_num}_{rollout_num}.mp4
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video-path", type=str, default="")
    parser.add_argument("-e", "--earliest-epoch", type=int, default=None)
    parser.add_argument("-n", "--num-videos", type=int, default=None)
    args = parser.parse_args()

    video_path = args.video_path
    output_video_name = os.path.dirname(video_path).split("/")[-1]

    lines = []
    for root, dirs, files in os.walk(video_path, topdown=False):
        for name in files:
            name_wo_ext, ext = name.split(".")
            if ext == "mp4":
                try:
                    epoch_str, rollout_num_str = name_wo_ext.split("_")
                    epoch_num = int(epoch_str.replace("epoch", ""))
                    if epoch_num >= args.earliest_epoch:
                        sort_int = 100 * epoch_num + int(rollout_num_str)
                        lines.append(
                            (sort_int, f"file '{os.path.join(root, name)}'"))
                except:
                    pass

    if args.num_videos is not None and len(lines) > args.num_videos:
        random_lines = []
        all_idxs = np.arange(0, len(lines))
        rand_idxs = np.sort(all_idxs[:args.num_videos])
        for i in rand_idxs:
            random_lines.append(lines[i])
        lines = random_lines
    if len(lines) > 0:
        lines = sorted(lines)
        print(lines)

        video_list_fpath = os.path.join(video_path, "input.txt")
        with open(video_list_fpath, "w") as f:
            for _, line in lines:
                f.write(line + "\n")

        output_mp4_path = os.path.join(video_path, f"{output_video_name}.mp4")
        os.system(
            f"ffmpeg -f concat -safe 0 -i {video_list_fpath} "
            f"-c copy {output_mp4_path}")
        output_gif_path = os.path.join(video_path, f"{output_video_name}.gif")
        os.system(
            f"ffmpeg -i {output_mp4_path} -filter:v fps=20 {output_gif_path}")
