import argparse
from datetime import datetime
import os
import warnings

import numpy as np
from PIL import Image


def horiz_center_crop(img_np):
    """Assumes horizontal axis is the wider one"""
    half_min_dim = int(min(img_np.shape[:2]) // 2)
    mid_pt = int(img_np.shape[1] // 2)
    lo_w, hi_w = mid_pt - half_min_dim, mid_pt + half_min_dim
    img_np = img_np[:, lo_w:hi_w]
    return img_np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--start-hhmmss", type=str, help="Ex: 00:00:05")
    parser.add_argument("--end-hhmmss", type=str, help="Ex: 00:00:10")
    parser.add_argument("--num-frames", type=int, required=True)
    parser.add_argument(
        "--out-ext", type=str, choices=["jpg", "png"], default="jpg")
    args = parser.parse_args()

    """
    Example usage:
    python scripts/video_to_filmstrip.py --video .../IMG_1212.MOV --start-hhmmss 00:00:10 --end-hhmmss 00:00:18 --num-frames 20 --out-ext png
    """

    out_dir = os.path.join(
        os.path.dirname(args.video),
        os.path.basename(args.video).split(".")[0])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        warnings.warn("Out directory already exists.")

    date_format = "%H:%M:%S"
    video_clip_num_secs = (
        datetime.strptime(args.end_hhmmss, date_format)
        - datetime.strptime(args.start_hhmmss, date_format)).seconds
    frame_sampling_rate = args.num_frames / video_clip_num_secs

    vid_to_frames_cmd = (
        f'ffmpeg -i {args.video} -ss {args.start_hhmmss} -to {args.end_hhmmss}'
        f' -s 360x640 -q:v 2 -r {frame_sampling_rate} {out_dir}/frame%03d.png')
    print(vid_to_frames_cmd)
    os.system(vid_to_frames_cmd)

    frames = []
    for i in range(1, 1000):
        img_fname = f"frame{str(i).zfill(3)}.png"
        img_path = os.path.join(out_dir, img_fname)
        if not os.path.exists(img_path):
            break
        pil_img = Image.open(img_path)
        img_np = np.asarray(pil_img)

        # Rotate clockwise
        img_np = np.rot90(img_np, axes=(-3, -2))
        modified_img_np = horiz_center_crop(img_np)

        frames.append(modified_img_np)
        im = Image.fromarray(modified_img_np)
        im.save(img_path)

    filmstrip = np.concatenate(frames, axis=1)
    filmstrip_pil = Image.fromarray(filmstrip)
    filmstrip_path = os.path.join(out_dir, f"filmstrip.{args.out_ext}")
    print("filmstrip_path", filmstrip_path)
    filmstrip_pil.save(filmstrip_path)
