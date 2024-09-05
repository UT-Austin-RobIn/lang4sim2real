import argparse
import glob
import os
from pathlib import Path


def get_ext_fnames(dir_path, ext):
    fnames = []
    for fname in glob.glob(os.path.join(dir_path, f"*.{ext}")):
        fnames.append(Path(fname).stem)
    return fnames


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', required=True)
    args = parser.parse_args()

    # move raw_images --> images
    old_im_dir = os.path.join(args.dir, "raw_images")
    new_im_dir = os.path.join(args.dir, "images")
    if os.path.exists(old_im_dir):
        os.rename(old_im_dir, new_im_dir)

    # read which ids are train and val images
    label_dir = os.path.join(args.dir, "labels")
    train_label_dir = os.path.join(label_dir, "train")
    val_label_dir = os.path.join(label_dir, "val")
    train_fnames = get_ext_fnames(train_label_dir, "txt")
    val_fnames = get_ext_fnames(val_label_dir, "txt")
    print("len(train_fnames)", len(train_fnames))
    print("len(val_fnames)", len(val_fnames))

    # Make train and val folders
    train_im_dir = os.path.join(new_im_dir, "train")
    val_im_dir = os.path.join(new_im_dir, "val")
    os.makedirs(train_im_dir, exist_ok=True)
    os.makedirs(val_im_dir, exist_ok=True)

    # Rename the images to same as txt files and Move into train/val folder
    im_dirs = glob.glob(f"{new_im_dir}/*")
    im_dirs.remove(train_im_dir)
    im_dirs.remove(val_im_dir)
    print("im_dirs", im_dirs)
    for im_batch_dir_path in im_dirs:
        print("im_batch_dir_path", im_batch_dir_path)
        im_batch_dir_stem = Path(im_batch_dir_path).stem
        for im_batch_fpath in glob.glob(
                os.path.join(im_batch_dir_path, "*.png")):
            im_stem = Path(im_batch_fpath).stem
            new_im_stem = f"{im_batch_dir_stem}_{im_stem}"

            if new_im_stem in train_fnames:
                split_dir = train_im_dir
            elif new_im_stem in val_fnames:
                split_dir = val_im_dir
            else:
                print("new_im_stem", new_im_stem)
                raise NotImplementedError

            new_im_fpath = os.path.join(split_dir, f"{new_im_stem}.png")
            mv_cmd = f"mv {im_batch_fpath} {new_im_fpath}"
            # print(mv_cmd)
            os.system(mv_cmd)

        # Remove old folder
        os.system(f"rm -r {im_batch_dir_path}")

    print(train_im_dir)
    print(val_im_dir)
