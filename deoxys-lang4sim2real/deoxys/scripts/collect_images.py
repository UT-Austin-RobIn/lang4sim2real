import argparse
import os

import cv2
import numpy as np

from deoxys.envs import init_env


def save_images(args):
    env = init_env("frka_base")

    obs = env.reset()
    count = 0

    print(args.save_directory)
    if not os.path.exists(args.save_directory) and not args.no_save:
        os.mkdir(args.save_directory)

    while True:
        k = input(
            "Press ENTER to take image, ctrl-c to quit, \n"
            "c to close gripper fully, h to close gripper halfway, \n"
            "o to open gripper, r to reset")
        if args.rand_action:
            xy_a = np.random.normal(scale=1.0, size=(2,))
            z_a = np.random.normal(loc=-0.2, scale=0.5, size=(1,))
            gripper_a = np.random.uniform(low=-1.0, high=0.0, size=(1,))
            if k == "c":
                gripper_a = np.array([-0.0])
            if k == "h":
                gripper_a = np.array([-0.5])
            elif k == "o":
                gripper_a = np.array([-1.0])
            a = np.concatenate([xy_a, z_a, np.zeros((2, )), gripper_a])
            print("action", a)
            if k == "r":
                obs = env.reset()
            else:
                obs, *_ = env.step(a)

        print("Getting Image \n")
        obs = env.get_obs()
        image_path = os.path.join(
            args.save_directory, args.prefix + '_{}.png'.format(count))
        if not args.no_save:
            print(image_path)
            cv2.imwrite(
                image_path,
                cv2.cvtColor(obs['image_480x640'], cv2.COLOR_RGB2BGR))
        count += 1


if __name__ == "__main__":
    # python ~/Projects/albert/deoxys_v2/deoxys/scripts/collect_images.py -s /home/robin/Projects/albert/datasets/obj_detection/20240104 --rand-action
    # Upload to supervisely and annotate
    # --mark with train/val annotations
    # --download by converting supervisely to yolov5 format
    # tar -xvf 281001_ahg...
    # Modify data_config.yaml (train and val paths need to be made absolute)
    # python train.py --data ~/Projects/albert/datasets/obj_detection/yolov5_annotated/20231231/data_config.yaml --epochs 100 --weights yolov5s.pt --cfg yolov5s.yaml  --batch-size 64
    # get checkpoint path
    # Ex: /home/robin/Projects/albert/yolov5/runs/train/exp6/weights/best.pt
    # python detect.py --weights /home/robin/Projects/albert/yolov5/yolov5/runs/train/exp6/weights/best.pt --img 640 480 --conf 0.1 --source /home/robin/Projects/albert/datasets/obj_detection/yolov5_annotated/20230318_v3/images/train/ds0_out_0.png    # Update the params.py checkpoint_path
    # Update params.py OBJECT_DETECTOR_CLASSES and DL_OBJECT_DETECTOR_CHECKPOINT with data_config.yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save_directory', default='images')
    parser.add_argument('-p', '--prefix', default='out')
    parser.add_argument("--rand-action", action="store_true", default=False)
    parser.add_argument("--no-save", action="store_true", default=False)
    args = parser.parse_args()
    # import ipdb; ipdb.set_trace()

    if args.no_save:
        assert args.rand_action

    save_images(args)
