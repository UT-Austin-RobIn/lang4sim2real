import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image
import pygame as pg

from robosuite.utils.labeling_gui import InputBox, render

keypress_to_skip_amt_map = {
    'v': -10,
    'b': -1,
    'n': 1,
    'm': 10,
}


def put_arr(surface, myarr):
    bv = surface.get_buffer()
    bv.write(myarr.tostring(), 0)


def wait_for_keypress(disp):
    while True:
        # import ipdb; ipdb.set_trace()
        events = pg.event.get()
        # print("events", events)
        for event in events:
            if (event.type == pg.KEYDOWN
                    and event.unicode in keypress_to_skip_amt_map.keys()):
                return event.unicode


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    args = parser.parse_args()
    pg.init()
    screen = pg.display.set_mode((640, 640))
    clock = pg.time.Clock()

    labels_path = os.path.join(args.dir, "labels.csv")
    image_dir = args.dir

    img_w, img_h, _ = (256, 256, 3)
    img_multiplier = 2
    disp_img_w, disp_img_h = img_multiplier * img_w, img_multiplier * img_h

    text_box_h = 32
    text_box_w = disp_img_w
    input_box1 = InputBox(0, disp_img_h, text_box_w, text_box_h)
    input_boxes = [input_box1]

    labels_df = pd.read_csv(labels_path)
    dataset_len = len(labels_df)

    i = 0
    while i < dataset_len:
        img_fname = labels_df.iloc[i]["img_fname"]
        label = labels_df.iloc[i]["lang"]
        im = Image.open(os.path.join(image_dir, img_fname))
        # import ipdb; ipdb.set_trace()
        im_transposed = np.transpose(im, (1, 0, 2))
        im = pg.surfarray.make_surface(im_transposed)
        # print("im", im)
        # pg.surfarray.blit_array(screen, im)
        # put_arr(screen, im)
        # screen.blit(im, (0,0))
        input_box1.render_text(label)
        print(label)

        render(screen, clock, im, input_boxes, disp_img_w, disp_img_h)

        key = wait_for_keypress(screen)
        skip_amt = keypress_to_skip_amt_map[key]
        i += skip_amt
