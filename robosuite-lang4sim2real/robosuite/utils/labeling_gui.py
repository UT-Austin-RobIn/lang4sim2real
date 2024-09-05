import os
import pygame as pg
import numpy as np
import pickle

pg.init()
COLOR_INACTIVE = pg.Color('lightskyblue3')
COLOR_ACTIVE = pg.Color('dodgerblue2')
FONT = pg.font.Font(None, 32)


class InputBox:
    def __init__(self, x, y, w, h, text=''):
        self.rect = pg.Rect(x, y, w, h)
        self.w = w
        self.color = COLOR_INACTIVE
        self.text = self.render_text(text)
        self.active = False

    def handle_event(self, event):
        if event.type == pg.MOUSEBUTTONDOWN:
            # If the user clicked on the input_box rect.
            if self.rect.collidepoint(event.pos):
                # Toggle the active variable.
                self.active = not self.active
            else:
                self.active = False
            # Change the current color of the input box.
            self.color = COLOR_ACTIVE if self.active else COLOR_INACTIVE
        if event.type == pg.KEYDOWN:
            if self.active:
                if event.key == pg.K_RETURN:
                    # print(self.text)
                    return self.text
                    self.text = ""
                elif event.key == pg.K_BACKSPACE:
                    self.text = self.text[:-1]
                elif event.unicode == "c":
                    # Clear the text
                    self.text = ""
                else:
                    self.text += event.unicode
                # Re-render the text.
                self.txt_surface = FONT.render(self.text, True, self.color)

    def update(self):
        # Resize the box if the text is too long.
        width = max(self.w, self.txt_surface.get_width()+10)
        self.rect.w = width

    def render_text(self, text):
        self.text = text
        self.txt_surface = FONT.render(text, True, self.color)
        return self.text

    def draw(self, screen):
        # Blit the text.
        screen.blit(self.txt_surface, (self.rect.x+5, self.rect.y+5))
        # Blit the rect.
        pg.draw.rect(screen, self.color, self.rect, 2)


def get_image_paths_to_label(dir_path):
    image_paths = []

    _, _, fnames = list(os.walk(dir_path))[0]
    for fname in fnames:
        if fname[-4:] == ".png":
            image_path = os.path.join(dir_path, fname)
            image_paths.append(image_path)
    image_paths = sorted(image_paths)
    return image_paths


def wait_for_user_label_and_render_input(
        screen, clock, image, input_boxes, disp_img_w, disp_img_h):
    done_typing = False
    while not done_typing:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                done_typing = True
            for box in input_boxes:
                input_text = box.handle_event(event)
                if input_text is not None:
                    return input_text

        render(screen, clock, image, input_boxes, disp_img_w, disp_img_h)


def render(screen, clock, image, input_boxes, disp_img_w, disp_img_h):
    for box in input_boxes:
        box.update()

    screen.fill((30, 30, 30))
    for box in input_boxes:
        box.draw(screen)

    image = pg.transform.scale(image, (disp_img_w, disp_img_h))
    screen.blit(image, (0, 0))

    pg.display.flip()
    clock.tick(30)


def main(screen, dir_path, fout_path):
    clock = pg.time.Clock()

    # Get list of images
    image_paths = get_image_paths_to_label(dir_path)
    assert len(image_paths) > 0

    # Text box init
    im = pg.image.load(image_paths[0])
    img_w, img_h = im.get_width(),  im.get_height()
    img_multiplier = 4
    disp_img_w, disp_img_h = img_multiplier * img_w, img_multiplier * img_h

    text_box_h = 32
    text_box_w = disp_img_w
    input_box1 = InputBox(0, disp_img_h, text_box_w, text_box_h)
    input_boxes = [input_box1]

    # Datastructures for saving to disk
    im_arr_dataset = None
    labels_dataset = []

    for image_path in image_paths:
        print("image_path", image_path)
        im = pg.image.load(image_path)
        input_text = wait_for_user_label_and_render_input(
            screen, clock, im, input_boxes, disp_img_w, disp_img_h)
        print("input_text", input_text)

        # SAVE the dataset.
        im_arr = pg.surfarray.pixels3d(im)
        # ^ converts pg surf to np array (uint8)
        if im_arr_dataset is None:
            im_arr_dataset = np.expand_dims(im_arr, axis=0)
            # Make sure the im_arr is all uint8.
        else:
            im_arr = np.expand_dims(im_arr, axis=0)
            im_arr_dataset = np.concatenate((im_arr_dataset, im_arr), axis=0)
        labels_dataset.append(input_text)

        # After each label, save it.
        print("im_arr_dataset.shape", im_arr_dataset.shape)
        assert im_arr_dataset.shape[0] == len(labels_dataset)
        dataset = (im_arr_dataset, labels_dataset)
        with open(fout_path, "wb") as fout:
            pickle.dump(dataset, fout, protocol=4)


if __name__ == '__main__':
    screen = pg.display.set_mode((640, 480))
    dir_path = "/home/albert/scratch/20210905/"
    fout_path = "/home/albert/scratch/20210905/dataset.pkl"
    main(screen, dir_path, fout_path)
    pg.quit()
