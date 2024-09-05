from functools import partial
from matplotlib import patches, animation
import matplotlib.pyplot as plt
import numpy as np

from deoxys.envs import init_env
from deoxys.utils.obj_detector import ObjectDetectorDL


def plot_img_with_bboxes(i):
    img = np.random.randint(0, 255, size=(480, 640), dtype=np.uint8)
    rand_x1 = np.random.randint(0, 100)
    rand_x2 = np.random.randint(200, 300)
    rand_y1 = np.random.randint(0, 100)
    rand_y2 = np.random.randint(200, 300)
    bboxes = {
        "hi": (rand_x1, rand_y1, rand_x2, rand_y2),
        "lo": (rand_y1, rand_x1, rand_y2, rand_x2)}

    plt.cla()  # clear axes
    colors = ['r', 'c', 'k', 'g', 'w', 'y', 'm']
    for i, (obj_name, obj_bbox) in enumerate(bboxes.items()):
        x_min, y_min, x_max, y_max = obj_bbox
        w, h = x_max - x_min, y_max - y_min
        rect = patches.Rectangle(
            (x_min, y_min), w, h, linewidth=4, edgecolor=colors[i],
            facecolor='none')
        plt.gca().add_patch(rect)
        plt.text(
            x_min, y_min - 10, obj_name,
            fontsize=14, color=colors[i], fontweight="bold")
    plt.imshow(img)


def dummy_animation():
    ani = animation.FuncAnimation(
        plt.gcf(), plot_img_with_bboxes, interval=250)
    plt.show()


def obj_detector_animation():
    env = init_env("frka_pp")
    obj_detector = ObjectDetectorDL(env=env)

    def dummy(i, obj_detector):
        # img, bboxes = obj_detector.plot_img_with_bboxes()
        # img = obj_detector.get_img(transpose=False)
        img = np.random.randint(0, 255, size=(480, 640), dtype=np.uint8)
        plt.imshow(img)

    ani = animation.FuncAnimation(
        plt.gcf(), partial(dummy, obj_detector=obj_detector), interval=250)
    plt.show()


if __name__ == "__main__":
    # dummy_animation()
    obj_detector_animation()
