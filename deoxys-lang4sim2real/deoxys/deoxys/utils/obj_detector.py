from deoxys.utils.params import *
from deoxys.utils.control_utils import get_obs_for_calib
from yolov5.utils.general import non_max_suppression

from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
import torch


class ObjectDetector:
    def __init__(
            self, env, save_dir, detector_type='base',
            image_xyz=None, skip_reset=False):
        self.env = env
        self.save_dir = save_dir
        self.detector_type = detector_type
        self.image_xyz = image_xyz
        self.skip_reset = skip_reset


class ObjectDetectorDL(ObjectDetector):
    def __init__(
            self, weights=None, classes=None, env=None,
            save_dir="", image_xyz=None, gpu_id='0',
            use_gpu=True, skip_reset=False):
        super().__init__(
            env, save_dir, image_xyz=image_xyz, detector_type='dl',
            skip_reset=skip_reset)
        self.gpu_id = int(gpu_id)
        self.use_gpu = use_gpu
        self.scaling_factor = 0.01
        if weights is None:
            weights = DL_OBJECT_DETECTOR_CHECKPOINT
        if classes is None:
            classes = OBJECT_DETECTOR_CLASSES

        self.classes = classes
        # load a model pre-trained on COCO
        self.model = torch.hub.load(
            'ultralytics/yolov5', 'yolov5s', pretrained=False,
            classes=len(classes))
        self.model.load_state_dict(torch.load(weights)['model'].state_dict())
        if self.use_gpu:
            self.model.cuda(self.gpu_id)
        self.model.eval()

    def get_bounding_box(self, img, conf_thresh=None):
        img = torch.tensor(img).permute(2, 0, 1).cuda()[None] / 255
        with torch.no_grad():
            pred = self.model(img)
            pred = non_max_suppression(
                pred, conf_thres=0.1, iou_thres=0.45, classes=None,
                agnostic=False, max_det=1000)
            pred = np.array(pred[0].cpu().detach())
        # This is (num_box_proposals, 6)
        # columns are: (min_x, min_y, max_x, max_y, confidence, obj_class)

        # Store boxes by obj_class and confidence scores
        bboxes = dict()
        for box_xyxyco in pred:
            xyxy, conf, obj_class = (
                box_xyxyco[:4], box_xyxyco[4], int(box_xyxyco[5]))
            if obj_class not in bboxes:
                bboxes[obj_class] = dict()
            bboxes[obj_class][tuple(xyxy)] = conf

        if not conf_thresh:
            # Only keep highest confidence box in each obj_class.
            best_in_class_bboxes = dict()
            for obj_class in bboxes:
                obj_name = self.classes[obj_class]
                best_in_class_bboxes[obj_name] = np.array(
                    max(bboxes[obj_class], key=bboxes[obj_class].get))
            bbox_dict = best_in_class_bboxes
        else:
            bbox_dict = dict()
            for obj_class in bboxes:
                obj_name = self.classes[obj_class]
                for xyxy, conf in bboxes[obj_class].items():
                    if conf >= conf_thresh:
                        if obj_name not in bbox_dict:
                            bbox_dict[obj_name] = []
                        bbox_dict[obj_name].append(np.array(xyxy))

        return bbox_dict

    def get_bounding_boxes_batch(self, imgs):
        results = self.model(imgs)
        if self.save_dir != "":
            results.save(self.save_dir)
        ret_list = []
        dfs = results.pandas().xyxy
        for df in dfs:
            best_estimates = df.loc[df.groupby('class')['confidence'].idxmax()]
            bounding_boxes = dict()
            for i in range(len(best_estimates)):
                object_name = self.classes[best_estimates.iloc[i]['class']]
                bounding_boxes[object_name] = self.scaling_factor * np.array([
                    best_estimates.iloc[i]['ymin'],
                    best_estimates.iloc[i]['ymax'],
                    best_estimates.iloc[i]['xmin'],
                    best_estimates.iloc[i]['xmax']])
            ret_list.append(bounding_boxes)

        return ret_list

    @staticmethod
    def centroid_from_bounding_box(box):
        box = np.array(box)
        if len(box.shape) == 1:
            box = box[None]
        return np.concatenate(
            [((box[:, 0] + box[:, 2]) / 2)[:, None],
             ((box[:, 1] + box[:, 3]) / 2)[:, None]], axis=1)

    def get_centroids(self, img, conf_thresh=None):
        bounding_boxes = self.get_bounding_box(img, conf_thresh=conf_thresh)
        centroids = dict()
        for key, box in bounding_boxes.items():
            centroids[key] = self.centroid_from_bounding_box(box)
        return centroids

    def get_centroids_batch(self, imgs):
        bounding_boxes_list = self.get_bounding_boxes_batch(imgs)
        ret = []
        for bounding_boxes in bounding_boxes_list:
            centroids = dict()
            for key, box in bounding_boxes.items():
                centroids[key] = self.centroid_from_bounding_box(box)
            ret.append(centroids)
        return ret

    def get_img(
            self, transpose=True, ret_old_pose=False, lift_before_reset=False,
            wait_secs=1.0):
        obs = get_obs_for_calib(
            self.env, self.image_xyz,
            skip_reset=self.skip_reset, ret_old_pose=ret_old_pose,
            lift_before_reset=lift_before_reset, wait_secs=wait_secs)
        image = obs['image_480x640']
        if transpose:
            image = np.transpose(image, (2, 0, 1))
        return image

    def plot_img_with_bboxes(self):
        img = self.get_img(transpose=False)
        bboxes = self.get_bounding_box(img)

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
        return img, bboxes


if __name__ == "__main__":
    from PIL import Image

    obj_detector = ObjectDetectorDL(env=None)
    img_path = "/home/robin/Projects/albert/datasets/obj_detection/yolov5_annotated/20230318_v3/images/train/ds0_out_0.png"
    img = np.array(Image.open(img_path))
    print(obj_detector.get_centroids(img))
