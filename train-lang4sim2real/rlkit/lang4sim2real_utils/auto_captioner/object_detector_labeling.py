import argparse
import os

import cv2
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, predict, annotate
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
import torch

from labeller_utils import TrajLabellerDataset, save_confusion_matrix


class GroundedDINOWrapper:
    def __init__(self, target_obj_name, gdino_path):
        py_path = os.path.join(
            gdino_path,
            "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
        pth_path = os.path.join(
            gdino_path,
            "GroundingDINO/weights/groundingdino_swint_ogc.pth")
        self.gdino_model = load_model(py_path, pth_path)
        self.text_prompt = f"{target_obj_name} . plate . clear container ."
        self.box_thresh = 0.3
        self.text_thresh = 0.2

    def get_bboxes(self, image):
        """Takes most likely box per class"""
        _, image = self.preprocess_img(image.squeeze().numpy())
        boxes, logits, phrases = predict(
            model=self.gdino_model,
            image=image,
            caption=self.text_prompt,
            box_threshold=self.box_thresh,
            text_threshold=self.text_thresh
        )
        phrase_set = set(phrases)
        final_boxes = torch.zeros((len(phrase_set), 4))
        final_phrases = []
        for num, phrase in enumerate(phrase_set):
            indices = [i for i, p in enumerate(phrases) if p == phrase]
            max = logits[indices[0]]
            index = indices[0]
            for i in indices:
                if logits[i] > max:
                    max = logits[i]
                    index = i
            final_boxes[num] = boxes[index]
            final_phrases.append(phrases[index])
        final_logits = logits[:len(final_boxes)]

        obj_name_to_bbox_dict = {
            "clear container": None, "plate": None, "carrot": None,
            "wooden block": None}
        for phrase in final_phrases:
            obj_name_to_bbox_dict[phrase] = final_boxes[
                final_phrases.index(phrase)]
        obj_name_to_bbox_dict["final_boxes"] = final_boxes

        return obj_name_to_bbox_dict, final_logits, final_phrases

    def annotate_frame_bboxes(
            self, image_source, boxes, logits, phrases, num=0):
        annotation = annotate(
            image_source=image_source, boxes=boxes, logits=logits,
            phrases=phrases)
        cv2.imwrite(os.path.join("./output", f"x{num}img.png"), annotation)
        print(f"Saved to {out_path}")
        return annotation

    def preprocess_img(self, numpy_image):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image = Image.fromarray(numpy_image.astype('uint8'), 'RGB')
        
        image_transformed, _ = transform(image, None)
        return numpy_image, image_transformed


class StageLabeller:
    def __init__(self, path, num_substeps, args):
        self.target_obj_name = ["carrot", "wooden block"][0]
        self.args = args
        self.gdino = GroundedDINOWrapper(self.target_obj_name, args.gdino_path)
        self.gripper_predictor = self.load_gripper_model()
        self.gripper_predictor.eval()
        self.path = path
        assert num_substeps in [1, 10]
        self.num_substeps = num_substeps
        hdf5_kwargs = dict(
            max_demos_per_task=np.inf,
            task_indices=[0],
        )
        self.ds = TrajLabellerDataset(self.path, hdf5_kwargs)

    def load_gripper_model(self):
        return torch.load(self.args.gripper_state_pred_model).to("cuda:0")

    def get_gripper_traj_pred(self, traj_img):
        """
        traj_img: (T, 3, 128, 128)
        """
        with torch.no_grad():
            gripper_traj_pred = self.gripper_predictor(traj_img.to("cuda:0"))
        ee_pred = gripper_traj_pred[0]  # (T, 3)
        gripper_open_pred = torch.argmax(
            gripper_traj_pred[1], dim=-1) == 0  # bool(T,)
        return ee_pred, gripper_open_pred

    def bbox_to_polygon(self, bbox):
        x_min, y_min, x_len, y_len = bbox
        x_max = x_min + x_len
        y_max = y_min + y_len
        polygon = [
            (x_min, y_min),
            (x_min, y_max),
            (x_max, y_max),
            (x_max, y_min),
            (x_min, y_min)
        ]
        return Polygon(polygon)

    def annotate_from_hdf5(self):
        ds_loader = torch.utils.data.DataLoader(
            self.ds, batch_size=1, shuffle=False)
        pred_stages = []
        num_correct = 0
        correct_stages = []
        task_demo_id_to_pred_stages_map = {}
        total = 0
        for batch_idx, (
                torch_img,
                img_uint8,
                gs,
                stage_nums,
                task_demo_id) in enumerate(ds_loader):
            torch_img = torch_img.squeeze()
            img_uint8 = img_uint8.squeeze()
            traj_ee_pred, traj_gripper_open_pred = (
                self.get_gripper_traj_pred(torch_img))
            print(f"\nLabeling trajectory {batch_idx}...")
            curr_stage = 0
            traj_pred_stages = []
            for t in range(0, torch_img.shape[0], self.num_substeps):
                obj_name_to_bbox_dict, logits, phrases = (
                    self.gdino.get_bboxes(img_uint8[t]))

                ahead2 = 2 * self.num_substeps
                ahead1 = self.num_substeps
                future_gripper_open_pred = (
                    traj_gripper_open_pred[t + ahead2]
                    if t + ahead2 < torch_img.shape[0]
                    else True)
                next_ee_pred = (
                    traj_ee_pred[t + ahead1]
                    if t + ahead1 < torch_img.shape[0]
                    else None)
                curr_stage = self.predict_stage(
                    obj_name_to_bbox_dict,
                    traj_gripper_open_pred[t],
                    future_gripper_open_pred,
                    traj_ee_pred[t],
                    next_ee_pred,
                    curr_stage)
                print(
                    f"real stage: {stage_nums[0][t]}, "
                    f"pred stage: {curr_stage}")
                num_correct += (curr_stage == stage_nums[0][t])
                total += 1
                print(f"accuracy {num_correct / total}")
                traj_pred_stages.append(curr_stage)
                correct_stages.append(stage_nums[0][t].detach().item())

            pred_stages.extend(traj_pred_stages)
            pred_stages_maybe_upsampled = self.maybe_upsample_pred_stages(
                traj_pred_stages)
            task_demo_id_to_pred_stages_map[task_demo_id[0]] = (
                pred_stages_maybe_upsampled)
        save_confusion_matrix(correct_stages, pred_stages)

        return task_demo_id_to_pred_stages_map

    def maybe_upsample_pred_stages(self, pred_stages):
        # upsample pred_stages if needed
        if self.num_substeps > 1:
            pred_stages_upsampled = []
            for pred_stage in pred_stages:
                pred_stages_upsampled.extend([pred_stage] * 10)
            pred_stages = pred_stages_upsampled[self.num_substeps - 1:]
        return pred_stages

    def predict_stage(
            self,
            obj_name_to_bbox_dict,
            gripper_open_pred,
            future_gripper_open_pred,
            ee_pred,
            next_ee_pred,
            curr_stage):
        def obj_in_cont(obj_bbox, cont_bbox, thresh=.7):
            obj_poly = self.bbox_to_polygon(obj_bbox)
            cont_poly = self.bbox_to_polygon(cont_bbox)
            intersection = obj_poly.intersection(cont_poly).area
            obj_area = obj_poly.area
            return (intersection / obj_area) > thresh

        def obj_vertical_with_cont(obj_bbox, cont_bbox):
            obj_bbox[1] = cont_bbox[1]  # make vertical y values same
            obj_bbox[3] = cont_bbox[3]
            obj_poly = self.bbox_to_polygon(obj_bbox)
            cont_poly = self.bbox_to_polygon(cont_bbox)
            intersection = obj_poly.intersection(cont_poly).area
            obj_area = obj_poly.area
            return (intersection / obj_area) > 0.95

        def vertical_with_side_of_cont(obj_bbox, cont_bbox):
            return (
                (obj_bbox[0] < cont_bbox[0])
                and (obj_bbox[0] + obj_bbox[2] > cont_bbox[0]))
        
        def gripper_moving_horizontal(ee_pred, next_ee_pred):
            if next_ee_pred is None:
                return False
            delta = (next_ee_pred - ee_pred).detach().cpu().numpy()
            if ((np.linalg.norm(delta[:2]) > 0.03)
                    and (np.linalg.norm(delta[:2]) > delta[2])):
                return True
            return False
        
        def gripper_above_thresh(ee_pred):
            """
            Tune this threshold.
            .18 gets slightly higher overall accuracy but mostly because
            it favors common classes
            """
            return ee_pred[2] > 0.21

        def gripper_below_thresh(ee_pred):
            return ee_pred[2] < 0.14

        def gripper_moving_down(ee_pred, next_ee_pred):
            if next_ee_pred[2] - ee_pred[2] < -.015:
                return True
            return False
        
        def get_pp_stage_within_step(
                cur_obj_box,
                cur_cont_box,
                cur_gripper_open_pred,
                future_gripper_open_pred,
                cur_ee_pred,
                next_ee_pred=None,
                cur_stage=0,
                step_idx=0):
            # Returns a number from 0-6 for the stage idx
            if step_idx == 2:
                print("here")
                return -1
            if cur_cont_box is None or cur_obj_box is None:
                return cur_stage
            if ((cur_stage in [0, 1] and cur_gripper_open_pred)
                    or future_gripper_open_pred):
                # gripper is open so 0 1 or 6
                if cur_stage >= 5 or obj_in_cont(cur_obj_box, cur_cont_box):
                    print(cur_stage)
                    return 6
                elif not gripper_moving_horizontal(cur_ee_pred, next_ee_pred):
                    # moving down or still
                    return 1
                else:
                    # moving horizontal
                    return 0
            else:
                # gripper is closed so 2, 3, 4, 5
                if step_idx == 0 and obj_vertical_with_cont(
                        cur_obj_box, cur_cont_box):
                    return 5
                else:
                    if gripper_above_thresh(cur_ee_pred):
                        return 4
                    elif gripper_below_thresh(cur_ee_pred):
                        return 2
                    else:
                        return 3

        print(f"gripper open {gripper_open_pred}, ee_pos pred {ee_pred}")

        # Return curr stage if can't find objs
        if "clear container" not in obj_name_to_bbox_dict:
            return curr_stage

        target_obj_bbox = obj_name_to_bbox_dict[self.target_obj_name]
        clear_cont_bbox = obj_name_to_bbox_dict["clear container"]
        plate_bbox = obj_name_to_bbox_dict["plate"]

        if curr_stage < 7:
            if target_obj_bbox is None or clear_cont_bbox is None:
                print(
                    f"missing obj or cont bbox, target obj bbox "
                    f"{target_obj_bbox}, clear cont bbox {clear_cont_bbox}")
                step_idx = 0
            elif ((obj_in_cont(target_obj_bbox, clear_cont_bbox, thresh=0.5))
                    and gripper_moving_horizontal(ee_pred, next_ee_pred)):
                print("idx 1")
                step_idx = 1
            else:
                print(
                    "idx 0",
                    obj_in_cont(target_obj_bbox, clear_cont_bbox),
                    gripper_moving_horizontal(ee_pred, next_ee_pred))
                step_idx = 0
        elif curr_stage < 13:
            if plate_bbox is None or clear_cont_bbox is None:
                print(
                    f"missing plate or cont bbox, plate obj bbox "
                    f"{plate_bbox}, clear cont bbox {clear_cont_bbox}")
                step_idx = 1
            elif obj_in_cont(clear_cont_bbox, plate_bbox, thresh=0.5):
                step_idx = 2
            else:
                step_idx = 1
        else: 
            step_idx = 2

        if step_idx == 0:
            cur_obj_box = obj_name_to_bbox_dict[self.target_obj_name]
            cur_cont_box = obj_name_to_bbox_dict["clear container"]
        else:
            cur_obj_box = obj_name_to_bbox_dict["clear container"]
            cur_cont_box = obj_name_to_bbox_dict["plate"]
        curr_stage = 7 if curr_stage < 7 and step_idx == 1 else curr_stage

        pp_stage_within_step_idx = get_pp_stage_within_step(
            cur_cont_box=cur_cont_box,
            cur_obj_box=cur_obj_box, 
            cur_gripper_open_pred=gripper_open_pred,
            future_gripper_open_pred=future_gripper_open_pred,
            cur_ee_pred=ee_pred, 
            next_ee_pred=next_ee_pred,
            cur_stage=(curr_stage % 7),
            step_idx=step_idx)
        pred_stage = 7 * step_idx + pp_stage_within_step_idx
        return pred_stage

def get_bbox_annotate_img(img_path):
    from PIL import Image
    target_obj_name = "carrot"
    gdino = GroundedDINOWrapper(target_obj_name)
    # img_uint8 = np.uint8(Image.open(img_path))
    img_uint8 = cv2.imread(img_path)
    img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2RGB)
    new_h = int(img_uint8.shape[1] // 2.5)
    new_w = int(img_uint8.shape[0] // 2.5)
    img_uint8 = cv2.resize(img_uint8, dsize=(new_h, new_w), interpolation=cv2.INTER_CUBIC)
    obj_name_to_bbox_dict, logits, phrases = (
        gdino.get_bboxes(torch.tensor(img_uint8)))

    annotated_frame = gdino.annotate_frame_bboxes(
        img_uint8,
        obj_name_to_bbox_dict["final_boxes"], logits, phrases, num=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gdino-path", type=str, default="/home/albert/dev/")
    parser.add_argument("--buffer-path", type=str, required=True)
    parser.add_argument("--gripper-state-pred-model", type=str, required=True)
    args = parser.parse_args()

    import ipdb; ipdb.set_trace()
    num_substeps = 10
    stage_labeller = StageLabeller(args.buffer_path, num_substeps, args)
    preds = stage_labeller.annotate_from_hdf5()
    input("Are you sure you want to write changes to the buffer? "
          "CTRL+C to quit.")
    stage_labeller.ds.write_predicted_stage_nums(args.buffer_path, preds)
