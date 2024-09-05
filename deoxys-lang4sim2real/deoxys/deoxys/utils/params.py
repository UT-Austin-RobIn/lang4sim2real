import numpy as np

DL_RGB_TO_ROBOT_TRANSMATRIX = np.array([[  -0.028045,    -0.38571],
 [ 0.00020127,   0.0011857],
 [  0.0024057,  0.00057728],
 [-1.8339e-07,  1.2393e-07],
 [-1.4664e-07, -1.5693e-06],
 [-2.2277e-06, -1.7513e-07],])

AT_RGB_TO_ROBOT_TRANSMATRIX = np.array([[ 1.07367090e-01, -7.84324089e-01],
 [-1.57833965e-05,  1.51060072e-03],
 [ 1.45368937e-03,  4.64404649e-05],
 [ 5.52243539e-08, -9.45062427e-08],
 [ 3.67685636e-08,  4.64931828e-08],
 [-3.01471167e-08, -1.22177399e-07],]
)

DL_OBJECT_DETECTOR_CHECKPOINT = '/home/robin/Projects/albert/yolov5/yolov5/runs/train/exp22/weights/best.pt'

OBJECT_DETECTOR_CLASSES = [
    "eu_white_plug",
    "3d_printer_spool",
    "bowl",
    "plate",
    "us_white_plug",
    "clear_container",
    "rb_ckbd",
    "blender",
    "velcro_doubleloop",
    "plug",
    "trash bin",
    "trash bin lid",
    "paper ball",
    "plastic_jar",
    "paper_rect_box",
    "green_rect_block",
    "carrot",
    "bridge",
    "green_cylinder",
]  # See data_config.yaml

OBJ_TO_PICK_PT_Z_MAP = {
    "paper_rect_box": 0.06,
    "green_cylinder": 0.04,
    "carrot": 0.04,
    "bowl": 0.07,
    "clear_container": 0.05,
    "bridge": 0.04,
    "plug": 0.03,
    "eu_white_plug": 0.035,
    "velcro_doubleloop": 0.04,
}

OBJ_TO_DROP_PT_Z_MAP = {
    "paper_rect_box": 0.1,
    "carrot": 0.09,
}

OBJ_TO_PICK_XYZ_OFFSET = {
    "carrot": np.array([0., 0., 0.]),
    "paper_rect_box": np.array([0., 0., 0.]),
    "bowl": np.array([0.0, -0.05, 0.]),
    "clear_container": np.array([0.0, -0.08, 0.]),
    # ^ when doing bridge: "clear_container": np.array([0.0, -0.1, 0.]),
    "bridge": np.array([0.07, 0.01, 0.]),  # np.array([0., -0.01, 0.]),
    "velcro_doubleloop": np.array([0.03, 0., 0.]),
    "plug": np.array([0.03, 0., 0.]),
    "eu_white_plug": np.array([0.07, 0., 0.]),
}

OBJ_TO_GRASP_AC_MAP = {
    "bridge": -0.5,  # -0.53,
    "paper_rect_box": -.35,  # -0.3,
    "plastic_jar": -0.8,
    "carrot": -0.2,  # -0.53,  # -0.33
    "bowl": -0.0,
    "clear_container": -0.0,
    "velcro_doubleloop": -0.0,
    "plug": -0.2,
    "eu_white_plug": -0.1,
}

CONT_XY_LIMITS = {
    "lo": np.array([-np.inf, -0.01]),
    "hi": np.array([0.42, np.inf]),
}
