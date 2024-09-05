import time

import cv2

from deoxys.utils.apriltag_utils import AprilTagDetector
from rpl_vision_utils.networking.camera_redis_interface import (
    CameraRedisSubInterface)

cr_interface = CameraRedisSubInterface(redis_host="172.16.0.1", camera_id=0)
cr_interface.start()

atd = AprilTagDetector()

while True:
    im_info = cr_interface.get_img()
    # im = Image.fromarray(im_info["color"])
    # im.save("20230521.png")
    img = cv2.cvtColor(im_info["color"], cv2.COLOR_BGR2GRAY)
    try:
        theta = atd.get_tag_rotation(img)
        print("theta", theta)
        pos = atd.get_robot_xy_of_tag(img)
        print("pos", pos)
    except:
        print("not getting tag")
    time.sleep(1)

print("tags", tags)
