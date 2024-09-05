import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, Point


class PolygonWrapper:
    """
    A wrapper around shapely polygon that supports things like
    sampling uniformly from the polygon.
    """
    def __init__(self, *args, **kwargs):
        self.polygon = Polygon(*args, **kwargs)
        self.exterior = self.polygon.exterior

    def sample_unif_rand_pt(self, max_num_tries=5000):
        # This is naive rejection sampling from stack overflow.
        # If need a more efficient algorithm for very weird polygons
        # a tiny fraction of their bounding box, use
        # triangulation: https://codereview.stackexchange.com/a/204289
        min_x, min_y, max_x, max_y = self.polygon.bounds

        num_tries = 0
        while num_tries < max_num_tries:
            rand_pt = Point(
                [np.random.uniform(min_x, max_x),
                 np.random.uniform(min_y, max_y)])
            if rand_pt.within(self.polygon):
                return np.array([rand_pt.x, rand_pt.y])

        print(
            f"Could not find a random point in polygon after {max_num_tries}")

    def sample_unif_rand_pts(self, num_pts):
        rand_pts = []
        for i in range(num_pts):
            rand_pt = self.sample_unif_rand_pt()
            rand_pts.append(rand_pt)
        return rand_pts

    def contains(self, xy):
        if not isinstance(xy, Point):
            xy = Point(*xy)
        return self.polygon.contains(xy)

    def show_pt(self, pt, fname):
        x, y = self.polygon.exterior.xy
        plt.clf()
        plt.plot(x, y)
        plt.plot(*pt, 'r.')
        plt.savefig(fname)
        # plt.pause(3)
        # plt.close()


if __name__ == "__main__":
    gripper_z_offset = 0.08
    workspace_xyz_limits = {
        "lo": np.array([0.28, -0.24, 0.025 + gripper_z_offset]),
        "hi": np.array([0.65, 0.22, 0.3 + gripper_z_offset]),
    }

    neg_pad = 0.03
    wrapped_obj_init_limits = {
        True: {  # gripper opened
            "lo": np.array([
                0.3367, -0.1698 + neg_pad, workspace_xyz_limits["lo"][2]]),
            "hi": np.array([
                0.5559, 0.15 - neg_pad, 0.248 - 0.5 * neg_pad])},
        False: {  # gripper closed
            "lo": np.array([
                0.3367 + neg_pad,
                -0.1276 + neg_pad,
                workspace_xyz_limits["lo"][2]]),
            "hi": np.array([
                0.5559 - neg_pad,
                0.122 - neg_pad,
                0.248])},
    }

    top_x = (
        0.5 * wrapped_obj_init_limits[True]["lo"][0]
        + 0.5 * wrapped_obj_init_limits[True]["hi"][0])
    mid_x = wrapped_obj_init_limits[True]["lo"][0]
    bottom_x = workspace_xyz_limits["lo"][0] + 0.02
    y0 = workspace_xyz_limits["lo"][1] + 0.03
    y1 = workspace_xyz_limits["lo"][1] + 0.07
    y2 = wrapped_obj_init_limits[True]["lo"][1] - 0.02
    y3 = wrapped_obj_init_limits[True]["lo"][1]
    pts = [
        # Points start from 12 o'clock and go clockwise.
        (top_x, y1),
        (bottom_x, y0),
        (bottom_x, y3),
        (mid_x, y3),
        (mid_x, y2),
        (top_x, y2),
    ]
    print(pts)
    obj_xy_init_polygon = PolygonWrapper(pts)

    x, y = obj_xy_init_polygon.exterior.xy

    rand_pts = obj_xy_init_polygon.sample_unif_rand_pts(50)
    print(x, y)
    plt.plot(x, y)
    rand_pts = np.array(rand_pts)
    plt.plot(*rand_pts.T, 'r.')
    plt.show()
