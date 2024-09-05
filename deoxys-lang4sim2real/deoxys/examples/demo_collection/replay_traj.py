import argparse
from pathlib import Path

import h5py

from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str, default="alice.yml")
    parser.add_argument(
        "--controller-cfg", type=str, default="osc-position-controller.yml"
    )
    parser.add_argument("--folder", type=Path, default="example_data")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Initialize robot interface
    controller_cfg = (
        YamlConfig(config_root + f"/{args.controller_cfg}").as_easydict())
    controller_type = "OSC_POSITION"

    robot_interface = FrankaInterface(config_root + f"/{args.interface_cfg}")

    demo_file_name = str(args.folder / "demo.hdf5")
    demo = h5py.File(demo_file_name, "r")

    episode = demo["data/ep_0"]

    actions = episode["actions"][()]

    for action in actions:
        robot_interface.control(
            controller_type=controller_type,
            action=action,
            controller_cfg=controller_cfg,
        )

    robot_interface.close()


if __name__ == "__main__":
    main()
