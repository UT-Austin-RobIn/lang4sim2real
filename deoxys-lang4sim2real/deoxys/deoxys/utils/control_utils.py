import numpy as np
import time


def reset_joints(robot_interface, controller_cfg, logger, reset_joint_pos_idx=0):
    # Golden resetting joints (from Yifeng)
    # reset_joint_positions = [
    #     0.09162008114028396,
    #     -0.19826458111314524,
    #     -0.01990020486871322,
    #     -2.4732269941140346,
    #     -0.01307073642274261,
    #     2.30396583422025,
    #     0.8480939705504309,
    # ]
    reset_joint_position_options = [
        [ # 20230326
            0.087946,   -0.016741,   -0.039851,     -2.5719,   -0.012878,      2.5522,     0.84145],
        [ # 20240103
            0.088363,    -0.16786,    -0.03984,     -2.5369,   -0.012567,      2.3819,     0.84169],
        [ # 20240112  ee pos: 0.4838,    0.017224,     0.27115
            0.083965,    -0.12385,   -0.046626,     -2.3713,   -0.012875,      2.2573,      0.82544],  # wrist: 0.82544  # 2.7
    ]

    reset_joint_positions = reset_joint_position_options[reset_joint_pos_idx]

    # This is for varying initialization of joints a little bit to
    # increase data variation.
    # reset_joint_positions = [
    #     e + np.clip(np.random.randn() * 0.005, -0.005, 0.005)
    #     for e in reset_joint_positions
    # ]
    action = reset_joint_positions + [-1.0]

    while True:
        if len(robot_interface._state_buffer) > 0:
            logger.info(f"Current Robot joint: {np.round(robot_interface.last_q, 3)}")
            logger.info(f"Desired Robot joint: {np.round(robot_interface.last_q_d, 3)}")

            if (
                np.max(
                    np.abs(
                        np.array(robot_interface._state_buffer[-1].q)
                        - np.array(reset_joint_positions)
                    )
                )
                < 1e-3
            ):
                break
        robot_interface.control(
            controller_type="JOINT_POSITION",
            action=action,
            controller_cfg=controller_cfg,
        )


def get_obs_for_calib(
        env, image_xyz=None, skip_reset=False,
        ret_old_pose=False, lift_before_reset=False, wait_secs=1.0):
    obs = env.get_obs_state()
    joint_angles = obs['joint_pos']
    if image_xyz is None:
        if not skip_reset:
            env.reset(lift_before_reset=lift_before_reset)
    else:
        env.reach_xyz_pos(image_xyz)
    time.sleep(wait_secs)  # wait for arm to catch up
    obs = env.get_obs()
    if ret_old_pose:
        env.reach_joint_pose(joint_angles)
    return obs
