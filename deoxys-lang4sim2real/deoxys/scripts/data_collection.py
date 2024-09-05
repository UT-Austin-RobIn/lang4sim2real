"""Scripted Data Collection on the Real Robot"""
import argparse
import collections
import h5py
import numpy as np
import os
import time
from tqdm import tqdm

from deoxys.envs import init_env
from deoxys.policies import get_policy_class
from deoxys.policies.wrap_wire import WrapWire
from deoxys.utils.params import *
from deoxys.utils.data_collection_utils import (
    get_timestamp, load_env_info, concat_hdf5,
    paint_pp_rewards, get_obj_xy_pos, is_camera_feed_live)


class ScriptedDataCollector:
    def __init__(self, args):
        self.out_dir = args.out_dir
        self.obj_name = OBJECT_DETECTOR_CLASSES[args.obj_id]
        self.multistep_env = args.multistep_env

        kwargs = {"substeps_per_step": args.substeps_per_step}
        if args.state_mode is not None:
            kwargs['state_mode'] = args.state_mode
        if args.multistep_env or args.env in ["frka_wirewrap"]:
            kwargs['target_obj_name'] = self.obj_name
        if args.env in ["frka_wirewrap"]:
            kwargs['obj_set'] = args.obj_set
        self.env_name = args.env
        self.env = init_env(self.env_name, **kwargs)

        self.policy = get_policy_class(args.policy)(
            self.env,
            gripper_ac=OBJ_TO_GRASP_AC_MAP[self.obj_name])
        self.T = int(args.horiz)  # horizon

        self.ds_keys = [
            "observations", "actions", "rewards", "next_observations",
            "terminals"]
        self.obs_keys = ["image", "state"]
        self.next_obs_keys = ["state"]
        self.img_lang_keys = ['lang_list', 'lang_stage_num', 'target_obj']

        if args.multistep_env:
            self.multistep_keys = [
                'rews_by_step', 'step_idx', 'step_idx_to_num_stages_map',
                'rev_rews_by_step', 'rev_step_idx',
                'lang_stage_num_within_step']
        else:
            self.multistep_keys = []

        self.demo_fpaths = []

    def init_empty_traj_dict(self):
        traj_dict = {}
        for ds_key in self.ds_keys + self.img_lang_keys + self.multistep_keys:
            if ds_key in ["observations", "next_observations"]:
                traj_dict[ds_key] = {}
            else:
                traj_dict[ds_key] = []

        for obs_key in self.obs_keys:
            traj_dict["observations"][obs_key] = []

        for next_obs_key in self.next_obs_keys:
            traj_dict["next_observations"][next_obs_key] = []

        return traj_dict

    def add_transition(
            self, traj_dict, obs_dict, action, r, next_obs_dict, done, info,
            img_lang_dict, multistep_dict):
        for obs_key in self.obs_keys:
            if isinstance(obs_dict[obs_key], list):
                assert len(obs_dict[obs_key]) == self.env.substeps_per_step
                traj_dict["observations"][obs_key].extend(obs_dict[obs_key])
            else:
                traj_dict["observations"][obs_key].append(obs_dict[obs_key])
        for next_obs_key in self.next_obs_keys:
            if isinstance(next_obs_dict[next_obs_key], list):
                assert (
                    len(next_obs_dict[obs_key]) == self.env.substeps_per_step)
                traj_dict["next_observations"][next_obs_key].extend(
                    next_obs_dict[next_obs_key])
            else:
                traj_dict["next_observations"][next_obs_key].append(
                    next_obs_dict[next_obs_key])

        traj_dict["actions"].extend([action] * self.env.substeps_per_step)
        traj_dict["rewards"].extend([r] * self.env.substeps_per_step)
        traj_dict["terminals"].extend([done] * self.env.substeps_per_step)

        for img_lang_key in img_lang_dict:
            traj_dict[img_lang_key].extend(
                [img_lang_dict[img_lang_key]] * self.env.substeps_per_step)

        for multistep_key in multistep_dict:
            traj_dict[multistep_key].extend(
                [multistep_dict[multistep_key]] * self.env.substeps_per_step)

    def get_idx_from_obs_dict(self, obs_dict, idx):
        assert len(obs_dict) > 0
        indexed_obs_dict = {}
        for key in obs_dict:
            assert isinstance(obs_dict[key], list)
            indexed_obs_dict[key] = obs_dict[key][idx]
        return indexed_obs_dict

    def truncate_leading_substeps(self, traj_dict):
        """
        assumes all items in traj_dict are of dimension
        self.T * self.env.substeps_per_step
        except obs keys which are (self.T - 1) * self.env.substeps_per_step + 1

        truncate the leading self.env.substeps_per_step - 1 dimensions
        to make all things in traj_dict have the same horizon.
        """
        k = self.env.substeps_per_step
        new_T = (self.T - 1) * k + 1
        for ds_key in traj_dict:
            if ds_key == "next_observations":
                for obs_key in traj_dict[ds_key]:
                    traj_dict[ds_key][obs_key] = (
                        traj_dict[ds_key][obs_key][k-1:])
                    assert len(traj_dict[ds_key][obs_key]) == new_T
            elif ds_key == "observations":
                for obs_key in traj_dict[ds_key]:
                    assert len(traj_dict[ds_key][obs_key]) == new_T
            else:
                traj_dict[ds_key] = traj_dict[ds_key][k-1:]
        return traj_dict

    def maybe_update_scripted_policy_multistep(
            self, prev_step_idx, prev_rev_step_idx, task_id, do_forward):
        """
        Only switch policies if env has advanced
        """
        no_switch_forward = (
            do_forward
            and (
                (prev_step_idx == self.env.step_idx)
                or (self.env.step_idx == self.env.num_steps)))
        no_switch_reverse = (
            (not do_forward)
            and (
                (prev_rev_step_idx == self.env.rev_step_idx)
                or (self.env.rev_step_idx == self.env.num_steps)))

        if no_switch_forward or no_switch_reverse:
            return

        if do_forward:
            step_idx = self.env.step_idx
        else:
            step_idx = self.env.num_steps - self.env.rev_step_idx - 1

        obj_name, cont_name = self.get_obj_cont_name(do_forward)
        print(f"obj_name, cont_name = ({obj_name}, {cont_name})")
        pick_pt, drop_pt, lift_pt_z = self.get_pick_drop_lift_pt(
            do_forward, obj_name, cont_name)
        policy_class = get_policy_class(
            self.env.step_kwargs['skills'][step_idx])
        self.policy = policy_class(
            self.env,
            gripper_ac=OBJ_TO_GRASP_AC_MAP[obj_name])
        self.policy.reset(
            pick_pt, drop_pt, obj_name, lift_pt_z=lift_pt_z)

    def try_get_obj_xy_pos(self, obj_name):
        try:
            obj_pos_in_robot_coords = get_obj_xy_pos(self.env, obj_name)
        except:
            obj_pos_in_robot_coords = None
        return obj_pos_in_robot_coords

    def get_pick_drop_lift_pt(self, do_forward, obj_name, cont_name=""):
        # Compute pick and drop pt
        obj_pos_in_robot_coords = self.try_get_obj_xy_pos(obj_name)
        while obj_pos_in_robot_coords is None:
            input(f"Cannot find {obj_name} while trying do_forward={do_forward}."
                  " Move into view of camera then press enter.")
            obj_pos_in_robot_coords = self.try_get_obj_xy_pos(obj_name)
        pick_pt_z = OBJ_TO_PICK_PT_Z_MAP[obj_name] + self.env.gripper_z_offset
        pick_pt = (
            np.concatenate([obj_pos_in_robot_coords, [pick_pt_z]])
            + OBJ_TO_PICK_XYZ_OFFSET[obj_name])
        lift_pt_z = None  # Only used in backward direction
        if do_forward:
            if cont_name != "":
                drop_xy_offset = self.env.step_kwargs[
                    'drop_pos_offsets'][self.env.step_idx]
                drop_xy = get_obj_xy_pos(self.env, cont_name)
                drop_pt = (
                    np.concatenate([drop_xy, [pick_pt_z]]) + drop_xy_offset)
            elif self.env_name == "frka_wirewrap":
                drop_pt = np.array([0.31, -0.18, pick_pt_z + 0.01])
            elif self.env_name == "frka_pp":
                drop_pt = np.array([0.35, 0.09, pick_pt_z + 0.05])
        else:
            # Resetting object out of container
            # Pick random xy point outside of container.
            if self.multistep_env:
                pick_pt[2] += self.env.rev_step_kwargs["pick_pt_z_offsets"][
                    self.env.rev_step_idx]
                lift_pt_z = (
                    0.12 + self.env.gripper_z_offset +
                    self.env.rev_step_kwargs[
                        "lift_pt_z_offsets"][self.env.rev_step_idx])
                rnd_non_cont_xy = self.env.propose_xy_init_pos(obj_name)
                if cont_name != "":
                    # step_idx = self.env.num_steps - self.env.rev_step_idx - 1
                    drop_offset = self.env.rev_step_kwargs['drop_pos_offsets'][
                        self.env.rev_step_idx]
            else:
                rnd_non_cont_xy = self.env.propose_obj_xy_init_pos()
                drop_offset = np.array([0., 0., 0.])
                if self.env_name == "frka_wirewrap":
                    drop_offset = np.array([0., 0., 0.01])
                elif self.env_name == "frka_pp":
                    drop_offset = np.array([0., 0., 0.05])
            drop_pt = (
                np.concatenate([rnd_non_cont_xy, [pick_pt_z]]) + drop_offset)
            if self.env_name == "frka_wirewrap":
                self.env.obj_xy_init_polygon.show_pt(
                    drop_pt[:2], fname="drop_pt_proposal.png")
        return pick_pt, drop_pt, lift_pt_z

    def is_do_forward(self):
        if self.env_name in ["frka_wirewrap"]:
            return self.env.is_do_forward()
            # ^ if unwrapped do forward, else backward
        elif not self.multistep_env:
            # single step env
            obj_pos_in_robot_coords = get_obj_xy_pos(self.env, self.obj_name)
            return not self.env.obj_xy_placed_in_cont(obj_pos_in_robot_coords)
        else:
            rews_by_step = self.env.reward_by_step(info={})
            # Only do forward if each step has 0 reward.
            assert set(rews_by_step).issubset({0.0, 1.0})
            return (np.array(rews_by_step) == 0).all()

    def get_obj_cont_name(self, do_forward):
        if not self.multistep_env or self.env_name in ["frka_wirewrap"]:
            target_obj_name = self.obj_name
            cont_name = ""
        else:
            target_obj_name, cont_name = self.env.get_obj_cont_name(do_forward)
        return target_obj_name, cont_name

    def get_lang_stage_num_multistep(
            self, dir_agnostic_step_idx, lang_stage_within_step,
            step_idx_to_num_stages_map, do_forward):
        # calculate multistep lang stage_num across diff steps
        # 1. use step_idx before it changed in self.env.step
        # 2. make sure step_idx doesn't advance above self.env.num_steps else
        # the lang_stage_num addition will not make sense.
        step_idx = dir_agnostic_step_idx
        step_idx = min(self.env.num_steps - 1, step_idx)
        rev = -1 if not do_forward else 1
        # ^ flip the step_idx_to_num_stages_map array
        cum_num_stages = np.sum(
            np.array(step_idx_to_num_stages_map)[::rev][:step_idx])
        lang_stage_num = cum_num_stages + lang_stage_within_step
        return lang_stage_num

    def collect_single_traj(self):
        # Keep trying until a successful traj is collected.
        successful_traj = False
        while not successful_traj:
            # Init traj datastructures
            traj_dict = self.init_empty_traj_dict()

            do_forward = self.is_do_forward()
            print("do_forward", do_forward)
            task_id = int(not do_forward)
            env_kwargs = {}
            multistep_dict = {}  # Only stores current transition, not entire traj
            if self.multistep_env or self.env_name in ["frka_wirewrap"]:
                env_kwargs["do_forward"] = do_forward
            obs = self.env.reset(**env_kwargs)

            # these need to be decided after the rev_step_idx is calculated
            # in reset.
            obj_name, cont_name = self.get_obj_cont_name(do_forward)
            pick_pt, drop_pt, lift_pt_z = self.get_pick_drop_lift_pt(
                do_forward, obj_name, cont_name)

            obs_for_policy = obs
            if self.multistep_env:
                step_idx = (
                    self.env.step_idx if do_forward
                    else self.env.rev_step_idx)
                lang_list, step_idx_to_num_stages_map = (
                    self.env.get_lang_by_stages(do_forward))
            elif self.env_name in ["frka_wirewrap"]:
                lang_list = self.env.get_lang_by_stages(do_forward)
            else:
                lang_list = self.env.get_lang_by_stages(
                    do_forward, self.obj_name)

            if type(self.policy) == WrapWire:
                self.policy.reset(
                    pick_pt, drop_pt,
                    not do_forward,
                    gripper_ac=OBJ_TO_GRASP_AC_MAP[obj_name])
            else:
                self.policy.reset(
                    pick_pt, drop_pt, obj_name,
                    gripper_ac=OBJ_TO_GRASP_AC_MAP[obj_name],
                    lift_pt_z=lift_pt_z)

            for t in range(self.T):
                if t == 0:
                    start_time = time.time()
                action, agent_info = self.policy.get_action(obs_for_policy)

                # Store info for img-lang (phase1)
                img_lang_dict = dict(
                    target_obj=self.obj_name,
                )

                noise = np.concatenate(
                    [np.random.normal(scale=args.noise, size=(3,)),
                     np.zeros(self.env.action_dim - 3,)])
                action += noise
                # print(
                #     "action", action,
                #     "\t stage", agent_info['policy_lang_stage_num'])

                if self.env.num_steps > 1:
                    prev_step_idx = self.env.step_idx
                    prev_rev_step_idx = self.env.rev_step_idx
                else:
                    prev_step_idx = 0
                    prev_rev_step_idx = 0

                time_overhead = time.time() - start_time
                next_obs, r, done, info = self.env.step(action, time_overhead)
                start_time = time.time()

                if info['forced_reset']:
                    break

                if self.multistep_env:
                    self.maybe_update_scripted_policy_multistep(
                        prev_step_idx, prev_rev_step_idx, task_id, do_forward)
                    multistep_dict['rews_by_step'] = info['rews_by_step']
                    multistep_dict['rev_rews_by_step'] = info['rev_rews_by_step']
                    multistep_dict['step_idx'] = info['step_idx']
                    multistep_dict['rev_step_idx'] = info['rev_step_idx']
                    img_lang_dict['lang_stage_num'] = (
                        self.get_lang_stage_num_multistep(
                            prev_step_idx if do_forward else prev_rev_step_idx,
                            agent_info['policy_lang_stage_num'],
                            step_idx_to_num_stages_map,
                            do_forward))
                    print(
                        t, action, "lang_stage_num",
                        img_lang_dict['lang_stage_num'])
                    multistep_dict['lang_stage_num_within_step'] = (
                        agent_info['policy_lang_stage_num'])
                else:
                    img_lang_dict['lang_stage_num'] = (
                        agent_info['policy_lang_stage_num'])

                self.add_transition(
                    traj_dict, obs, action, r, next_obs, done, info,
                    img_lang_dict, multistep_dict)

                # Check that the camera feed didn't freeze.
                camera_feed_is_live = is_camera_feed_live(obs, next_obs)

                if not camera_feed_is_live:
                    print("camera froze up, quitting collection of this traj.")
                    input("Replug in the camera. Then press enter.")
                    break

                obs = next_obs
                if self.env.substeps_per_step == 1:
                    obs_for_policy = obs
                elif self.env.substeps_per_step > 1:
                    obs_for_policy = self.get_idx_from_obs_dict(next_obs, -1)
                else:
                    raise NotImplementedError

            # If successful, save
            if self.multistep_env or self.env_name in ["frka_wirewrap"]:
                successful_traj = bool(traj_dict["rewards"][-1])
            else:
                obj_pos_in_robot_coords = get_obj_xy_pos(
                    self.env, self.obj_name, lift_before_reset=True)
                obj_placed_in_cont = self.env.obj_xy_placed_in_cont(
                    obj_pos_in_robot_coords)
                obj_placed_out_cont = self.env.obj_xy_placed_out_cont(
                    obj_pos_in_robot_coords)

                successful_traj = (
                    (do_forward and obj_placed_in_cont)
                    or (not do_forward and obj_placed_out_cont))

            # successful_traj only when camera feed hasn't frozen
            successful_traj = successful_traj and camera_feed_is_live

            if successful_traj:
                gripper_states_arr = (
                    np.array(traj_dict['next_observations']['state'])[
                        :, self.env.gripper_state_idx])

                if not self.multistep_env:
                    # Only paint rewards if single step env.
                    # multistep env will have ground truth reward function
                    traj_dict['rewards'] = paint_pp_rewards(
                        self.env, self.env.substeps_per_step * self.T,
                        successful_traj, gripper_states_arr)
                traj_dict = self.truncate_leading_substeps(traj_dict)
                traj_dict['lang_list'] = lang_list

                if self.multistep_env:
                    traj_dict['step_idx_to_num_stages_map'] = (
                        step_idx_to_num_stages_map)
                self.save_single_demo_hdf5(traj_dict, task_id, do_forward)
            else:
                # Open gripper if gripper ends closed.
                last_gripper_state = traj_dict['next_observations']['state'][
                    -1][self.env.gripper_state_idx]
                if not self.env.gripper_open(last_gripper_state):
                    print("opening gripper at end of traj")
                    self.env.step(np.array([0., 0., 0., 0., 0., -1.]))

    def save_single_demo_hdf5(self, traj_dict, task_id, do_forward):
        # Create single-demo hdf5 file.
        demo_file_name = f"{self.out_dir}/{get_timestamp()}.hdf5"
        demo_file = h5py.File(demo_file_name, "w")

        grp = demo_file.create_group("data")
        self.task_to_grp_map = {
            0: grp.create_group("0"),  # Forward
            1: grp.create_group("1"),  # Backward
        }

        # Save the dataset group
        task_id_grp = self.task_to_grp_map[task_id]

        if "is_multistep" not in task_id_grp.attrs:
            task_id_grp.attrs['is_multistep'] = args.multistep_env

        traj_idx = self.task_id_to_num_trajs_map[task_id]
        ep_grp = task_id_grp.create_group(f"demo_{traj_idx}")
        self.task_id_to_num_trajs_map[task_id] += 1

        obs_grp = ep_grp.create_group("observations")
        next_obs_grp = ep_grp.create_group("next_observations")

        for obs_key in self.obs_keys:
            obs_grp.create_dataset(
                obs_key, data=np.stack(traj_dict["observations"][obs_key]))
        for next_obs_key in self.next_obs_keys:
            next_obs_grp.create_dataset(
                next_obs_key,
                data=np.stack(traj_dict["next_observations"][next_obs_key]))

        ep_grp.create_dataset("actions", data=np.stack(traj_dict["actions"]))
        ep_grp.create_dataset("rewards", data=np.stack(traj_dict["rewards"]))
        ep_grp.create_dataset(
            "terminals", data=np.stack(traj_dict["terminals"]))
        # ep_grp.create_dataset("info", data=np.stack(traj_dict["info"]))
        ep_grp.create_dataset(
            "lang_stage_num", data=np.array(traj_dict["lang_stage_num"]))

        if self.multistep_env:
            ep_grp.create_dataset("lang_stage_num_within_step", data=np.array(
                traj_dict["lang_stage_num_within_step"]))

            if do_forward:
                rews_by_step = traj_dict["rews_by_step"]
                step_idx = traj_dict["step_idx"]
            else:
                rews_by_step = traj_dict["rev_rews_by_step"]
                step_idx = traj_dict["rev_step_idx"]
            ep_grp.create_dataset(
                "rewards_by_step", data=np.array(rews_by_step))
            ep_grp.create_dataset("step_idx", data=np.array(step_idx))
            ep_grp.attrs["step_idx_to_num_stages_map"] = np.array(
                traj_dict['step_idx_to_num_stages_map'])

        ep_grp.attrs["target_obj"] = traj_dict["target_obj"][0]
        ep_grp.attrs["lang_list"] = traj_dict["lang_list"]
        ep_grp.attrs["num_samples"] = len(ep_grp["actions"])

        self.demo_fpaths.append(demo_file_name)

    def collect_trajs(self, n):
        self.task_id_to_num_trajs_map = collections.Counter()

        os.makedirs(f"{self.out_dir}", exist_ok=True)

        for i in tqdm(range(n)):
            self.collect_single_traj()

        self.env.reset()

        # Save to a single demo file.
        # Concat everything in self.demo_fpaths
        env_info = load_env_info(args.env)
        print("self.demo_fpaths", self.demo_fpaths)
        out_path = concat_hdf5(
            self.demo_fpaths, args.out_dir, env_info, args.env,
            demo_attrs_to_del=["lang_list", "is_multistep"])

        print("TaskID --> # trajs collected", self.task_id_to_num_trajs_map)


if __name__ == "__main__":
    # python ~/Projects/albert/deoxys_v2/deoxys/scripts/data_collection.py --out-dir /home/robin/Projects/albert/datasets/realrobot_debug --policy pick_place --env frka_pp --horiz 18 --noise 0.05 --obj-id 1 --num 1 --state-mode 1 --substeps-per-step 1
    # python ~/Projects/albert/deoxys_v2/deoxys/scripts/data_collection.py --out-dir /home/robin/Projects/albert/datasets/realrobot_debug --policy pick_place --env frka_obj_pot_stove --horiz 18 --noise 0.05 --obj-id 3 --num 1 --state-mode 1 --substeps-per-step 1 --multistep-env
    # python ~/Projects/albert/deoxys_v2/deoxys/scripts/data_collection.py --out-dir /home/robin/Projects/albert/datasets/realrobot_debug --policy pick_place_n --env frka_obj_bowl_plate --horiz 45 --noise 0.05 --obj-id 6 --num 2 --state-mode 1 --substeps-per-step 1 --multistep-env
    # python ~/Projects/albert/deoxys_v2/deoxys/scripts/data_collection.py --out-dir /home/robin/Projects/albert/datasets/realrobot_debug --policy wrap_wire --env frka_wirewrap --horiz 45 --noise 0.05 --obj-id 2 --num 2 --state-mode 1 --substeps-per-step 1
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--horiz", type=str, required=True)
    parser.add_argument("--obj-id", type=int, required=True)
    parser.add_argument("--num", type=int, required=True)
    parser.add_argument(
        "--noise", type=float, default=0.0,
        help="std dev (scale) of gaussian noise")
    parser.add_argument(
        "--state-mode", type=int, default=None, choices=[0, 1, None])
    parser.add_argument("--substeps-per-step", type=int, default=1)
    parser.add_argument("--multistep-env", action="store_true", default=False)
    parser.add_argument("--obj-set", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--stage-num-scheme", type=str, choices=["v1", "v2"], default=None)
    args = parser.parse_args()

    if args.env == "frka_wirewrap":
        assert args.stage_num_scheme is not None

    args.out_dir = os.path.join(args.out_dir, get_timestamp())
    collector = ScriptedDataCollector(args)

    collector.collect_trajs(args.num)
