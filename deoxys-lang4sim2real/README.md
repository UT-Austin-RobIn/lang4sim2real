# deoxsys-lang4sim2real
This is our version of the [original deoxys repo](https://github.com/UT-Austin-RPL/deoxys_control) with our 3 real-world environments (pick-and-place, 2-step pick-and-place, and wire wrap), associated scripted forward and reset policies to collect data near-autonomously with initial position randomization, and evaluation scripts for evaluating policy checkpoints.

Important files and folders added on top of the original deoxys:
- **[Object Detector](./deoxys/deoxys/utils/obj_detector.py)**
- **[Settings file](./deoxys/deoxys/utils/params.py)**: Modify this for your own workspace settings
- **[Object Detector to Robot Coordinate Calibration](./deoxys/scripts/obj_pixel_robot_coord_calibration.py)**
- **Environments**: [1pp](./deoxys/deoxys/envs/franka_env.py), [2pp](./deoxys/deoxys/envs/franka_env.py), [ww](./deoxys/deoxys/envs/franka_wirewrap_env.py)
- **Scripted policies**: [pick-place](./deoxys/deoxys/policies/pick_place.py), [wrap wire](./deoxys/deoxys/policies/wrap_wire.py)
- **[Real-world Data collection](./deoxys/scripts/data_collection.py)**
- **[Real-world Policy Evaluator](./deoxys/scripts/eval_collector.py)**

<p align="center">
<img src="./deoxys_github_logo.png">
</p>

[**[Documentation]**](https://ut-austin-rpl.github.io/deoxys-docs/html) &ensp; 

Deoxys is a modular, real-time controller library for Franka Emika Panda arm, aiming to facilitate a wide range of robot learning research. Deoxys comes with a user-friendly python interface and real-time controller implementation in C++. If you are a [robosuite](https://github.com/ARISE-Initiative/robosuite) user, Deoxys APIs provide seamless transfer 
from you simulation codebase to real robot experiments!




https://user-images.githubusercontent.com/21077484/206338997-8dbaa128-dc63-4911-84ca-64d80a05673f.mp4



## Cite the Original Deoxys Codebase

```
@article{zhu2022viola,
  title={VIOLA: Imitation Learning for Vision-Based Manipulation with Object Proposal Priors},
  author={Zhu, Yifeng and Joshi, Abhishek and Stone, Peter and Zhu, Yuke},
  journal={6th Annual Conference on Robot Learning (CoRL)},
  year={2022}
}
```


# Installation of codebase

Overall, the installation has three parts:
1. Install dependencies by running `InstallPackage`
2. Compile desktop-side codebase (Python)
3. Compile NUC-side codebase (C++)

Here are the details. For more information, please refer to the [Codebase Installation Page](https://ut-austin-rpl.github.io/deoxys-docs/html/installation/codebase_installation.html).

Clone this repo to the robot workspace directory on Desktop computer (e.g. `/home/USERNAME/robot-control-ws`)

``` shell
cd deoxys_control/deoxys
```

## Install dependencies

Run the `InstallPackage` file to install necessary packages.
``` shell
./InstallPackage
```


## Deoxys - Desktop

Make sure that you are in your python virtual environment before
	building this.
``` shell
make -j build_deoxys=1
```

And install all the python dependencies (feel free to add pull requests if anything is missing) from `deoxys_control/requirements.txt`, by doing:
```shell
pip install -U -r requirements.txt
```

Q: We might want to read states faster than the control rate. The best
way might be run another node and write states into redis buffer?


## Franka Interface - Intel NUC

Franka Interface is the part which is supposed to run on NUC. Run this 
command in directory `deoxys_control/deoxys/` on Intel NUC. 

``` shell
make -j build_franka=1
```

## A laundry list of pointers:
   - [How to turn on/off the robot](https://ut-austin-rpl.github.io/deoxys-docs/html/tutorials/running_robots.html)
   - [How to install spacemouse](https://ut-austin-rpl.github.io/deoxys-docs/html/tutorials/using_teleoperation_devices.html)
   - [How to set up the RTOS](https://ut-austin-rpl.github.io/deoxys-docs/html/installation/system_prerequisite.html)
   - [How to record and replay a trajectory](https://ut-austin-rpl.github.io/deoxys-docs/html/tutorials/record_and_replay.html)
   - [How to write a simple motor program](https://ut-austin-rpl.github.io/deoxys-docs/html/tutorials/handcrafting_motor_program.html)

# Control the robot

## Commands on Desktop

Here is a quick guide to run `Deoxys`.

Under `deoxys_control/deoxys`,  run

``` shell
python examples/run_deoxys_with_space_mouse.py 
```

Change 1) spacemouse vendor_id and product_id ([here](https://github.com/UT-Austin-RPL/deoxys_control/blob/eb8d69f7f0838389fca81cac6b250ba05fc97f92/deoxys/examples/run_deoxys_with_space_mouse.py#L19)) 2) robot interface 
config ([here](https://github.com/UT-Austin-RPL/deoxys_control/blob/eb8d69f7f0838389fca81cac6b250ba05fc97f92/deoxys/examples/run_deoxys_with_space_mouse.py#L16)) if necessary.

You might also check and change the PC / NUC names [here](https://github.com/UT-Austin-RPL/deoxys_control/blob/master/deoxys/config/charmander.yml). 

## Commands on Control PC (Intel NUC)

Under `deoxys_control/deoxys`, run two commands. One for real-time control of the arm, one for non
real-time control of the gripper.

``` shell
bin/franka-interface config/charmander.yml
```

``` shell
bin/gripper-interface config/charmander.yml
```
