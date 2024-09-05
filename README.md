# lang4sim2real

## Introduction
This codebase contains the training code and algorithm for the Lang4Sim2Real paper:

**Natural Language Can Help Bridge the Sim2Real Gap** <br />
Albert Yu, Adeline Foote, Raymond J. Mooney, and Roberto Martín-Martín<br />
Robotics: Science and Systems (RSS), 2024 <br />
[Web](https://robin-lab.cs.utexas.edu/lang4sim2real/) | [PDF](https://arxiv.org/pdf/2405.10020) | [5-min video](https://youtu.be/UHi91NWQf08?si=wxIKpl4DReOL3Tiw) <br />


This codebase builds on the functions/classes from the previously released repo, [deltaco](https://github.com/Alacarter/deltaco), which was released with the [Del-TaCo paper](https://arxiv.org/pdf/2210.04476.pdf).

## Citation
```
@inproceedings{yu2024lang4sim2real,
      title={Natural Language Can Help Bridge the Sim2Real Gap},
      author={Yu, Albert and Foote, Adeline and Mooney, Raymond and Martín-Martín, Roberto},
      booktitle={Robotics: Science and Systems (RSS), 2024},
      year={2024}
}
```

## Table of Contents: Steps to Reproducing Our Sim2Real Results
- [**Step 0.** Setting Up](#step-0-setting-up)
- [**Step 1.** Collect Sim+Real Data](#step-1-collect-data)
  * [Option A: Download Our Datasets](#option-a-download-our-datasets)
  * [Option B: Collect Your Own Data](#option-b-collect-your-own-data)
- [**Step 2.** Pretrain Policy CNN](#step-2-pretrain-policy-cnn)
  * [Option A: Download Our Pretrained Checkpoints](#option-a-download-our-pretrained-checkpoints)
  * [Option B: Pretrain Your Own Checkpoint](#option-b-pretrain-your-own-checkpoint)
- [**Step 3.** Train Policies with Multi-task, Multi-domain BC](#step-3-train-policies-with-multi-task-multi-domain-bc)
- [**Step 4.** Evaluate Policies](#step-4-evaluate-policies)

## Step 0. Setting Up
After cloning this repo and `cd`-ing into it:
```
cd train-lang4sim2real
conda env create -f env.yml
pip install -e .
python setup.py develop
cp rlkit/launchers/config_template.py rlkit/launchers/config.py
```

Modify the `LOCAL_LOG_DIR` in `train-lang4sim2real/rlkit/launchers/config.py` to a path on your machine where the experiment logs will be saved.

Pip-install the local sim environments (our version of the original [robosuite](https://github.com/ARISE-Initiative/robosuite) repo):
```
cd ../robosuite-lang4sim2real
pip install -r requirements.txt
pip install -e .
```

If you plan on collecting data from scratch, also pip-install our local version of the original [robomimic](https://github.com/ARISE-Initiative/robomimic) repo by following:
```
cd ../robomimic-lang4sim2real/robomimic
pip install -e .
```

On a machine with access to a real Franka Emika Panda robot, create a new python environment and install the real environments (our version of [deoxys](https://github.com/UT-Austin-RPL/deoxys_control)):
```
cd ../../deoxys-lang4sim2real
./InstallPackages
make -j build_deoxys=1
pip install -U -r requirements.txt
```
Follow instructions for compiling NUC codebase [here](https://github.com/UT-Austin-RPL/deoxys_control?tab=readme-ov-file#franka-interface---intel-nuc), as well as additional documentation [here](https://zhuyifengzju.github.io/deoxys_docs/html/index.html).

Clone and install the [sentence transformers repo](https://github.com/UKPLab/sentence-transformers).
```
cd ../..
git clone git@github.com:UKPLab/sentence-transformers.git
pip install -e .
```

Be aware that there may be more dependences you may need to `pip install` to run specific parts of our code. Open an issue if you have difficulty with installation.

### Extra setup steps
#### BLEURT
To install the [pytorch implementation of BLEURT](https://github.com/lucadiliello/bleurt-pytorch), run:
```
pip install git+https://github.com/lucadiliello/bleurt-pytorch.git
```
Afterwards, you should be able to run:
```
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
```

#### CLIP
If you wish to run experiments involving CLIP as the visual backbone of the policy, you will need to install [open_clip](https://github.com/mlfoundations/open_clip) and add the following line to your `~/.bashrc` file:

`export PYTHONPATH="$PYTHONPATH:[path_to_openclip_repo]/src"`

#### R3M
If you wish to run experiments with R3M as the visual backbone of the policy, see the [r3m repo](https://github.com/facebookresearch/r3m) for installation details.


## Step 1. Collect Sim+Real Data
### Option A: Download Our Datasets
All our datasets are [on Box](https://utexas.box.com/s/jb4ybp6z93d69txnwtxjmn9epkag5ws3). However, the 2-step Pick-and-Place datasets are [on OneDrive](https://utexas-my.sharepoint.com/:f:/g/personal/ayms_utexas_edu/EpJvNuf-lvBHhCaFVrUqW_cBEYJcUkWU9Kc2QMsTxprUeQ?e=bTOSHa) due to being larger than the Box file size limit.
#### Pick-and-Place
##### sim2real
[.../1pp_sim2real.hdf5](https://utexas.box.com/s/kpysx9fevilynm4clswzqfbz13pgn86c)
- 0-3: Sim prior domain, 400 trajs/task, 200 timesteps/traj. 4 different robosuite objects for the four task indices.
- 4-5: Real target domain, 500 trajs/task, 18 timesteps/traj. carrot, forward or backward directions for the two task indices.
- 6-7: Real prior task, target domain, 50 trajs/task. 18 timesteps/traj. paper box, forward or backward directions for the two task indices.

##### sim2sim
[.../1pp_sim2sim.hdf5](https://utexas.box.com/s/5r7ywfmrpwhp5t972n9hh2e07boqjunf)
- 0-3: Sim prior domain, 400 trajs/task, 200 timesteps/traj.
- 4-7: Sim target domain, 95 trajs/task, 200 timesteps/traj.

#### 2-step Pick-and-Place
##### sim2real
[.../2pp_sim2real.hdf5](https://utexas-my.sharepoint.com/:u:/g/personal/ayms_utexas_edu/ER4uilUH-GFOr0flSZS60O8BnC0s6ysbzB28QstYXg_nug?e=5JTd6M)
- 0-3: Sim prior domain, 1375 trajs/task, 320 timesteps/traj. 4 different robosuite objects for the four task indices.
- 4-5: Real target domain, 102 trajs (task 4), 101 trajs (task 5), 45 timesteps/traj. carrot into bowl onto plate.


##### sim2sim
[.../2pp_sim2sim.hdf5](https://utexas-my.sharepoint.com/:u:/g/personal/ayms_utexas_edu/EZcsujoMsr5Nn9sfMhdWw8QB8MgJ_QG83Fwjb0wWnz3w0w?e=UjXWIv)
- 0-3: Sim prior domain, 1375 trajs/task, 320 timesteps/traj. 4 different robosuite objects for the four task indices.
- 4-7: Sim target domain, 100 trajs/task, 320 timesteps/traj. 4 different robosuite objects for the four task indices.

#### Wrap Wire
##### sim2real
[.../ww_sim2real.hdf5](https://utexas.box.com/s/4iemyiqlqgljoe74pv81yywibjhg6zgq)
- 0: Sim prior domain, 1000 trajs, 200 timesteps/traj.
- 1: Real target domain target task, 98 trajs, 45 timesteps/traj.
- 2: Real target domain unused task (reverse data of task 1), 102 trajs, 45 timesteps/traj.

##### sim2sim
[.../ww_sim2sim.hdf5](https://utexas.box.com/s/isg0hrjnkq6il9ppx7fh1614kwy034ql)
- 0-1: Sim prior domain, 400 trajs/task, 200 timesteps/traj. 0 (counterclockwise), 1 (clockwise).
- 2-3: Sim target domain, 100 trajs/task, 200 timesteps/traj. 2 (counterclockwise), 3 (clockwise).

### Baseline Sim2Real Datasets
- To collect **Domain Rando** datasets, simply add the flag `--randomize wide` to the `collect_demonstrations_parallel.py` data collection script described [below](#option-b-collect-your-own-data).
- To collect **ADR+RNA** datasets, simply add the flag `--adr-rna` to the `collect_demonstrations_parallel.py` data collection script.
#### Pick-and-Place
[.../1pp_domain-rando_sim2real.hdf5](https://utexas.box.com/s/195cfxbzy85nw8g5tsdhnv6gamgyolmr)
[.../1pp_adr-rna_sim2real.hdf5](https://utexas.box.com/s/vxm2jjqk5zr9a171a565hjlnt6k0t1bb)
The task indices of the two baseline datasets are as described:
- 0-3: Sim prior domain, data collected from domain randomization or ADR+RNA. 400 trajs/task, 200 timesteps/traj. 4 different robosuite objects for the four task indices.
- 4-5: Real target domain, 500 trajs/task, 18 timesteps/traj. carrot, forward or backward directions for the two task indices.
- 6-7: Real prior task, target domain, 50 trajs/task. 18 timesteps/traj. paper box, forward or backward directions for the two task indices.

#### 2-step Pick-and-Place
[.../2pp_domain-rando_sim2real.hdf5](https://utexas-my.sharepoint.com/:u:/g/personal/ayms_utexas_edu/ETZ3pQw5JBBLnqDrxWm9PcwBH_aV3yXNmkvooBhu6BbZ3w?e=JVfZBZ)
[.../2pp_adr-rna_sim2real.hdf5](https://utexas-my.sharepoint.com/:u:/g/personal/ayms_utexas_edu/EQHNhNB-l3hAjUvlALbhpmQB9g3ytbvz5ss29svXDr5i5w?e=6Vp0tQ)
The task indices of the two baseline datasets are as described:
- 0-3: Sim prior domain, data collected from domain randomization or ADR+RNA. 1400 trajs/task, 320 timesteps/traj.
- 4-5: Real target task, target domain, 102 trajs (task 4), 101 trajs (task 5), 45 timesteps/traj. carrot into bowl onto plate (forward and reverse task directions).
- 6-7: Real prior task, target domain, 50 trajs/task, 45 timesteps/traj. wooden bridge block into bowl onto plate (forward and reverse task directions).

#### Wrap Wire
[.../ww_domain-rando_sim2real.hdf5](https://utexas.box.com/s/rwiz3ims4q6u2mn86jv1efmkjg910pua)
[.../ww_adr-rna_sim2real.hdf5](https://utexas.box.com/s/cvs9hkjbpvu426sym94cc4736cpaca3w)
The task indices of the two baseline datasets are as described:
- 0: Sim prior domain, data collected from domain randomization or ADR+RNA. 1024 trajs (domain rando) and 950 trajs (domain rando).
- 1-2: Real target task, target domain. 98 trajs/task. Wrapping wire with eu plug around blender.
- 3-4: Real prior task, target domain. 51 trajs/task. Wrapping ethernet cable with wooden bridge block around spool.

### Option B: Collect Your Own Data
#### Sim
##### Pick-and-Place
###### Source Domain
```
python robosuite-lang4sim2real/robosuite/scripts/collect_demonstrations_parallel.py --robots Panda --environment Multitaskv2 --device scripted-policy --noise-std 0.05 -n 1600 -p 40 --task-idx-intervals 0-3 --directory [.../data_collection_out_dir] --camera agentview --img-dim 128 --state-mode 1 --multitask-hdf5-format --intra-thread-delay 30
```

###### Target Domain
```
python robosuite-lang4sim2real/robosuite/scripts/collect_demonstrations_parallel.py --robots Panda --environment Multitaskv2_ang1_fr5damp50 --device scripted-policy --noise-std 0.05 -n 400 -p 20 --task-idx-intervals 0-3 --directory [.../data_collection_out_dir] --camera agentview --img-dim 128 --state-mode 1 --multitask-hdf5-format --intra-thread-delay 3
```

##### 2-step Pick-and-Place
###### Source Domain
```
python robosuite-lang4sim2real/robosuite/scripts/collect_demonstrations_parallel.py --robots Panda --environment PPObjToPotToStove --device scripted-policy --noise-std 0.05 -n 5600 -p 56 --task-idx-intervals 0-3 --directory [.../data_collection_out_dir] --camera agentview --img-dim 128 --state-mode 1 --multitask-hdf5-format --intra-thread-delay 40
```

###### Target Domain
```
python robosuite-lang4sim2real/robosuite/scripts/collect_demonstrations_parallel.py --robots Panda --environment PPObjToPotToStove_ang1_fr5damp50 --device scripted-policy --noise-std 0.05 -n 400 -p 20 --task-idx-intervals 0-3 --directory [.../data_collection_out_dir] --camera agentview --img-dim 128 --state-mode 1 --multitask-hdf5-format --intra-thread-delay 1
```

##### Wire Wrap
###### Source Domain
```
python robosuite-lang4sim2real/robosuite/scripts/collect_demonstrations_parallel.py --robots Panda --environment WrapUnattachedWire_v2 --device scripted-policy --noise-std 0.05 -n 800 -p 20 --task-idx-intervals 0-1 --directory [.../data_collection_out_dir] --camera agentview --img-dim 128 --state-mode 1 --multitask-hdf5-format --intra-thread-delay 5 --policy wrap-relative-location
```

###### Target Domain
```
python robosuite-lang4sim2real/robosuite/scripts/collect_demonstrations_parallel.py --robots Panda --environment WrapUnattachedWire_ang1_fr5damp50_v2 --device scripted-policy --noise-std 0.05 -n 200 -p 10 --task-idx-intervals 0-1 --directory [.../data_collection_out_dir] --camera agentview --img-dim 128 --state-mode 1 --multitask-hdf5-format --policy wrap-relative-location --save-video --intra-thread-delay 1
```

#### Real
##### Pick-and-Place
```
python deoxys-lang4sim2real/deoxys/scripts/data_collection.py --out-dir [.../data_collection_out_dir] --policy pick_place --env frka_pp --horiz 18 --noise 0.05 --obj-id 1 --num 1 --state-mode 1 --substeps-per-step 1
```
##### 2-step Pick-and-Place
```
python deoxys-lang4sim2real/deoxys/scripts/data_collection.py --out-dir [.../data_collection_out_dir] --policy pick_place_n --env frka_obj_bowl_plate --horiz 45 --noise 0.05 --obj-id 6 --num 2 --state-mode 1 --substeps-per-step 1 --multistep-env
```
##### Wire Wrap
```
python deoxys-lang4sim2real/deoxys/scripts/data_collection.py --out-dir [.../data_collection_out_dir] --policy wrap_wire --env frka_wirewrap --horiz 45 --noise 0.05 --obj-id 2 --num 2 --state-mode 1 --substeps-per-step 1
```

#### Concatenating buffer command
```
python robosuite-lang4sim2real/robosuite/scripts/concat_hdf5.py -e [env_name] -p [buffer1_path] [buffer2_path] ... -d [out_dir_path] --concat-mode relabel-task-idx
```
- There are two concat-modes. `relabel-task-idx` gives each task in each buffer a new task idx, starting from 0. For instance, if buffer1 contained tasks 0-3 and buffer2 contained tasks 1-2, then the output buffer would contain tasks 0-5 (where buffer2's tasks 1-2 get mapped to tasks 4-5 in the output buffer). `merge-on-task-idx` combines all the demos in each buffer with the same task-idx under that task-idx in the output buffer.

#### Running VLM-based Automated Stage Labeler
1. Train gripper state predictor
```
python train-lang4sim2real/rlkit/lang4sim2real_utils/auto_captioner/train_gripper_state_pred.py --img-dir [.../2pp_sim2real.hdf5] --batch_size 256 --lr 0.02 --dom1-num-demos-per-task 100 --dom1-task-idxs 0-0 --num-epochs 100 --out-dir [.../out_dir]
```
- We expect `gripper_state_loss` to end up around 0.33, `ee_pos_l1_err` to end up around 0.049, and `gripper_classif_acc` to be 0.98.

2. Set up [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) so that you can run `from groundingdino.util.inference import load_model`.

3. Run automatic stage labeler with the checkpoint from step 1 to get a buffer that has `pred_stage_num` alongside `lang_stage_num`.
```
python train-lang4sim2real/rlkit/lang4sim2real_utils/auto_captioner/object_detector_labeling.py ----gdino-path [.../parent_dir_of_gdino] --buffer-path [.../.hdf5] --gripper-state-pred-model [.../.pt from step 1]
```



## Step 2. Pretrain Policy CNN
### Option A: Download Our Pretrained Checkpoints
Download the [Pretrained ResNet-18 checkpoints](https://utexas.box.com/s/mavrgdf541zivoxd18q6xg8ulozqt7nw) we used for our experiments.

Each checkpoint file is named with three attributes:
- task: {1pp, 2pp, ww} for pick-and-place, 2-step pick-and-place, and wire wrap.
- setting: {sim2real, sim2sim}
- method: {lang-reg, lang-dist, stage-classif}

### Option B: Pretrain Your Own Checkpoint
All commands shown below are language regression variant. See [here](#running-language-distance-pretraining) for running the language distance variant, and [here](#running-stage-classification-ablation) for running stage classification ablation.

#### Pick-and-Place
##### Sim2Real
Running this command requires downloading [.../1pp_pretrain_real_prior-task.hdf5](https://utexas.box.com/s/9j3jft7eb7d6vwxs2mlcdctbttxeyq2u). Note that this trains with the real world prior task (pick-place paper box) instead of the target task (pick-place carrot).
```
python train-lang4sim2real/rlkit/lang4sim2real_utils/train/train_policy_cnn_lang4sim2real.py --dom1-img-dir [.../1pp_sim2real.hdf5] --dom1-task-idxs 0-3 --dom1-num-demos-per-task 100 --dom2-img-dir [.../1pp_pretrain_real_prior-task.hdf5] --dom2-task-idxs 0-0 --dom2-num-demos-per-task 50 --batch-size 256 --num-epochs 150 --lr 0.04 --out-dir [.../phase1_out_dir] --img-aug pad_crop --pad-size 12 --variant lang-reg --save-ckpt-freq 50 --shuffle-demos
```


##### Sim2Sim
```
python train-lang4sim2real/rlkit/lang4sim2real_utils/train/train_policy_cnn_lang4sim2real.py --dom1-img-dir [.../1pp_sim2sim.hdf5] --dom1-task-idxs 0-3 --dom1-num-demos-per-task 100 --dom2-img-dir [.../1pp_sim2sim.hdf5] --dom2-task-idxs 7-7 --dom2-num-demos-per-task 100 --batch-size 256 --num-epochs 150 --lr 0.04 --out-dir [.../phase1_out_dir] --img-aug pad_crop --pad-size 12 --variant lang-reg --save-ckpt-freq 50 --shuffle-demos
```

#### 2-step Pick-and-Place
##### Sim2Real
Running this command requires downloading [.../2pp_pretrain_real_target-task.hdf5](https://utexas.box.com/s/05mzpapqrwvukfu5jgn8ujx7fnz9sqr2).
```
python train-lang4sim2real/rlkit/lang4sim2real_utils/train/train_policy_cnn_lang4sim2real.py --dom1-img-dir [.../2pp_sim2real.hdf5] --dom1-task-idxs 0-3 --dom1-num-demos-per-task 100 --dom2-img-dir [.../2pp_pretrain_real_target-task.hdf5] --dom2-task-idxs 0-0 --dom2-num-demos-per-task 100 --batch-size 256 --num-epochs 150 --lr 0.04 --out-dir [.../phase1_out_dir] --img-aug pad_crop --pad-size 12 --variant lang-reg --save-ckpt-freq 50 --shuffle-demos
```
##### Sim2Sim
```
python train-lang4sim2real/rlkit/lang4sim2real_utils/train/train_policy_cnn_lang4sim2real.py --dom1-img-dir [.../2pp_sim2sim.hdf5] --dom1-task-idxs 0-3 --dom1-num-demos-per-task 100 --dom2-img-dir [.../2pp_sim2sim.hdf5] --dom2-task-idxs 7-7 --dom2-num-demos-per-task 100 --batch-size 256 --num-epochs 150 --lr 0.04 --out-dir [.../phase1_out_dir] --img-aug pad_crop --pad-size 12 --variant lang-reg --save-ckpt-freq 50 --shuffle-demos
```

#### Wrap Wire

##### Sim2Real
Running this command requires downloading [.../ww_pretrain_real_target-task.hdf5](https://utexas.box.com/s/a78igt6pyoiq27bu0tmmdz8yq836o9k2).
```
python train-lang4sim2real/rlkit/lang4sim2real_utils/train/train_policy_cnn_lang4sim2real.py --dom1-img-dir [.../ww_sim2real.hdf5] --dom1-task-idxs 0-0 --dom1-num-demos-per-task 100 --dom2-img-dir [.../ww_pretrain_real_target-task.hdf5] --dom2-task-idxs 0-0 --dom2-num-demos-per-task 100 --batch-size 256 --num-epochs 150 --lr 0.04 --out-dir [.../phase1_out_dir] --img-aug pad_crop --pad-size 12 --variant lang-reg --save-ckpt-freq 50 --shuffle-demos
```
##### Sim2Sim
```
python train-lang4sim2real/rlkit/lang4sim2real_utils/train/train_policy_cnn_lang4sim2real.py --dom1-img-dir [.../ww_sim2sim.hdf5] --dom1-task-idxs 0-1 --dom1-num-demos-per-task 100 --dom2-img-dir [.../ww_sim2sim.hdf5] --dom2-task-idxs 3-3 --dom2-num-demos-per-task 100 --batch-size 256 --num-epochs 50 --lr 0.04 --out-dir [.../phase1_out_dir] --img-aug pad_crop --pad-size 12 --variant lang-reg --save-ckpt-freq 50 --shuffle-demos
```

#### Running language distance (BLEURT) variant
To pretrain with the language distance variant, you will need precomputed BLEURT score matrices [in this folder](./train-lang4sim2real/bleurt_matrices), or listed by experiment:

- **Sim2Real**: [1pp](./train-lang4sim2real/bleurt_matrices/Multitaskv2_0-3_frka_pp_0-0.npy), [2pp](./train-lang4sim2real/bleurt_matrices/PPObjToPotToStove_0-3_frka_obj_bowl_plate_0-0.npy), [ww](./train-lang4sim2real/bleurt_matrices/WrapUnattachedWire_0-0_frka_wirewrap_0-0.npy)
- **Sim2Sim**: [1pp](./train-lang4sim2real/bleurt_matrices/Multitaskv2_0-3_Multitaskv2_ang1_fr5damp50_3-3.npy), [2pp](./train-lang4sim2real/bleurt_matrices/PPObjToPotToStove_0-3_PPObjToPotToStove_ang1_fr5damp50_3-3.npy), [ww](./train-lang4sim2real/bleurt_matrices/WrapUnattachedWire_v2_0-1_WrapUnattachedWire_ang1_fr5damp50_v2_1-1.npy)

##### Compute Your Own BLEURT Similarity Score Matrices
If you would like to change the language annotations at each stage of the trajectory (which are stored in the hdf5 datasets as an attribute under each task idx), you can recompute the BLEURT score matrices:
```
python train-lang4sim2real/rlkit/plot/plot_bleurt_dist.py --dom1-img-dir [hdf5_path1]
--dom1-task-idxs 0-1 --dom2-img-dir [hdf5_path2]
--dom2-task-idxs 1-1 --batch-size 256
```
Change the `--dom*-task-idxs` flags as appropriate.

##### Running Language Distance Pretraining
Then add the flags `--variant lang-dist-dotprod --loss-arg-mult 40.0 --target-diff-mat-path [PATH_TO_BLEURT_TARGET_DIFF_MAT]` to the CNN pretraining command, where `[PATH_TO_BLEURT_TARGET_DIFF_MAT]` is the path of the BLEURT score matrix downloaded/computed above, and change `--variant lang-reg` to `--variant lang-dist-dotprod`.

##### Running Stage Classification Ablation
Change the flag: `--variant lang-reg` to `--variant stage-classif`. Keep `--lr 0.04`.




## Step 3. Train Policies with Multi-task, Multi-domain BC
### Pick-and-Place
#### Sim2Real
##### Ours
```
python train-lang4sim2real/experiments/multitask_bc.py --train-target-buffers [.../1pp_sim2real.hdf5] --xdomain-buffer-envs Multitaskv2 frka_pp --xdomain-env-instruct-prefixes Multitaskv2:Simulation frka_pp:Real --train-target-task-idx-intervals 0-4 --eval-task-idx-intervals 0-0 --max-path-len 200 18 --batch-size 57 --meta-batch-size 4 --task-emb-input-mode film --policy-num-film-inputs 1 --env frka_pp --realrobot-target-obj=carrot --num-tasks 8 --num-train-target-demos-per-task 400 --focus-train-task-idx-intervals 4-4 --focus-train-tasks-sample-prob 0.2 --num-focus-train-demos-per-task 100 --policy-resnet-conv-strides=2,2,1,1,1 --policy-cnn-ckpt [.../phase1_cnn_ckpt.pt] --policy-cnn-ckpt-unfrozen-mods film+cnnlastlayer --save-checkpoint-freq 50 --gpu 0 --num-epochs 300 --seed 310
```
##### No Pretrain (real)
```
python train-lang4sim2real/experiments/multitask_bc.py --train-target-buffers [.../1pp_sim2real.hdf5] --xdomain-buffer-envs Multitaskv2 frka_pp --xdomain-env-instruct-prefixes Multitaskv2:Simulation frka_pp:Real --train-target-task-idx-intervals 4-4 --eval-task-idx-intervals 0-0 --max-path-len 200 18 --batch-size 57 --meta-batch-size 4 --task-emb-input-mode film --policy-num-film-inputs 1 --env frka_pp --realrobot-target-obj=carrot --num-tasks 8 --num-train-target-demos-per-task 100 --policy-resnet-conv-strides=2,2,1,1,1 --save-checkpoint-freq 50 --gpu 0 --num-epochs 300 --seed 110
```
##### No Pretrain (sim+real)
```
python train-lang4sim2real/experiments/multitask_bc.py --train-target-buffers [.../1pp_sim2real.hdf5] --xdomain-buffer-envs Multitaskv2 frka_pp --xdomain-env-instruct-prefixes Multitaskv2:Simulation frka_pp:Real --train-target-task-idx-intervals 0-4 --eval-task-idx-intervals 0-0 --max-path-len 200 18 --batch-size 57 --meta-batch-size 4 --task-emb-input-mode film --policy-num-film-inputs 1 --env frka_pp --realrobot-target-obj=carrot --num-tasks 8 --num-train-target-demos-per-task 400 --focus-train-task-idx-intervals 4-4 --focus-train-tasks-sample-prob 0.2 --num-focus-train-demos-per-task 100 --policy-resnet-conv-strides=2,2,1,1,1 --save-checkpoint-freq 50 --gpu 0 --num-epochs 300 --seed 210
```

##### CLIP
```
python train-lang4sim2real/experiments/multitask_bc.py --train-target-buffers [.../1pp_sim2real.hdf5] --xdomain-buffer-envs Multitaskv2 frka_pp --xdomain-env-instruct-prefixes Multitaskv2:Simulation frka_pp:Real --train-target-task-idx-intervals 0-4 --eval-task-idx-intervals 0-0 --max-path-len 200 18 --batch-size 57 --meta-batch-size 4 --task-emb-input-mode concat_to_img_embs --env frka_pp --realrobot-target-obj=carrot --num-tasks 8 --num-train-target-demos-per-task 400 --focus-train-task-idx-intervals 4-4 --focus-train-tasks-sample-prob 0.2 --num-focus-train-demos-per-task 100 --policy-cnn-type clip --clip-ckpt=[.../clip_ckpt.pt]  --freeze-clip --save-checkpoint-freq 250 --gpu 0 --num-epochs 300 --seed 610
```

##### R3M
```
python train-lang4sim2real/experiments/multitask_bc.py --train-target-buffers [.../1pp_sim2real.hdf5] --xdomain-buffer-envs Multitaskv2 frka_pp --xdomain-env-instruct-prefixes Multitaskv2:Simulation frka_pp:Real --train-target-task-idx-intervals 0-4 --eval-task-idx-intervals 0-0 --max-path-len 200 18 --batch-size 57 --meta-batch-size 4 --task-emb-input-mode concat_to_img_embs --env frka_pp --realrobot-target-obj=carrot --num-tasks 8 --num-train-target-demos-per-task 400 --focus-train-task-idx-intervals 4-4 --focus-train-tasks-sample-prob 0.2 --num-focus-train-demos-per-task 100 --policy-cnn-type r3m --freeze-policy-cnn --save-checkpoint-freq 50 --gpu 0 --num-epochs 300 --seed 710
```

##### MMD
```
python train-lang4sim2real/experiments/multitask_bc.py --train-target-buffers [.../1pp_sim2real.hdf5] --xdomain-buffer-envs Multitaskv2 frka_pp --xdomain-env-instruct-prefixes Multitaskv2:Simulation frka_pp:Real --train-target-task-idx-intervals 0-4 --eval-task-idx-intervals 0-0 --max-path-len 200 18 --batch-size 57 --meta-batch-size 4 --task-emb-input-mode film --policy-num-film-inputs 1 --env frka_pp --realrobot-target-obj=carrot --num-tasks 8 --num-train-target-demos-per-task 400 --focus-train-task-idx-intervals 4-4 --focus-train-tasks-sample-prob 0.2 --num-focus-train-demos-per-task 100 --mmd-coefficient 0.01 --policy-resnet-conv-strides=2,2,1,1,1 --save-checkpoint-freq 50 --gpu 0 --num-epochs 300 --seed 1440
```

##### Domain Randomization
```
python train-lang4sim2real/experiments/multitask_bc.py --train-target-buffers [.../1pp_domain-rando_sim2real.hdf5] --xdomain-buffer-envs Multitaskv2 frka_pp --xdomain-env-instruct-prefixes Multitaskv2:Simulation frka_pp:Real --train-target-task-idx-intervals 0-4 --eval-task-idx-intervals 0-0 --max-path-len 200 18 --batch-size 57 --meta-batch-size 4 --task-emb-input-mode film --policy-num-film-inputs 1 --env frka_pp --realrobot-target-obj=carrot --num-tasks 8 --num-train-target-demos-per-task 400 --focus-train-task-idx-intervals 4-4 --focus-train-tasks-sample-prob 0.2 --num-focus-train-demos-per-task 100 --policy-resnet-conv-strides=2,2,1,1,1 --save-checkpoint-freq 50 --gpu 0 --num-epochs 300 --seed 1640
```
##### ADR+RNA
```
python train-lang4sim2real/experiments/multitask_bc.py --train-target-buffers [.../1pp_adr-rna_sim2real.hdf5] --xdomain-buffer-envs Multitaskv2 frka_pp --xdomain-env-instruct-prefixes Multitaskv2:Simulation frka_pp:Real --train-target-task-idx-intervals 0-4 --eval-task-idx-intervals 0-0 --max-path-len 200 18 --batch-size 57 --meta-batch-size 4 --task-emb-input-mode film --policy-num-film-inputs 1 --env frka_pp --realrobot-target-obj=carrot --num-tasks 8 --num-train-target-demos-per-task 400 --focus-train-task-idx-intervals 4-4 --focus-train-tasks-sample-prob 0.2 --num-focus-train-demos-per-task 100 --policy-resnet-conv-strides=2,2,1,1,1 --save-checkpoint-freq 50 --gpu 0 --num-epochs 300 --seed 1340
```
#### Sim2Sim (target domain, prior task data)
##### Ours
```
python train-lang4sim2real/experiments/multitask_bc.py --train-target-buffers [.../1pp_sim2sim.hdf5] --xdomain-buffer-envs Multitaskv2 Multitaskv2_ang1_fr5damp50 --xdomain-env-instruct-prefixes Multitaskv2:Simulation Multitaskv2_ang1_fr5damp50:Real --train-target-task-idx-intervals 0-4 --eval-task-idx-intervals 0-0 --max-path-len 200 200 --batch-size 57 --meta-batch-size 4 --task-emb-input-mode film --policy-num-film-inputs 1 --env Multitaskv2_ang1_fr5damp50 --realrobot-target-obj="" --num-tasks 8 --num-train-target-demos-per-task 400 --focus-train-task-idx-intervals 4-4 --focus-train-tasks-sample-prob 0.2 --num-focus-train-demos-per-task 100 --policy-resnet-conv-strides=2,2,1,1,1 --policy-cnn-ckpt [.../phase1_cnn_ckpt.pt] --policy-cnn-ckpt-unfrozen-mods film+cnnlastlayer --save-checkpoint-freq 1000 --gpu 0 --num-epochs 500 --seed 460
```

##### No Pretrain (real)
```
python train-lang4sim2real/experiments/multitask_bc.py --train-target-buffers [.../1pp_sim2sim.hdf5] --xdomain-buffer-envs Multitaskv2 Multitaskv2_ang1_fr5damp50 --xdomain-env-instruct-prefixes Multitaskv2:Simulation Multitaskv2_ang1_fr5damp50:Real --train-target-task-idx-intervals 4-4 7-7 --eval-task-idx-intervals 0-0 --max-path-len 200 200 --batch-size 57 --meta-batch-size 4 --task-emb-input-mode film --policy-num-film-inputs 1 --env Multitaskv2_ang1_fr5damp50 --realrobot-target-obj="" --num-tasks 8 --num-train-target-demos-per-task 100 --policy-resnet-conv-strides=2,2,1,1,1 --save-checkpoint-freq 1000 --gpu 0 --num-epochs 500 --seed 160
```

##### No Pretrain (sim+real)
```
python train-lang4sim2real/experiments/multitask_bc.py --train-target-buffers [.../1pp_sim2sim.hdf5] --xdomain-buffer-envs Multitaskv2 Multitaskv2_ang1_fr5damp50 --xdomain-env-instruct-prefixes Multitaskv2:Simulation Multitaskv2_ang1_fr5damp50:Real --train-target-task-idx-intervals 0-4 7-7 --eval-task-idx-intervals 0-0 --max-path-len 200 200 --batch-size 57 --meta-batch-size 4 --task-emb-input-mode film --policy-num-film-inputs 1 --env Multitaskv2_ang1_fr5damp50 --realrobot-target-obj="" --num-tasks 8 --num-train-target-demos-per-task 400 --focus-train-task-idx-intervals 4-4 7-7 --focus-train-tasks-sample-prob 0.333 --num-focus-train-demos-per-task 100 --policy-resnet-conv-strides=2,2,1,1,1 --save-checkpoint-freq 1000 --gpu 0 --num-epochs 500 --seed 260
```

##### CLIP
```
python train-lang4sim2real/experiments/multitask_bc.py --train-target-buffers [.../1pp_sim2sim.hdf5] --xdomain-buffer-envs Multitaskv2 Multitaskv2_ang1_fr5damp50 --xdomain-env-instruct-prefixes Multitaskv2:Simulation Multitaskv2_ang1_fr5damp50:Real --train-target-task-idx-intervals 0-4 7-7 --eval-task-idx-intervals 0-0 --max-path-len 200 200 --batch-size 57 --meta-batch-size 4 --task-emb-input-mode concat_to_img_embs --env Multitaskv2_ang1_fr5damp50 --realrobot-target-obj="" --num-tasks 8 --num-train-target-demos-per-task 400 --focus-train-task-idx-intervals 4-4 7-7 --focus-train-tasks-sample-prob 0.333 --num-focus-train-demos-per-task 100 --policy-cnn-type clip --clip-ckpt=[.../clip_ckpt.pt]  --freeze-clip --gpu 0 --num-epochs 500 --seed 660
```

##### R3M
```
python train-lang4sim2real/experiments/multitask_bc.py --train-target-buffers [.../1pp_sim2sim.hdf5] --xdomain-buffer-envs Multitaskv2 Multitaskv2_ang1_fr5damp50 --xdomain-env-instruct-prefixes Multitaskv2:Simulation Multitaskv2_ang1_fr5damp50:Real --train-target-task-idx-intervals 0-4 7-7 --eval-task-idx-intervals 0-0 --max-path-len 200 200 --batch-size 57 --meta-batch-size 4 --task-emb-input-mode concat_to_img_embs --env Multitaskv2_ang1_fr5damp50 --realrobot-target-obj="" --num-tasks 8 --num-train-target-demos-per-task 400 --focus-train-task-idx-intervals 4-4 7-7 --focus-train-tasks-sample-prob 0.333 --num-focus-train-demos-per-task 100 --policy-cnn-type r3m --freeze-policy-cnn --gpu 0 --num-epochs 500 --seed 760
```


### 2-step Pick-and-Place
#### Sim2Real
##### Ours
```
python train-lang4sim2real/experiments/multitask_bc.py --train-target-buffers [.../2pp_sim2real.hdf5] --xdomain-buffer-envs PPObjToPotToStove frka_obj_bowl_plate --xdomain-env-instruct-prefixes PPObjToPotToStove:Simulation frka_obj_bowl_plate:Real --train-target-task-idx-intervals 0-4 --eval-task-idx-intervals 0-0 --max-path-len 320 45 --batch-size 57 --meta-batch-size 4 --task-emb-input-mode film --policy-num-film-inputs 1 --env frka_obj_bowl_plate --realrobot-target-obj carrot  --num-tasks 6 --num-train-target-demos-per-task 1375 --focus-train-task-idx-intervals 4-4 --focus-train-tasks-sample-prob 0.2 --num-focus-train-demos-per-task 100 --policy-resnet-conv-strides=2,2,1,1,1 --policy-cnn-ckpt [.../phase1_cnn_ckpt.pt] --policy-cnn-ckpt-unfrozen-mods film+cnnlastlayer --save-checkpoint-freq 50 --gpu 0 --num-epochs 600 --seed 110
```

##### No Pretrain (real)
```
python train-lang4sim2real/experiments/multitask_bc.py --train-target-buffers [.../2pp_sim2real.hdf5] --xdomain-buffer-envs PPObjToPotToStove frka_obj_bowl_plate --xdomain-env-instruct-prefixes PPObjToPotToStove:Simulation frka_obj_bowl_plate:Real --train-target-task-idx-intervals 4-4 --eval-task-idx-intervals 0-0 --max-path-len 320 45 --batch-size 57 --meta-batch-size 4 --task-emb-input-mode film --policy-num-film-inputs 1 --env frka_obj_bowl_plate --realrobot-target-obj carrot  --num-tasks 6 --num-train-target-demos-per-task 100 --policy-resnet-conv-strides=2,2,1,1,1 --save-checkpoint-freq 50 --gpu 0 --num-epochs 600 --seed 110
```

##### No Pretrain (sim+real)
```
python train-lang4sim2real/experiments/multitask_bc.py --train-target-buffers [.../2pp_sim2real.hdf5] --xdomain-buffer-envs PPObjToPotToStove frka_obj_bowl_plate --xdomain-env-instruct-prefixes PPObjToPotToStove:Simulation frka_obj_bowl_plate:Real --train-target-task-idx-intervals 0-4 --eval-task-idx-intervals 0-0 --max-path-len 320 45 --batch-size 57 --meta-batch-size 4 --task-emb-input-mode film --policy-num-film-inputs 1 --env frka_obj_bowl_plate --realrobot-target-obj carrot  --num-tasks 6 --num-train-target-demos-per-task 1375 --focus-train-task-idx-intervals 4-4 --focus-train-tasks-sample-prob 0.2 --num-focus-train-demos-per-task 100 --policy-resnet-conv-strides=2,2,1,1,1 --save-checkpoint-freq 50 --gpu 0 --num-epochs 600 --seed 210
```

##### CLIP
```
python train-lang4sim2real/experiments/multitask_bc.py --train-target-buffers [.../2pp_sim2real.hdf5] --xdomain-buffer-envs PPObjToPotToStove frka_obj_bowl_plate --xdomain-env-instruct-prefixes PPObjToPotToStove:Simulation frka_obj_bowl_plate:Real --train-target-task-idx-intervals 0-4 --eval-task-idx-intervals 0-0 --max-path-len 320 45 --batch-size 57 --meta-batch-size 4 --task-emb-input-mode concat_to_img_embs --env frka_obj_bowl_plate --realrobot-target-obj=carrot --num-tasks 6 --num-train-target-demos-per-task 1400 --focus-train-task-idx-intervals 4-4 --focus-train-tasks-sample-prob 0.2 --num-focus-train-demos-per-task 100 --policy-cnn-type clip --clip-ckpt=[.../clip_ckpt.pt]  --freeze-clip --save-checkpoint-freq 250 --gpu 0 --num-epochs 600 --seed 310
```

##### R3M
```
python train-lang4sim2real/experiments/multitask_bc.py --train-target-buffers [.../2pp_sim2real.hdf5] --xdomain-buffer-envs PPObjToPotToStove frka_obj_bowl_plate --xdomain-env-instruct-prefixes PPObjToPotToStove:Simulation frka_obj_bowl_plate:Real --train-target-task-idx-intervals 0-4 --eval-task-idx-intervals 0-0 --max-path-len 320 45 --batch-size 57 --meta-batch-size 4 --task-emb-input-mode concat_to_img_embs --env frka_obj_bowl_plate --realrobot-target-obj=carrot --num-tasks 6 --num-train-target-demos-per-task 1400 --focus-train-task-idx-intervals 4-4 --focus-train-tasks-sample-prob 0.2 --num-focus-train-demos-per-task 100 --policy-cnn-type r3m --freeze-policy-cnn --save-checkpoint-freq 50 --gpu 0 --num-epochs 600 --seed 310
```

##### MMD
```
python train-lang4sim2real/experiments/multitask_bc.py --train-target-buffers [.../2pp_sim2real.hdf5] --xdomain-buffer-envs PPObjToPotToStove frka_obj_bowl_plate --xdomain-env-instruct-prefixes PPObjToPotToStove:Simulation frka_obj_bowl_plate:Real --train-target-task-idx-intervals 0-4 --eval-task-idx-intervals 0-0 --max-path-len 320 45 --batch-size 57 --meta-batch-size 4 --task-emb-input-mode film --policy-num-film-inputs 1 --env frka_obj_bowl_plate --realrobot-target-obj carrot  --num-tasks 6 --num-train-target-demos-per-task 1375 --focus-train-task-idx-intervals 4-4 --focus-train-tasks-sample-prob 0.2 --num-focus-train-demos-per-task 100 --policy-resnet-conv-strides=2,2,1,1,1 --mmd-coefficient 0.01 --save-checkpoint-freq 50 --gpu 0 --num-epochs 600 --seed 220
```

##### Domain Randomization
```
python train-lang4sim2real/experiments/multitask_bc.py --train-target-buffers [.../2pp_domain-rando_sim2real.hdf5] --xdomain-buffer-envs PPObjToPotToStove frka_obj_bowl_plate --xdomain-env-instruct-prefixes PPObjToPotToStove:Simulation frka_obj_bowl_plate:Real --train-target-task-idx-intervals 0-4 --eval-task-idx-intervals 0-0 --max-path-len 320 45 --batch-size 57 --meta-batch-size 4 --task-emb-input-mode film --policy-num-film-inputs 1 --env frka_obj_bowl_plate --realrobot-target-obj=carrot --num-tasks 8 --num-train-target-demos-per-task 1400 --focus-train-task-idx-intervals 4-4 --focus-train-tasks-sample-prob 0.2 --num-focus-train-demos-per-task 100 --policy-resnet-conv-strides=2,2,1,1,1 --save-checkpoint-freq 50 --gpu 0 --num-epochs 600 --seed 1112
```

##### ADR+RNA
```
python train-lang4sim2real/experiments/multitask_bc.py --train-target-buffers [.../2pp_adr-rna_sim2real.hdf5] --xdomain-buffer-envs PPObjToPotToStove frka_obj_bowl_plate --xdomain-env-instruct-prefixes PPObjToPotToStove:Simulation frka_obj_bowl_plate:Real --train-target-task-idx-intervals 0-4 --eval-task-idx-intervals 0-0 --max-path-len 320 45 --batch-size 57 --meta-batch-size 4 --task-emb-input-mode film --policy-num-film-inputs 1 --env frka_obj_bowl_plate --realrobot-target-obj=carrot --num-tasks 8 --num-train-target-demos-per-task 1400 --focus-train-task-idx-intervals 4-4 --focus-train-tasks-sample-prob 0.2 --num-focus-train-demos-per-task 100 --policy-resnet-conv-strides=2,2,1,1,1 --save-checkpoint-freq 50 --gpu 0 --num-epochs 500 --seed 2012
```

#### Sim2Sim (target domain, prior task data)
##### Ours
```
python train-lang4sim2real/experiments/multitask_bc.py --train-target-buffers [.../2pp_sim2sim.hdf5] --xdomain-buffer-envs PPObjToPotToStove PPObjToPotToStove_ang1_fr5damp50 --xdomain-env-instruct-prefixes PPObjToPotToStove:Simulation PPObjToPotToStove_ang1_fr5damp50:Real --train-target-task-idx-intervals 0-4 --eval-task-idx-intervals 0-0 --max-path-len 320 320 --batch-size 57 --meta-batch-size 4 --task-emb-input-mode film --policy-num-film-inputs 1 --env PPObjToPotToStove_ang1_fr5damp50 --realrobot-target-obj="" --num-tasks 8 --num-train-target-demos-per-task 1400 --focus-train-task-idx-intervals 4-4 --focus-train-tasks-sample-prob 0.2 --num-focus-train-demos-per-task 100 --policy-resnet-conv-strides=2,2,1,1,1 --policy-cnn-ckpt [.../phase1_cnn_ckpt.pt] --policy-cnn-ckpt-unfrozen-mods film+cnnlastlayer --save-checkpoint-freq 1000 --gpu 0 --num-epochs 600 --seed 420
```

##### No Pretrain (real)
```
python train-lang4sim2real/experiments/multitask_bc.py --train-target-buffers [.../2pp_sim2sim.hdf5] --xdomain-buffer-envs PPObjToPotToStove PPObjToPotToStove_ang1_fr5damp50 --xdomain-env-instruct-prefixes PPObjToPotToStove:Simulation PPObjToPotToStove_ang1_fr5damp50:Real --train-target-task-idx-intervals 4-4 7-7 --eval-task-idx-intervals 0-0 --max-path-len 320 320 --batch-size 57 --meta-batch-size 4 --task-emb-input-mode film --policy-num-film-inputs 1 --env PPObjToPotToStove_ang1_fr5damp50 --realrobot-target-obj="" --num-tasks 8 --num-train-target-demos-per-task 100 --policy-resnet-conv-strides=2,2,1,1,1 --save-checkpoint-freq 1000 --gpu 0 --num-epochs 500 --seed 120
```

##### No Pretrain (sim+real)
```
python train-lang4sim2real/experiments/multitask_bc.py --train-target-buffers [.../2pp_sim2sim.hdf5] --xdomain-buffer-envs PPObjToPotToStove PPObjToPotToStove_ang1_fr5damp50 --xdomain-env-instruct-prefixes PPObjToPotToStove:Simulation PPObjToPotToStove_ang1_fr5damp50:Real --train-target-task-idx-intervals 0-4 7-7 --eval-task-idx-intervals 0-0 --max-path-len 320 320 --batch-size 57 --meta-batch-size 4 --task-emb-input-mode film --policy-num-film-inputs 1 --env PPObjToPotToStove_ang1_fr5damp50 --realrobot-target-obj="" --num-tasks 8 --num-train-target-demos-per-task 1400 --focus-train-task-idx-intervals 4-4 7-7 --focus-train-tasks-sample-prob 0.333 --num-focus-train-demos-per-task 100 --policy-resnet-conv-strides=2,2,1,1,1 --save-checkpoint-freq 1000 --gpu 0 --num-epochs 600 --seed 220
```

##### CLIP
```
python train-lang4sim2real/experiments/multitask_bc.py --train-target-buffers [.../2pp_sim2sim.hdf5] --xdomain-buffer-envs PPObjToPotToStove PPObjToPotToStove_ang1_fr5damp50 --xdomain-env-instruct-prefixes PPObjToPotToStove:Simulation PPObjToPotToStove_ang1_fr5damp50:Real --train-target-task-idx-intervals 0-4 7-7 --eval-task-idx-intervals 0-0 --max-path-len 320 320 --batch-size 57 --meta-batch-size 4 --task-emb-input-mode concat_to_img_embs --env PPObjToPotToStove_ang1_fr5damp50 --realrobot-target-obj="" --num-tasks 8 --num-train-target-demos-per-task 1400 --focus-train-task-idx-intervals 4-4 7-7 --focus-train-tasks-sample-prob 0.333 --num-focus-train-demos-per-task 100 --policy-cnn-type clip --clip-ckpt=[.../clip_ckpt.pt]  --freeze-clip --gpu 0 --num-epochs 600 --seed 620
```

##### R3M
```
python train-lang4sim2real/experiments/multitask_bc.py --train-target-buffers [.../2pp_sim2sim.hdf5] --xdomain-buffer-envs PPObjToPotToStove PPObjToPotToStove_ang1_fr5damp50 --xdomain-env-instruct-prefixes PPObjToPotToStove:Simulation PPObjToPotToStove_ang1_fr5damp50:Real --train-target-task-idx-intervals 0-4 7-7 --eval-task-idx-intervals 0-0 --max-path-len 320 320 --batch-size 57 --meta-batch-size 4 --task-emb-input-mode concat_to_img_embs --env PPObjToPotToStove_ang1_fr5damp50 --realrobot-target-obj="" --num-tasks 8 --num-train-target-demos-per-task 1400 --focus-train-task-idx-intervals 4-4 7-7 --focus-train-tasks-sample-prob 0.333 --num-focus-train-demos-per-task 100 --policy-cnn-type r3m --freeze-policy-cnn --gpu 0 --num-epochs 600 --seed 720
```

### Wrap Wire
#### Sim2Real
##### Ours
```
python train-lang4sim2real/experiments/multitask_bc.py --train-target-buffers [.../ww_sim2real.hdf5] --xdomain-buffer-envs WrapUnattachedWire frka_wirewrap --xdomain-env-instruct-prefixes WrapUnattachedWire:Simulation frka_wirewrap:Real --train-target-task-idx-intervals 0-1 --eval-task-idx-intervals 0-0 --max-path-len 200 45 --batch-size 57 --meta-batch-size 4 --task-emb-input-mode film --policy-num-film-inputs 1 --env frka_wirewrap --realrobot-target-obj="eu white plug" --num-tasks 2 --num-train-target-demos-per-task 1000 --focus-train-task-idx-intervals 1-1 --focus-train-tasks-sample-prob 0.5 --num-focus-train-demos-per-task 100 --policy-resnet-conv-strides=2,2,1,1,1 --policy-cnn-ckpt [.../phase1_cnn_ckpt.pt] --policy-cnn-ckpt-unfrozen-mods film+cnnlastlayer --save-checkpoint-freq 50 --gpu 0 --num-epochs 600 --seed 410
```

##### No Pretrain (real)
```
python train-lang4sim2real/experiments/multitask_bc.py --train-target-buffers [.../ww_sim2real.hdf5] --xdomain-buffer-envs WrapUnattachedWire frka_wirewrap --xdomain-env-instruct-prefixes WrapUnattachedWire:Simulation frka_wirewrap:Real --train-target-task-idx-intervals 0-0 --eval-task-idx-intervals 0-0 --max-path-len 200 45 --batch-size 57 --meta-batch-size 4 --task-emb-input-mode film --policy-num-film-inputs 1 --env frka_wirewrap --realrobot-target-obj="eu white plug" --num-tasks 2 --num-train-target-demos-per-task 1000 --policy-resnet-conv-strides=2,2,1,1,1 --save-checkpoint-freq 50 --gpu 0 --num-epochs 600 --seed 110
```

##### No Pretrain (sim+real)
```
python train-lang4sim2real/experiments/multitask_bc.py --train-target-buffers [.../ww_sim2real.hdf5] --xdomain-buffer-envs WrapUnattachedWire frka_wirewrap --xdomain-env-instruct-prefixes WrapUnattachedWire:Simulation frka_wirewrap:Real --train-target-task-idx-intervals 0-1 --eval-task-idx-intervals 0-0 --max-path-len 200 45 --batch-size 57 --meta-batch-size 4 --task-emb-input-mode film --policy-num-film-inputs 1 --env frka_wirewrap --realrobot-target-obj="eu white plug" --num-tasks 2 --num-train-target-demos-per-task 1000 --focus-train-task-idx-intervals 1-1 --focus-train-tasks-sample-prob 0.5 --num-focus-train-demos-per-task 100 --policy-resnet-conv-strides=2,2,1,1,1 --save-checkpoint-freq 50 --gpu 0 --num-epochs 600 --seed 210
```

##### CLIP
```
python train-lang4sim2real/experiments/multitask_bc.py --train-target-buffers [.../ww_sim2real.hdf5] --xdomain-buffer-envs WrapUnattachedWire frka_wirewrap --xdomain-env-instruct-prefixes WrapUnattachedWire:Simulation frka_wirewrap:Real --train-target-task-idx-intervals 0-1 --eval-task-idx-intervals 0-0 --max-path-len 200 45 --batch-size 57 --meta-batch-size 4 --task-emb-input-mode concat_to_img_embs --env frka_wirewrap --realrobot-target-obj="eu white plug" --realrobot-obj-set 0 --num-tasks 3 --num-train-target-demos-per-task 1000 --focus-train-task-idx-intervals 1-1 --focus-train-tasks-sample-prob 0.5 --num-focus-train-demos-per-task 100 --policy-cnn-type clip --clip-ckpt=[.../clip_ckpt.pt]  --freeze-clip --save-checkpoint-freq 250 --gpu 0 --num-epochs 600 --seed 410
```

##### R3M
```
python train-lang4sim2real/experiments/multitask_bc.py --train-target-buffers [.../ww_sim2real.hdf5] --xdomain-buffer-envs WrapUnattachedWire frka_wirewrap --xdomain-env-instruct-prefixes WrapUnattachedWire:Simulation frka_wirewrap:Real --train-target-task-idx-intervals 0-1 --eval-task-idx-intervals 0-0 --max-path-len 200 45 --batch-size 57 --meta-batch-size 4 --task-emb-input-mode concat_to_img_embs --env frka_wirewrap --realrobot-target-obj="eu white plug" --realrobot-obj-set 0 --num-tasks 3 --num-train-target-demos-per-task 1000 --focus-train-task-idx-intervals 1-1 --focus-train-tasks-sample-prob 0.5 --num-focus-train-demos-per-task 100 --policy-cnn-type r3m --freeze-policy-cnn --save-checkpoint-freq 50 --gpu 0 --num-epochs 600 --seed 410
```

##### MMD
```
python train-lang4sim2real/experiments/multitask_bc.py --train-target-buffers [.../ww_sim2real.hdf5] --xdomain-buffer-envs WrapUnattachedWire frka_wirewrap --xdomain-env-instruct-prefixes WrapUnattachedWire:Simulation frka_wirewrap:Real --train-target-task-idx-intervals 0-1 --eval-task-idx-intervals 0-0 --max-path-len 200 45 --batch-size 57 --meta-batch-size 4 --task-emb-input-mode film --policy-num-film-inputs 1 --env frka_wirewrap --realrobot-target-obj="eu white plug" --num-tasks 2 --num-train-target-demos-per-task 1000 --focus-train-task-idx-intervals 1-1 --focus-train-tasks-sample-prob 0.5 --num-focus-train-demos-per-task 100 --policy-resnet-conv-strides=2,2,1,1,1 --save-checkpoint-freq 50 --mmd-coefficient 1e-4 --gpu 0 --num-epochs 600 --seed 210
```

##### Domain Randomization
```
python train-lang4sim2real/experiments/multitask_bc.py --train-target-buffers [.../ww_domain-rando_sim2real.hdf5] --xdomain-buffer-envs WrapUnattachedWire frka_wirewrap --xdomain-env-instruct-prefixes WrapUnattachedWire:Simulation frka_wirewrap:Real --train-target-task-idx-intervals 0-1 --eval-task-idx-intervals 0-0 --max-path-len 200 45 --batch-size 57 --meta-batch-size 4 --task-emb-input-mode film --policy-num-film-inputs 1 --env frka_wirewrap --realrobot-target-obj="eu white plug" --realrobot-obj-set 0 --num-tasks 5 --num-train-target-demos-per-task 1000 --focus-train-task-idx-intervals 1-1 --focus-train-tasks-sample-prob 0.333 --num-focus-train-demos-per-task 100 --policy-resnet-conv-strides=2,2,1,1,1 --save-checkpoint-freq 50 --gpu 0 --num-epochs 500 --seed 1140
```

##### ADR+RNA
```
python train-lang4sim2real/experiments/multitask_bc.py --train-target-buffers [.../ww_adr-rna_sim2real.hdf5] --xdomain-buffer-envs WrapUnattachedWire frka_wirewrap --xdomain-env-instruct-prefixes WrapUnattachedWire:Simulation frka_wirewrap:Real --train-target-task-idx-intervals 0-1 --eval-task-idx-intervals 0-0 --max-path-len 200 45 --batch-size 57 --meta-batch-size 4 --task-emb-input-mode film --policy-num-film-inputs 1 --env frka_wirewrap --realrobot-target-obj="eu white plug" --realrobot-obj-set 0 --num-tasks 5 --num-train-target-demos-per-task 1000 --focus-train-task-idx-intervals 1-1 --focus-train-tasks-sample-prob 0.5 --num-focus-train-demos-per-task 100 --policy-resnet-conv-strides=2,2,1,1,1 --save-checkpoint-freq 50 --gpu 0 --num-epochs 500 --seed 2040
```

#### Sim2Sim (target domain, prior task data)
##### Ours
```
python train-lang4sim2real/experiments/multitask_bc.py --train-target-buffers [.../ww_sim2sim.hdf5] --xdomain-buffer-envs WrapUnattachedWire_v2 WrapUnattachedWire_ang1_fr5damp50_v2 --xdomain-env-instruct-prefixes WrapUnattachedWire_v2:Simulation WrapUnattachedWire_ang1_fr5damp50_v2:Real --train-target-task-idx-intervals 0-2 --eval-task-idx-intervals 0-0 --max-path-len 200 200 --batch-size 57 --meta-batch-size 4 --task-emb-input-mode film --policy-num-film-inputs 1 --env WrapUnattachedWire_ang1_fr5damp50_v2 --realrobot-target-obj="" --num-tasks 4 --num-train-target-demos-per-task 400 --focus-train-task-idx-intervals 2-2 --focus-train-tasks-sample-prob 0.333 --num-focus-train-demos-per-task 100 --policy-resnet-conv-strides=2,2,1,1,1 --policy-cnn-ckpt /home/mini_exps/lang4sim2real/phase1/2024-01-28_19-56-38/best.pt  --policy-cnn-ckpt-unfrozen-mods film+cnnlastlayer --save-checkpoint-freq 1000 --gpu 0 --num-epochs 600 --seed 321
```

##### No Pretrain (real)
```
python train-lang4sim2real/experiments/multitask_bc.py --train-target-buffers [.../ww_sim2sim.hdf5] --xdomain-buffer-envs WrapUnattachedWire_v2 WrapUnattachedWire_ang1_fr5damp50_v2 --xdomain-env-instruct-prefixes WrapUnattachedWire_v2:Simulation WrapUnattachedWire_ang1_fr5damp50_v2:Real --train-target-task-idx-intervals 2-3 --eval-task-idx-intervals 0-0 --max-path-len 200 200 --batch-size 57 --meta-batch-size 4 --task-emb-input-mode film --policy-num-film-inputs 1 --env WrapUnattachedWire_ang1_fr5damp50_v2 --realrobot-target-obj="" --num-tasks 4 --num-train-target-demos-per-task 100 --policy-resnet-conv-strides=2,2,1,1,1 --save-checkpoint-freq 1000 --gpu 0 --num-epochs 500 --seed 110
```

##### No Pretrain (sim+real)
```
python train-lang4sim2real/experiments/multitask_bc.py --train-target-buffers [.../ww_sim2sim.hdf5] --xdomain-buffer-envs WrapUnattachedWire_v2 WrapUnattachedWire_ang1_fr5damp50_v2 --xdomain-env-instruct-prefixes WrapUnattachedWire_v2:Simulation WrapUnattachedWire_ang1_fr5damp50_v2:Real --train-target-task-idx-intervals 0-3 --eval-task-idx-intervals 0-0 --max-path-len 200 200 --batch-size 57 --meta-batch-size 4 --task-emb-input-mode film --policy-num-film-inputs 1 --env WrapUnattachedWire_ang1_fr5damp50_v2 --realrobot-target-obj="" --num-tasks 4 --num-train-target-demos-per-task 400 --focus-train-task-idx-intervals 2-3 --focus-train-tasks-sample-prob 0.5 --num-focus-train-demos-per-task 100 --policy-resnet-conv-strides=2,2,1,1,1 --save-checkpoint-freq 1000 --gpu 0 --num-epochs 600 --seed 210
```

##### CLIP
```
python train-lang4sim2real/experiments/multitask_bc.py --train-target-buffers [.../ww_sim2sim.hdf5] --xdomain-buffer-envs WrapUnattachedWire_v2 WrapUnattachedWire_ang1_fr5damp50_v2 --xdomain-env-instruct-prefixes WrapUnattachedWire_v2:Simulation WrapUnattachedWire_ang1_fr5damp50_v2:Real --train-target-task-idx-intervals 0-3 --eval-task-idx-intervals 0-0 --max-path-len 200 200 --batch-size 57 --meta-batch-size 4 --task-emb-input-mode concat_to_img_embs --env WrapUnattachedWire_ang1_fr5damp50_v2 --realrobot-target-obj="" --num-tasks 4 --num-train-target-demos-per-task 400 --focus-train-task-idx-intervals 2-3 --focus-train-tasks-sample-prob 0.5 --num-focus-train-demos-per-task 100 --policy-cnn-type clip --clip-ckpt=[.../clip_ckpt.pt]  --freeze-clip --gpu 0 --num-epochs 500 --seed 610
```

##### R3M
```
python train-lang4sim2real/experiments/multitask_bc.py --train-target-buffers [.../ww_sim2sim.hdf5] --xdomain-buffer-envs WrapUnattachedWire_v2 WrapUnattachedWire_ang1_fr5damp50_v2 --xdomain-env-instruct-prefixes WrapUnattachedWire_v2:Simulation WrapUnattachedWire_ang1_fr5damp50_v2:Real --train-target-task-idx-intervals 0-3 --eval-task-idx-intervals 0-0 --max-path-len 200 200 --batch-size 57 --meta-batch-size 4 --task-emb-input-mode concat_to_img_embs --env WrapUnattachedWire_ang1_fr5damp50_v2 --realrobot-target-obj="" --num-tasks 4 --num-train-target-demos-per-task 400 --focus-train-task-idx-intervals 2-3 --focus-train-tasks-sample-prob 0.5 --num-focus-train-demos-per-task 100 --policy-cnn-type r3m --freeze-policy-cnn --gpu 0 --num-epochs 500 --seed 710
```

## Step 4. Evaluate Policies
### Sim2Sim
Evaluation Metrics can be found in the experiment output folder, in `progress.csv` in the `eval/env_infos/final/reward Mean` CSV key.

You may find `python train-lang4sim2real/rlkit/plot/metric_calculator_by_split.py [list of .../exp_dir or parent of exp dirs]` useful for computing all successes and averaging them based on specific hyperparameters.

### Sim2Real
To evaluate CLIP/R3M policies, you will take the ckpt `.pt` file from the experiment output folder, and add the `--cnn-type clip` or `--cnn-type r3m` flag. During evaluation, we gave 10% extra timesteps for the policy to finish executing (so if during data collection we allocated 18 timesteps for a trajectory, during evaluation we allowed 20, etc.).
#### Pick-and-Place
```
python deoxys-lang4sim2real/deoxys/scripts/eval_collector.py --ckpt [ckpt] --obj-id [obj-id] --env frka_pp --state-mode 1 --task-embedding lang --lang-prefix Real: --max-path-len 20 --num-tasks 2 --eval-task-idxs 0-0 --num-rollouts-per-task 10 --gpu 0
```

#### 2-step Pick-and-Place
```
python deoxys-lang4sim2real/deoxys/scripts/eval_collector.py --ckpt [ckpt] --obj-id [obj-id] --env frka_obj_bowl_plate --state-mode 1 --task-embedding lang --lang-prefix Real: --max-path-len 50 --num-tasks 2 --eval-task-idxs 0-0 --num-rollouts-per-task 10 --gpu 0
```

#### Wire Wrap
```
python deoxys-lang4sim2real/deoxys/scripts/eval_collector.py --ckpt [ckpt] --obj-id [obj-id] --env frka_wirewrap --state-mode 1 --task-embedding lang --lang-prefix Real: --max-path-len 50 --num-tasks 2 --eval-task-idxs 0-0 --num-rollouts-per-task 10 --gpu 0
```