# robomimic-lang4sim2real

This is our version of the [original robomimic repo](https://github.com/ARISE-Initiative/robomimic) to support collecting multitask buffers and other environment changes made in our version of robosuite.

Most of our changes are in the following files:
- [dataset_states_to_obs.py](./robomimic/scripts/dataset_states_to_obs.py): Converting dataset states into observations when saving a robosuite dataset.
- [convert_robosuite.py](./robomimic/scripts/conversion/convert_robosuite.py): Adding extra attributes when saving a robosuite datsaet.

We also removed the `docs/` folder to streamline this folder.

[**[Homepage]**](https://robomimic.github.io/) &ensp; [**[Documentation]**](https://robomimic.github.io/docs/introduction/overview.html) &ensp; [**[Study Paper]**](https://arxiv.org/abs/2108.03298) &ensp; [**[Study Website]**](https://robomimic.github.io/study/) &ensp; [**[ARISE Initiative]**](https://github.com/ARISE-Initiative)

-------
## Latest Updates
- [05/23/2022] **v0.2.1**: Updated website and documentation to feature more tutorials :notebook_with_decorative_cover:
- [12/16/2021] **v0.2.0**: Modular observation modalities and encoders :wrench:, support for [MOMART](https://sites.google.com/view/il-for-mm/home) datasets :open_file_folder:
- [08/09/2021] **v0.1.0**: Initial code and paper release

-------

**robomimic** is a framework for robot learning from demonstration.
It offers a broad set of demonstration datasets collected on robot manipulation domains and offline learning algorithms to learn from these datasets.
**robomimic** aims to make robot learning broadly *accessible* and *reproducible*, allowing researchers and practitioners to benchmark tasks and algorithms fairly and to develop the next generation of robot learning algorithms.

## Reproducing benchmarks

The robomimic framework also makes reproducing the results from different benchmarks and datasets easy. See the [datasets page](https://robomimic.github.io/docs/datasets/overview.html) for more information on downloading datasets and reproducing experiments.

## Troubleshooting

Please see the [troubleshooting](https://robomimic.github.io/docs/miscellaneous/troubleshooting.html) section for common fixes, or [submit an issue](https://github.com/ARISE-Initiative/robomimic/issues) on our github page.

## Contributing to robomimic
This project is part of the broader [Advancing Robot Intelligence through Simulated Environments (ARISE) Initiative](https://github.com/ARISE-Initiative), with the aim of lowering the barriers of entry for cutting-edge research at the intersection of AI and Robotics.
The project originally began development in late 2018 by researchers in the [Stanford Vision and Learning Lab](http://svl.stanford.edu/) (SVL).
Now it is actively maintained and used for robotics research projects across multiple labs.
We welcome community contributions to this project.
For details please check our [contributing guidelines](https://robomimic.github.io/docs/miscellaneous/contributing.html).

## Citation

Please cite [this paper](https://arxiv.org/abs/2108.03298) if you use this framework in your work:

```bibtex
@inproceedings{robomimic2021,
  title={What Matters in Learning from Offline Human Demonstrations for Robot Manipulation},
  author={Ajay Mandlekar and Danfei Xu and Josiah Wong and Soroush Nasiriany and Chen Wang and Rohun Kulkarni and Li Fei-Fei and Silvio Savarese and Yuke Zhu and Roberto Mart\'{i}n-Mart\'{i}n},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2021}
}
```
