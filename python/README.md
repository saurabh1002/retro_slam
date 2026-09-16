<div align="center">
    <h1>Retro-SLAM</h1>
    <a href="https://github.com/PRBonn/retro-slam/releases"><img src="https://img.shields.io/github/v/release/PRBonn/retro-slam?label=version" /></a>
    <a href="https://github.com/PRBonn/retro-slam/blob/main/LICENSE"><img src="https://img.shields.io/github/license/PRBonn/retro-slam" /></a>
    <a href="https://github.com/PRBonn/retro-slam/blob/main/"><img src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black" /></a>
    <a href="https://github.com/PRBonn/retro-slam/blob/main/"><img src="https://img.shields.io/badge/mac%20os-000000?&logo=apple&logoColor=white" /></a>
    <br />
    <br />
    <a href="https://github.com/PRBonn/retro-slam/blob/main/README.md#Install">Install</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/gupta2026arxiv.pdf">Paper</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href=https://github.com/PRBonn/retro-slam/issues>Contact Us</a>
  <br />
  <br />

[Retro-SLAM](https://arxiv.org/pdf/2607.13516): Improving Map Consistency in Graph-Based LiDAR SLAM Through Information-Aware Odometry and Retroactive Loop Closure

</div>

<hr />

## Install

```
pip install retro-slam
```

## Running the system
Next, follow the instructions on how to run the system by typing:
```
retro_slam_pipeline --help
```

This should print the following help message:

![help](https://github.com/user-attachments/assets/5a6fe624-2aaf-466f-8a18-51039b794000)

### Config
You can generate a default `config.yaml` by typing:

```
retro_slam_dump_config
```

which will generate a `retro_slam.yaml` file. Now, you can modify the parameters and pass the file to the `--config` option when running the `retro_slam_pipeline`.

Suggestion for indoor applications:
1. Reduce the `odometry.preprocessing.max_range` to 50.0, this will automatically reduce the `voxel_size` to 0.5.
2. Reduce the `local_mapper.splitting_distance` to a suitable distance based on the scale of the indoor environment.

### Install Python API (developer mode)
For development purposes:

```
sudo apt install git python3-pip libeigen3-dev libsuitesparse-dev
python3 -m pip install --upgrade pip
git clone https://github.com/PRBonn/retro-slam.git
cd retro-slam
make editable
```

### Paper Results
As we decided to continue developing **Retro-SLAM** beyond the scope of the IROS paper, we created a ``git tag`` so that researchers can consistently reproduce the results presented in the publication. To checkout at this tag, you can run the following:
```sh
git checkout RAL
```
Our development aims to surpass the performance of **Retro-SLAM** beyond the original results presented in the paper.

## Citation
If you use this library for any academic work, please cite our original paper:
```bib
@article{gupta2026arxiv,
  author   = {S. Gupta and N. Trekel and L.Wiesmann and C. Stachniss},
  title    = {{Improving Map Consistency in Graph-Based LiDAR SLAM Through Information-Aware Odometry and Retroactive Loop Closure}},
  journal  = {arXiv preprint},
  year     = 2026,
  volume   = {arXiv:2607.13516},
  url      = {https://arxiv.org/pdf/2607.13516},
}
```

## Acknowledgements
This project builds on top of [KISS-SLAM](https://github.com/PRBonn/kiss-slam).

## Contributing

We envision this as a community-driven project. We love to see how the project is growing, thanks to the contributions from the community. We would love to see your face in the list below; open a Pull Request!

<a href="https://github.com/PRBonn/retro-slam/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PRBonn/retro-slam" />
</a>

## Contact Us
For questions or feedback:
- GitHub Issues: https://github.com/PRBonn/retro-slam/issues
