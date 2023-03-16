# RA-SLAM: Real-time Semantic Reconstruction

This project is built for the perception system of an autonomous disinfection robot. The core
components of this module is a real-time 3D semantic reconstruction algorithm, with an efficient GPU implementation of Voxel Hashing which supports semmantic integration.

Official implementation of the perception system described in our IROS2022 paper Real-time Semantic 3D Reconstruction for High-Touch Surface Recognition for Robotic Disinfection [pdf](http://motion.cs.illinois.edu/papers/IROS2022-Qiu-RealTimeSemanticReconstruction.pdf).

If you find this work helpful in your project, please kindly cite our work at

```
@inproceedings{qiu2022-real,
  title={Real-time Semantic 3D Reconstruction for High-Touch Surface Recognition for Robotic Disinfection},
  author={Qiu, Ri-Zhao and Sun, Yixiao and Marques, Joao Marcos Correia and Hauser, Kris},
  booktitle={IROS},
  year={2022},
  organization={IEEE}
}
```

## Instllation

For installling dependencies and running examples, please refer to [INSTALL.md](./docs/INSTALL.md).

For calibrating camera intrinsics for live application, please refer to [this instruction](./docs/calibrate_stereo_camera.md).

## Running Example ScanNet reconstruction

The last step before running example reconstruction is obtaining sample camera footage and trained network weights.

As the ScanNet imposes certain limitations of data usage, users should obtain ScanNet samples from the [official page](http://www.scan-net.org/). But [here](https://drive.google.com/file/d/1FtSU9z8hpwNy5x9BptIsy6Q85SjHu4WQ/view?usp=sharing) we provide a sample example for convenience of testing build (this will be removed in the future).

The binary high-touch segmentation model from our paper can be found [here](https://drive.google.com/file/d/19T1htg-KdhLOOagh_f0xlFJOi0nkVV-o/view?usp=sharing).

```bash
./main/offline_eval --model ~/disinfect-slam/segmentation/ht_lt.pt --sens "/media/roger/My Book/data/scannet_v2/scans/scene0249_00/scene0249_00.sens" --download --render --debug
```

On a machine with GUI, you can expect to see a OpenGL window pops up with ray-casting view of the reconstructed scene.
The behavior is yet to be tested if the program is run on a remote server with no GUI, but with the download flag, at the end of the example several binary files will be created at the current folder (e.g., mesh_vertices.bin).

and you should see argparser working and sending you a list of arguments to put it.

## TODO

- Add logic to not update TSDF when tracking is lost
- Update segmentation inference example to take in custom images
- Add rotation arthimetics (e.g. rotation matrices interpolation)
- TensorRT for saving GPU memory
- Measure optimal intervals to run semantic segmentation frames
