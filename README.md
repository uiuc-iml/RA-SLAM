# Real-time Semantic Reconstruction

This project is built for the perception system of an autonomous disinfection robot. The core
components of this module is a real-time 3D semantic reconstruction algorithm, with an efficient GPU implementation of Voxel Hashing which supports semmantic integration.

## Instllation

For installing externel dependancies, to avoid collision with existing system packages, it is
recommended to install all packages (including the ones that OpenVSLAM depends on) into
a local directory such as `~/.local` instead of the system path `/usr/local`. This can
be easily achieved by adding `-DCMAKE_INSTALL_PREFIX=$HOME/.local` option to a CMake command.
Another benifit of this is that `sudo` privilege is not needed for the installation.

## Running Reconstruction

This is still WIP, but try running

```bash
./main/offline_eval
```

and you should see argparser working and sending you a list of arguments to put it.

## TODO

- Add logic to not update TSDF when tracking is lost
- Update segmentation inference example to take in custom images
- Add rotation arthimetics (e.g. rotation matrices interpolation)
- TensorRT for saving GPU memory
- Measure optimal intervals to run semantic segmentation frames
