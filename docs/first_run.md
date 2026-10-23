# Remote experiment run

Project: `/mnt/nvme1/drh/D_VLA`. Conda environment: `D_VLA` (`/mnt/nvme1/drh/miniforge3/envs/D_VLA`). Simulation variables: `LIBERO_CONFIG_PATH=/mnt/nvme1/drh/D_VLA/configs/libero`, `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`.

The original task HDF5 is `data/raw/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5`. It has 50 demonstrations with 128×128 two-camera RGB and no depth. The 448×448 RGB-D re-renderings are in `data/rendered/alphabet_soup_demo_<n>_full_rgbd.h5`; they provide individual pre-contact frames for the active single-frame model. The entire video is not a model input.

Contact probes and 47 provisional grasp labels are in `data/events/`. Segmentation may be rendered for offline object assignment and label audit, but is not a model feature. No experiment data is copied to the local computer.
