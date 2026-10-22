# First remote run

Project: `/mnt/nvme1/drh/D_VLA`. Conda environment: `D_VLA` (`/mnt/nvme1/drh/miniforge3/envs/D_VLA`).

For simulation, set `LIBERO_CONFIG_PATH=/mnt/nvme1/drh/D_VLA/configs/libero`, `MUJOCO_GL=egl`, and `PYOPENGL_PLATFORM=egl`. Run `/mnt/nvme1/drh/miniforge3/envs/D_VLA/bin/python src/render_rgbd_episode.py --input data/raw/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5 --output data/rendered/alphabet_soup_demo_0_full_rgbd.h5 --libero-root /home/drh/project/ECV-VLA/vendor/LIBERO --episode demo_0` from the project root.

The resulting HDF5 stores the **entire 148-frame demonstration**, with synchronized RGB and metric depth for both cameras, plus camera matrices. It has no segmentation. Original HDF5 demonstrations are in `data/raw/libero_object/`; their stored RGB is 128×128 and they have no depth, so the saved simulator states are re-rendered at 448×448.

Audit reports are in `data/audits/`; contact-derived candidate labels are in `data/events/`. The original single-frame segmented render is obsolete and is excluded from the input pipeline. No data is copied to the local computer.
