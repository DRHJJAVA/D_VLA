# First remote run

Project: `/mnt/nvme1/drh/D_VLA`
Conda environment: `D_VLA` (`/mnt/nvme1/drh/miniforge3/envs/D_VLA`)

Activate with `source /mnt/nvme1/drh/miniforge3/etc/profile.d/conda.sh && conda activate D_VLA`. Set `LIBERO_CONFIG_PATH=/mnt/nvme1/drh/D_VLA/configs/libero`, `MUJOCO_GL=egl`, and `PYOPENGL_PLATFORM=egl` for simulation.

Key outputs:
- `data/audits/libero_object_inventory.csv` and `.json`: ten task files, 500 demonstrations, 74,507 frames.
- `data/audits/alphabet_soup_schema_10.json`: first HDF5 integrity/field audit and SHA256.
- `data/audits/alphabet_soup_restore_10.json`: all ten restored; camera fields and state roundtrip error below 1e-12 (maximum observed 2.22e-16).
- `data/rendered/alphabet_soup_demo0_frame0/`: RGB PNG, per-camera NPZ containing depth in meters, instance IDs, XYZ world points and matrices.
- `data/events/alphabet_soup_demo0_contact_probe.json`: MuJoCo contact points and object/gripper poses for 148 frames.
- `data/events/alphabet_soup_demo0_grasp_candidates.json`: one numeric grasp candidate, pending manual surface review.

The original HDF5 files are in `data/raw/libero_object/`. Their stored RGB is 128×128 and contains no depth; the 448×448 RGB and depth in `data/rendered/` are re-rendered from the same saved state. The current point cloud is a full observation pilot, before workspace calibration and 4,096-point sampling.
