# D_VLA — LIBERO complete-video RGB-D grasp-region experiment

Project root: `/mnt/nvme1/drh/D_VLA`. Conda environment: `D_VLA` at `/mnt/nvme1/drh/miniforge3/envs/D_VLA`. All original and generated data remain on the remote data disk.

## Input contract
Each training example uses the **entire expert demonstration**, in temporal order, with RGB video and synchronized metric depth for `agentview` and `robot0_eye_in_hand`. Every saved state is rendered at 448×448. There is no temporal clipping or frame subsampling. No segmentation is produced by the current rendering pipeline or supplied as training input. MuJoCo contact and object pose information are used only to derive supervision and audit grasp events. See `docs/input_contract.md`.

## Layout
- `src/`: audit, rendering, event extraction, model and evaluation code.
- `configs/`: protocol settings and environment provenance.
- `data/raw/`: original LIBERO HDF5 demonstrations, untouched.
- `data/audits/`: dataset and state-restoration reports.
- `data/rendered/`: complete synchronized RGB-D demonstration videos.
- `data/events/`: contact-derived grasp event candidates.
- `data/splits/`: episode and asset partitions.
- `data/samples/`: training samples and caches.
- `outputs/`: checkpoints, metrics and visualizations.
- `logs/`: run logs.
- `docs/`: input contract, protocol and review records.

## Current stage
Ten LIBERO-Object tasks (500 demonstrations) are downloaded on the server. Ten alphabet-soup demonstrations passed state-restoration checks. A complete RGB-D video render for `demo_0` is in progress or complete; see `docs/STATUS.md`. One grasp event is a numeric candidate requiring review. No model training or performance result is claimed yet.

LIBERO source is `/home/drh/project/ECV-VLA/vendor/LIBERO` at commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`. The experiment runs in the dedicated `D_VLA` conda environment.
