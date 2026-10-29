# D_VLA — LIBERO RGB-D 3D grasp-region prediction

Project root: `/mnt/nvme1/drh/D_VLA`. Conda environment: `D_VLA` at `/mnt/nvme1/drh/miniforge3/envs/D_VLA`. All raw and derived data stay on the remote data disk.

## Active input and output
Each training sample is a **single pre-contact RGB-D frame** from the two synchronized cameras. Frozen DINOv2 features and a 4,096-point 3D scene cloud feed a pointwise model, which outputs a grasp-region score for each point. Complete demonstrations are used to derive expert finger/object contact labels. Segmentation is offline supervision metadata only and never model input. See `docs/experiment_plan_v1.md` and `docs/input_contract.md`.

## Layout
- `src/`: active audits, rendering, labeling, sample preparation, models, training and evaluation.
- `configs/`: fixed experiment settings and environment provenance.
- `data/raw/`: original LIBERO HDF5 demonstrations, untouched.
- `data/audits/`: dataset and state-restoration reports.
- `data/rendered/`: aligned RGB-D renderings; complete sequences are a source for selecting individual frames.
- `data/events/`: contact-derived grasp-event candidates and labels.
- `data/splits/`: episode/asset splits.
- `data/samples/`: point-cloud and DINO feature caches for single-frame training.
- `outputs/`: active checkpoints and metrics; previous full-video trial is archived separately.
- `logs/`: run logs.
- `docs/`: protocol, status, and review records.

## Current stage
All 50 demonstrations of the first LIBERO-Object task have been rendered as complete RGB-D sequences to support frame selection. Contact candidates have been extracted for 47 of 50 demonstrations, pending manual review. The active pointwise model is being prepared and trained. See `docs/STATUS.md`.

LIBERO source is `/home/drh/project/ECV-VLA/vendor/LIBERO` at commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`.


## LCJP003 current project

See [repository contents and server artifact requirements](docs/GITHUB_CONTENTS.md) and [the LCJP003 design and experiment log](docs/LCJP003_状态条件三维部位评分与DriftingVLA设计方案.md).
