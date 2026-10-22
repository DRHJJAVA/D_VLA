# D_VLA — LIBERO RGB-D 3D grasp-region prediction

Experiment source: `LIBERO_RGBD三维抓取关键区域预测实验方案.md` (v1.0, 2026-10-23). This project is remote-only; raw demonstrations and all generated data stay on this server.

## Layout
- `src/`: versioned preprocessing, restoration, labeling, models, and evaluation code.
- `configs/`: fixed experiment settings and environment provenance.
- `data/raw/`: original HDF5 demonstrations; never edit in place.
- `data/audits/`: schema and state-restoration reports.
- `data/rendered/`: synchronized RGB/depth/segmentation and camera metadata.
- `data/events/`: expert grasp event tables and contact checks.
- `data/splits/`: episode/asset partitions and leakage checks.
- `data/samples/`: point-cloud and visual-feature caches.
- `outputs/`: checkpoints, metrics, and visualizations.
- `logs/`: downloads and run logs.
- `docs/`: protocol and manual review records.

## Current stage
Stage 1 complete for the LIBERO-Object pilot: all ten task HDF5 files obtained; ten alphabet-soup demonstrations passed state restoration and camera-field checks. Stage 2 has a synchronized RGB-D pilot. One grasp event is a numeric candidate pending manual surface/contact review. Training remains gated on visual alignment, 30-event review, safe splits, and a 20-sample fit check. See `docs/STATUS.md` and `docs/first_run.md`.

## External, read-only dependencies
LIBERO checkout: `/home/drh/project/ECV-VLA/vendor/LIBERO` at commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`. Simulation Python: `/mnt/nvme1/drh/ECV-VLA-data/envs/smolvla-hf-control/bin/python` (`robosuite 1.4.0`). These remain outside the new project and are not modified.

## First dataset
LIBERO-Object HDF5 from `yifengzhu-hf/LIBERO-datasets` through the Hugging Face mirror, written directly to `data/raw/libero_object/`. All ten task files have been size-checked and the `.part` suffix removed. Dataset metadata and SHA256 will be recorded in `data/audits/`.
