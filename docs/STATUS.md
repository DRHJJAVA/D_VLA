# Experiment status (2026-10-23, Asia/Shanghai)

Completed:
- Created the `D_VLA` conda environment on the data disk.
- Downloaded all ten LIBERO-Object task HDF5 files directly on the server (500 demonstrations, 74,507 saved states).
- Verified 10/10 state restorations for the alphabet-soup task; two RGB and metric-depth cameras at 448×448 are available.
- Probed all 148 states of `demo_0` and identified one numeric two-finger grasp candidate, pending review.
- Corrected the input protocol to **the full demonstration RGB video plus matching depth at every frame**. The new renderer does not request or save segmentation.

Completed and validated: the complete 148-frame `demo_0` RGB-D demonstration is in `data/rendered/alphabet_soup_demo_0_full_rgbd.h5` (230 MB). Both cameras have all 148 RGB/depth frames and camera matrices; every depth pixel is finite and positive, frame indices are 0–147, and no segmentation fields are present.

In progress: batch rendering all 50 complete demonstrations for the first LIBERO-Object task; progress is logged in `logs/render_first_task.log` and `data/rendered/alphabet_soup_render_manifest.json`.

Remaining before training: validate whole-video RGB/depth alignment, review contact-derived grasp labels, define demonstration/asset-safe splits, design the full-sequence model, and run a small-sample fit check. No trained model or performance result is claimed. All data stays on the remote server.
