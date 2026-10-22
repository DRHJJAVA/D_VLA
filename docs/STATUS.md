# Experiment status (2026-10-23, Asia/Shanghai)

Completed:
- Created independent conda environment `D_VLA` on the data disk; pinned robosuite 1.4.0 and saved package snapshots.
- Downloaded all ten official LIBERO-Object HDF5 task files directly to the server (500 demonstrations, 74,507 state frames).
- Audited ten episodes in the alphabet-soup task: 10/10 state roundtrips passed, with two 448×448 RGB/depth/instance streams.
- Rendered `demo_0/frame_0` RGB, metric depth, instance mask, camera matrices, and world XYZ for both cameras.
- Probed all 148 states in `demo_0`; found one numerically qualified, 88-frame two-finger grasp candidate.

Quality gates still open:
- Visually check RGB/depth/segmentation/point-cloud alignment and collision contacts against the object surface.
- Randomly review 30 grasp events across tasks; initial 0.05 m lift and 0.03 m relative-drift thresholds are pilot values.
- Create leakage-safe episode and asset splits only after event/asset review.
- Then build samples, verify 20-sample overfit, and train baselines and full model.

No trained network, validation coverage, or generalization result is claimed. All data, logs, and plots stay on the remote server.
