# GitHub repository contents

This repository contains the D_VLA / LCJP003 research source, configuration, and design documentation. LCJP003 R5 is the active experiment; earlier versions are retained as implementation history and dependencies.

## Excluded server artifacts

Raw LIBERO demonstrations, RGB-D observations, generated input/teacher caches, pretrained weights, checkpoints, runtime environments, rollout recordings and large per-frame reports remain on the original server. The Git repository is not a complete training-data backup.

## Current execution

R5 source: experiments/lcjp003/r5/joint/v001/src/. The engineering continuation is under engineering/history_throughput/; its original handoff is bound to update 637 and must not be used to roll back or restart a later checkpoint. Closed-loop evaluation is under closed_loop_object50/. Reproduction requires recreating the environment, obtaining the original dependencies and datasets, and preparing or restoring the matching caches.

The main design and experiment log is LCJP003_状态条件三维部位评分与DriftingVLA设计方案.md in this directory. Historical result pages may require assets retained only on the server. Training and evaluation are not launched by uploading this repository.
