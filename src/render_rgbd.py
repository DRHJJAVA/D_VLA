#!/usr/bin/env python3
"""Render a restored expert state to aligned RGB/depth/instance and world points."""
import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from restore_audit import OffScreenRenderEnv, libero_utils
from asset_paths import localize_assets
from robosuite.utils import camera_utils
import robosuite.macros as macros

CAMERAS = ("agentview", "robot0_eye_in_hand")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--libero-root", type=Path, required=True)
    p.add_argument("--episode", default="demo_0")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--size", type=int, default=448)
    args = p.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.input, "r") as f:
        data = f["data"]
        bddl_name = Path(data.attrs["bddl_file_name"]).name
        bddl = args.libero_root / "libero/libero/bddl_files/libero_object" / bddl_name
        ep = data[args.episode]
        xml = ep.attrs["model_file"]
        if isinstance(xml, bytes): xml = xml.decode()
        xml = libero_utils.postprocess_model_xml(xml, {})
        xml, mapped = localize_assets(xml, args.libero_root)
        state = np.asarray(ep["states"][args.frame], dtype=np.float64)
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_names=list(CAMERAS),
                                 camera_heights=args.size, camera_widths=args.size,
                                 camera_depths=True, camera_segmentations="instance")
        try:
            env.reset()
            env.reset_from_xml_string(xml)
            env.sim.reset()
            env.sim.set_state_from_flattened(state)
            env.sim.forward()
            env._update_observables(force=True)
            obs = env.env._get_observations()
            report = {"episode": args.episode, "frame": args.frame,
                      "image_convention_original": macros.IMAGE_CONVENTION,
                      "saved_convention": "opencv_top_left",
                      "depth_unit": "meters", "xyz_frame": "world",
                      "mapped_libero_assets": mapped, "cameras": {}}
            if macros.IMAGE_CONVENTION not in ("opengl", "opencv"):
                raise ValueError(f"Unexpected image convention: {macros.IMAGE_CONVENTION}")
            for camera in CAMERAS:
                rgb = np.asarray(obs[camera + "_image"])
                depth_buffer = np.asarray(obs[camera + "_depth"])[..., 0]
                segmentation = np.asarray(obs[camera + "_segmentation_instance"])[..., 0]
                depth_m = camera_utils.get_real_depth_map(env.sim, depth_buffer)
                if macros.IMAGE_CONVENTION == "opengl":
                    rgb, depth_m, segmentation = (np.flipud(x) for x in (rgb, depth_m, segmentation))
                rgb = np.ascontiguousarray(rgb)
                depth_m = np.ascontiguousarray(depth_m, dtype=np.float32)
                segmentation = np.ascontiguousarray(segmentation)
                K = camera_utils.get_camera_intrinsic_matrix(env.sim, camera, args.size, args.size)
                T = camera_utils.get_camera_extrinsic_matrix(env.sim, camera)
                v, u = np.indices(depth_m.shape)
                x = (u - K[0, 2]) * depth_m / K[0, 0]
                y = (v - K[1, 2]) * depth_m / K[1, 1]
                xyz_camera = np.stack([x, y, depth_m], axis=-1)
                xyz_world = xyz_camera @ T[:3, :3].T + T[:3, 3]
                valid = np.isfinite(depth_m) & (depth_m > 0) & (depth_m < 5)
                Image.fromarray(rgb).save(args.output_dir / f"{camera}_rgb.png")
                np.savez_compressed(args.output_dir / f"{camera}.npz", rgb=rgb,
                                    depth_m=depth_m, segmentation=segmentation,
                                    xyz_world=xyz_world.astype(np.float32), valid=valid,
                                    K=K, camera_to_world=T)
                report["cameras"][camera] = {"rgb_shape": list(rgb.shape),
                                             "depth_range_m": [float(np.min(depth_m[valid])), float(np.max(depth_m[valid]))] if valid.any() else None,
                                             "valid_depth_fraction": float(valid.mean()),
                                             "segmentation_ids": np.unique(segmentation).astype(int).tolist(),
                                             "intrinsic": K.tolist(), "camera_to_world": T.tolist()}
            (args.output_dir / "metadata.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"output_dir": str(args.output_dir), "cameras": list(report["cameras"]),
                              "valid_depth_fraction": {k:v["valid_depth_fraction"] for k,v in report["cameras"].items()}}))
        finally:
            env.close()


if __name__ == "__main__":
    main()
