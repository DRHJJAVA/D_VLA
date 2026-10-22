#!/usr/bin/env python3
"""Render every saved state of one expert demonstration as synchronized RGB and depth video."""
import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
from asset_paths import localize_assets
from restore_audit import OffScreenRenderEnv, libero_utils
from robosuite.utils import camera_utils
import robosuite.macros as macros

CAMERAS = ("agentview", "robot0_eye_in_hand")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--episode", default="demo_0")
    parser.add_argument("--size", type=int, default=448)
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    if macros.IMAGE_CONVENTION not in ("opengl", "opencv"):
        raise ValueError(f"Unsupported image convention: {macros.IMAGE_CONVENTION}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".part")
    with h5py.File(args.input, "r") as source:
        data = source["data"]
        bddl_name = Path(data.attrs["bddl_file_name"]).name
        bddl = args.libero_root / "libero/libero/bddl_files/libero_object" / bddl_name
        episode = data[args.episode]
        states = episode["states"]
        nframes = len(states)
        xml = episode.attrs["model_file"]
        if isinstance(xml, bytes):
            xml = xml.decode()
        xml = libero_utils.postprocess_model_xml(xml, {})
        xml, mapped_assets = localize_assets(xml, args.libero_root)
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl), camera_names=list(CAMERAS),
            camera_heights=args.size, camera_widths=args.size,
            camera_depths=True, camera_segmentations=None,
        )
        try:
            env.reset()
            env.reset_from_xml_string(xml)
            env.sim.reset()
            with h5py.File(partial, "w") as output:
                output.attrs["episode"] = args.episode
                output.attrs["source_hdf5"] = str(args.input)
                output.attrs["num_frames"] = nframes
                output.attrs["image_convention"] = "opencv_top_left"
                output.attrs["depth_unit"] = "meters"
                output.attrs["input_type"] = "complete_demonstration_rgbd_video"
                output.attrs["has_segmentation"] = False
                output.create_dataset("frame_indices", data=np.arange(nframes, dtype=np.int32))
                groups = {}
                for camera in CAMERAS:
                    group = output.create_group(camera)
                    group.create_dataset("rgb", (nframes, args.size, args.size, 3), dtype="u1",
                                         chunks=(1, args.size, args.size, 3), compression="lzf")
                    group.create_dataset("depth_m", (nframes, args.size, args.size), dtype="f4",
                                         chunks=(1, args.size, args.size), compression="lzf")
                    group.create_dataset("K", (nframes, 3, 3), dtype="f4")
                    group.create_dataset("camera_to_world", (nframes, 4, 4), dtype="f4")
                    groups[camera] = group
                for index in range(nframes):
                    state = np.asarray(states[index], dtype=np.float64)
                    env.sim.set_state_from_flattened(state)
                    env.sim.forward()
                    env._update_observables(force=True)
                    obs = env.env._get_observations()
                    for camera in CAMERAS:
                        rgb = np.asarray(obs[camera + "_image"])
                        depth_buffer = np.asarray(obs[camera + "_depth"])[..., 0]
                        depth_m = camera_utils.get_real_depth_map(env.sim, depth_buffer)
                        if macros.IMAGE_CONVENTION == "opengl":
                            rgb = np.flipud(rgb)
                            depth_m = np.flipud(depth_m)
                        if rgb.shape != (args.size, args.size, 3) or depth_m.shape != (args.size, args.size):
                            raise ValueError(f"Unexpected RGB/depth shape for {camera} at frame {index}")
                        if not np.isfinite(depth_m).all() or np.any(depth_m <= 0):
                            raise ValueError(f"Invalid depth for {camera} at frame {index}")
                        groups[camera]["rgb"][index] = np.ascontiguousarray(rgb, dtype=np.uint8)
                        groups[camera]["depth_m"][index] = np.ascontiguousarray(depth_m, dtype=np.float32)
                        groups[camera]["K"][index] = camera_utils.get_camera_intrinsic_matrix(
                            env.sim, camera, args.size, args.size).astype(np.float32)
                        groups[camera]["camera_to_world"][index] = camera_utils.get_camera_extrinsic_matrix(
                            env.sim, camera).astype(np.float32)
                    if index % 10 == 0 or index == nframes - 1:
                        output.flush()
                        print(json.dumps({"frame": index + 1, "total": nframes}), flush=True)
        finally:
            env.close()
    partial.replace(args.output)
    metadata = {
        "episode": args.episode, "num_frames": nframes, "frame_indices": [0, nframes - 1],
        "cameras": list(CAMERAS), "resolution": [args.size, args.size],
        "input": "complete expert demonstration: every RGB frame and matching metric depth frame",
        "segmentation_used": False, "mapped_libero_assets": mapped_assets,
        "hdf5": str(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"completed": str(args.output), "frames": nframes}), flush=True)


if __name__ == "__main__":
    main()
