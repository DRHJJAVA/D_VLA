#!/usr/bin/env python3
"""Restore ten LIBERO expert states and audit synchronized two-camera observations."""
import argparse
import json
import os
from pathlib import Path
import traceback

import h5py
import numpy as np
from asset_paths import localize_assets
# robosuite 1.4.0 hard-codes /tmp/robosuite.log. On this shared server that file
# belongs to another user; redirect only that filename for this process at import.
import logging
_original_file_handler = logging.FileHandler
class _ProjectFileHandler(_original_file_handler):
    def __init__(self, filename, *args, **kwargs):
        if os.path.abspath(str(filename)) == "/tmp/robosuite.log":
            filename = "/mnt/nvme1/drh/D_VLA/logs/robosuite.log"
        super().__init__(filename, *args, **kwargs)
logging.FileHandler = _ProjectFileHandler
try:
    import libero
    _source_package = "/home/drh/project/ECV-VLA/vendor/LIBERO/libero"
    libero.__path__ = [_source_package] + [p for p in libero.__path__ if p != _source_package]
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.utils.utils as libero_utils
finally:
    logging.FileHandler = _original_file_handler


def save_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--libero-root", type=Path, required=True)
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--size", type=int, default=448)
    args = p.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    report = {"source": str(args.input), "render_size": args.size,
              "camera_names": ["agentview", "robot0_eye_in_hand"], "episodes": []}
    with h5py.File(args.input, "r") as f:
        data = f["data"]
        stored_bddl = data.attrs.get("bddl_file_name", "")
        if isinstance(stored_bddl, bytes): stored_bddl = stored_bddl.decode()
        bddl = args.libero_root / "libero/libero/bddl_files/libero_object" / Path(stored_bddl).name
        if not bddl.is_file(): raise FileNotFoundError(f"BDDL not found: {bddl}")
        report["bddl_file"] = str(bddl)
        names = sorted(data.keys(), key=lambda x: int(x[5:]) if x.startswith("demo_") and x[5:].isdigit() else 10**9)[:args.limit]
        print("creating environment", flush=True)
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_names=["agentview", "robot0_eye_in_hand"],
                                 camera_heights=args.size, camera_widths=args.size,
                                 camera_depths=True, camera_segmentations="instance")
        print("environment created", flush=True)
        try:
            for name in names:
                row = {"episode": name, "ok": False}
                try:
                    ep = data[name]
                    xml = ep.attrs["model_file"]
                    if isinstance(xml, bytes): xml = xml.decode("utf-8")
                    xml = libero_utils.postprocess_model_xml(xml, {})
                    xml, row["localized_libero_assets"] = localize_assets(xml, args.libero_root)
                    states = ep["states"]
                    print(name, "reset start", flush=True)
                    env.reset()
                    print(name, "reset complete", flush=True)
                    env.reset_from_xml_string(xml)
                    print(name, "xml loaded", flush=True)
                    env.sim.reset()
                    checks = []
                    for i in sorted({0, len(states) // 2}):
                        original = np.asarray(states[i], dtype=np.float64)
                        env.sim.set_state_from_flattened(original)
                        print(name, "state", i, "loaded", flush=True)
                        env.sim.forward()
                        restored = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
                        error = float(np.max(np.abs(restored - original))) if len(restored) == len(original) else None
                        env._update_observables(force=True)
                        obs = env.env._get_observations()
                        camera = {}
                        for key, value in obs.items():
                            if any(cam in key for cam in ("agentview", "robot0_eye_in_hand")):
                                arr = np.asarray(value)
                                camera[key] = {"shape": list(arr.shape), "dtype": str(arr.dtype),
                                               "finite": bool(np.isfinite(arr).all())}
                        checks.append({"frame": i, "state_length": len(original),
                                       "roundtrip_max_abs_error": error, "camera_fields": camera})
                    row["checks"] = checks
                    row["ok"] = all(x["roundtrip_max_abs_error"] is not None and x["roundtrip_max_abs_error"] < 1e-5
                                    and any(k.endswith("_image") for k in x["camera_fields"])
                                    and any(k.endswith("_depth") for k in x["camera_fields"])
                                    for x in checks)
                except Exception as exc:
                    row["error"] = repr(exc)
                    row["traceback"] = traceback.format_exc(limit=4)
                report["episodes"].append(row)
                save_report(args.output, report)
                print(name, "PASS" if row["ok"] else "FAIL", flush=True)
        finally:
            env.close()
    report["passed"] = sum(x["ok"] for x in report["episodes"])
    report["required"] = args.limit
    save_report(args.output, report)
    print(json.dumps({"passed": report["passed"], "required": args.limit, "output": str(args.output)}), flush=True)
    raise SystemExit(0 if report["passed"] == args.limit else 2)


if __name__ == "__main__":
    main()
