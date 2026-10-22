#!/usr/bin/env python3
"""Inventory LIBERO task files, episode counts, state fields, and object assets."""
import argparse
import csv
import json
from pathlib import Path
import re

import h5py

PATTERN = re.compile(r"/assets/((?:stable_hope_objects|stable_scanned_objects|turbosquid_objects|articulated_objects)/[^/]+)/")


def scalar(x):
    return x.decode("utf-8", "replace") if isinstance(x, bytes) else str(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-prefix", type=Path, required=True)
    p.add_argument("--expected-files", type=int, default=10)
    args = p.parse_args()
    files = sorted(args.input_dir.glob("*.hdf5"))
    if len(files) != args.expected_files:
        raise RuntimeError(f"Expected {args.expected_files} HDF5 files; found {len(files)}")
    rows = []
    for path in files:
        with h5py.File(path, "r") as f:
            group = f["data"]
            names = sorted(k for k in group if k.startswith("demo_"))
            if not names: raise RuntimeError(f"No demos in {path}")
            missing_states = sum("states" not in group[name] for name in names)
            if missing_states: raise RuntimeError(f"{missing_states} episodes lack states in {path}")
            frames = sum(len(group[name]["states"]) for name in names)
            missing_xml = sum("model_file" not in group[name].attrs for name in names)
            first = group[names[0]]
            xml = scalar(first.attrs["model_file"])
            assets = sorted(set(PATTERN.findall(xml)))
            attrs = group.attrs
            problem = json.loads(scalar(attrs.get("problem_info", "{}")))
            row = {"file": path.name, "bytes": path.stat().st_size,
                   "episodes": len(names), "frames": frames,
                   "missing_states": missing_states, "missing_model_xml": missing_xml,
                   "state_dimension": first["states"].shape[1],
                   "action_dimension": first["actions"].shape[1],
                   "rgb_observation_keys": sorted(k for k in first["obs"] if "rgb" in k),
                   "has_recorded_depth": any("depth" in k for k in first["obs"]),
                   "bddl_file": scalar(attrs.get("bddl_file_name", "")),
                   "language_instruction": problem.get("language_instruction"),
                   "object_asset_ids_in_first_model": assets}
            rows.append(row)
            print(f"{path.name}: {len(names)} episodes, {frames} frames", flush=True)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(json.dumps(rows,indent=2,ensure_ascii=False)+"\n")
    columns=["file","bytes","episodes","frames","missing_states","missing_model_xml",
             "state_dimension","action_dimension","has_recorded_depth","language_instruction"]
    with args.output_prefix.with_suffix(".csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for row in rows: writer.writerow({key:row.get(key) for key in columns})
    print(json.dumps({"tasks":len(rows),"episodes":sum(x["episodes"] for x in rows),
                      "frames":sum(x["frames"] for x in rows)},ensure_ascii=False))


if __name__ == "__main__":
    main()
