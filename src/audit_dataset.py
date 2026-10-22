#!/usr/bin/env python3
"""Read-only audit of original LIBERO HDF5 episode structure and provenance."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import h5py
import numpy as np


def scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return scalar(value.item())
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    return value


def shape_tree(group, depth=0):
    if depth > 3:
        return {}
    result = {}
    for name, item in group.items():
        result[name] = ({"shape": list(item.shape), "dtype": str(item.dtype)}
                        if isinstance(item, h5py.Dataset) else shape_tree(item, depth + 1))
    return result


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(path, limit):
    report = {"path": str(path.resolve()), "bytes": path.stat().st_size,
              "sha256": sha256(path), "environment": {}, "episodes": []}
    for package in ("libero", "robosuite", "mujoco", "h5py", "numpy"):
        try:
            report["environment"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["environment"][package] = None
    with h5py.File(path, "r") as handle:
        report["root_keys"] = list(handle.keys())
        if "data" not in handle:
            raise ValueError("Missing /data group")
        data = handle["data"]
        report["data_attrs"] = {key: scalar(value) for key, value in data.attrs.items()}
        names = sorted(data.keys(), key=lambda s: (not s.startswith("demo_"), int(s[5:]) if s.startswith("demo_") and s[5:].isdigit() else s))
        report["episode_count"] = len(names)
        for name in names[:limit]:
            episode = data[name]
            record = {"id": name, "attrs": {key: scalar(value) for key, value in episode.attrs.items() if key != "model_file"},
                      "fields": shape_tree(episode), "checks": {}}
            xml = episode.attrs.get("model_file")
            if xml is not None:
                if isinstance(xml, str): xml = xml.encode("utf-8")
                record["model_xml_sha256"] = hashlib.sha256(xml).hexdigest()
                record["model_xml_bytes"] = len(xml)
            states = episode.get("states")
            actions = episode.get("actions")
            record["checks"]["has_model_xml"] = xml is not None
            record["checks"]["has_states"] = states is not None and states.ndim == 2 and len(states) > 0
            record["checks"]["has_actions"] = actions is not None and actions.ndim == 2 and len(actions) > 0
            if states is not None and actions is not None:
                record["checks"]["state_action_length_match"] = abs(len(states) - len(actions)) <= 1
                for i in (0, len(states) // 2, len(states) - 1):
                    record["checks"][f"state_{i}_finite"] = bool(np.isfinite(states[i]).all())
            record["schema_pass"] = all(record["checks"].values())
            report["episodes"].append(record)
    report["schema_pass"] = len(report["episodes"]) == limit and all(e["schema_pass"] for e in report["episodes"])
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    if args.limit < 1: parser.error("--limit must be positive")
    report = audit(args.input, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n")
    print(json.dumps({"episode_count": report["episode_count"], "audited": len(report["episodes"]),
                      "schema_pass": report["schema_pass"], "sha256": report["sha256"],
                      "output": str(args.output)}, ensure_ascii=False))
    raise SystemExit(0 if report["schema_pass"] else 2)


if __name__ == "__main__":
    main()
