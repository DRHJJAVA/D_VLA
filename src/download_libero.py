#!/usr/bin/env python3
"""Download official LIBERO HDF5 files to this server, with size checks and resume."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess

REPO = "yifengzhu-hf/LIBERO-datasets"
MIRROR = "https://hf-mirror.com"


def listing(suite):
    url = f"{MIRROR}/api/datasets/{REPO}/tree/main/{suite}?recursive=true&expand=true"
    response = subprocess.run(["curl", "--fail", "--silent", "--show-error",
                               "--connect-timeout", "5", "--max-time", "30", url],
                              capture_output=True, text=True, check=True)
    entries = json.loads(response.stdout)
    return [(entry["path"], int(entry["size"])) for entry in entries
            if entry["path"].endswith(".hdf5")]


def download(item, root):
    path, size = item
    dest = root / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == size:
        return path, "already_verified", size
    part = dest.with_suffix(dest.suffix + ".part")
    url = f"{MIRROR}/datasets/{REPO}/resolve/main/{path}"
    for attempt in range(3):
        result = subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                                 "--retry", "5", "--retry-delay", "5", "--continue-at", "-",
                                 "--output", str(part), url], capture_output=True, text=True)
        if result.returncode == 0 and part.exists() and part.stat().st_size == size:
            part.replace(dest)
            return path, "downloaded", size
        print(f"retry {attempt+1}: {path}: curl={result.returncode}, bytes={part.stat().st_size if part.exists() else 0}, expected={size}, error={result.stderr[-250:]}", flush=True)
    raise RuntimeError(f"Failed size check after retries: {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="libero_object")
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    items = listing(args.suite)
    if not items: raise RuntimeError("No HDF5 in repository listing")
    print(f"{len(items)} HDF5 files; {sum(size for _, size in items)} bytes", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download, item, args.dest) for item in items]
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
