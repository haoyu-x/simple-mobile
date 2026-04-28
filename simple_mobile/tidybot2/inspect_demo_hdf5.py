#!/usr/bin/env python3
"""Open and summarize a robomimic-style HDF5 file (e.g. demo.hdf5 from convert_to_robomimic_hdf5.py).

Requires: pip install h5py

Usage:
  python inspect_demo_hdf5.py
  python inspect_demo_hdf5.py /path/to/demo.hdf5
  python inspect_demo_hdf5.py --tree demo.hdf5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py


def _dump_group(g: h5py.Group, prefix: str = "") -> None:
    for key in sorted(g.keys(), key=str):
        item = g[key]
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(item, h5py.Dataset):
            print(f"  {path}: shape={tuple(item.shape)}, dtype={item.dtype}")
        else:
            print(f"  {path}/")
            _dump_group(item, path)


def _summarize_data_group(data: h5py.Group) -> None:
    demos = sorted(
        (k for k in data.keys() if k.startswith("demo_")),
        key=lambda x: int(x.split("_", 1)[1]),
    )
    print(f"Episodes: {len(demos)}")
    if not demos:
        return
    for d in demos[: min(5, len(demos))]:
        g = data[d]
        if "actions" in g:
            a = g["actions"]
            print(f"  {d}/actions: shape={tuple(a.shape)}, dtype={a.dtype}")
        obs = g.get("obs")
        if obs is not None:
            for ok in sorted(obs.keys()):
                ds = obs[ok]
                print(f"  {d}/obs/{ok}: shape={tuple(ds.shape)}, dtype={ds.dtype}")
    if len(demos) > 5:
        print(f"  … and {len(demos) - 5} more episode(s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect robomimic / tidybot2 HDF5 demos.")
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("demo.hdf5"),
        help="Path to .hdf5 (default: ./demo.hdf5)",
    )
    parser.add_argument(
        "--tree",
        action="store_true",
        help="Print full group/dataset tree",
    )
    args = parser.parse_args()
    path = args.path.expanduser().resolve()
    if not path.is_file():
        print(f"Error: not a file: {path}", file=sys.stderr)
        sys.exit(1)

    with h5py.File(path, "r") as f:
        print(f"File: {path}")
        if args.tree:
            _dump_group(f)
        elif "data" in f and isinstance(f["data"], h5py.Group):
            _summarize_data_group(f["data"])
        else:
            print("(no top-level 'data' group; printing full tree)")
            _dump_group(f)


if __name__ == "__main__":
    main()
