"""Fetch only pinned source repositories with their small MD example, not the 133 GiB archive."""
import argparse
import subprocess
from pathlib import Path

SOURCES = [("MISATO", "https://github.com/t7morgen/misato-dataset.git", "7b06d532e2ed0719411fcc1b3ac39743db4ca10d"),
           ("misato-affinity", "https://github.com/kierandidi/misato-affinity.git", "d03e0e84ce0e73512dbc2f7741e7f36537874745")]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("data/raw"))
    args = p.parse_args()
    for name, url, commit in SOURCES:
        root = args.root / name
        if root.exists():
            existing = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
            if existing != commit: raise ValueError(f"{root}: different commit; preserve it and choose another --root")
        else:
            root.mkdir(parents=True)
            for command in [["init", str(root)], ["-C", str(root), "remote", "add", "origin", url],
                            ["-C", str(root), "fetch", "--depth", "1", "origin", commit],
                            ["-C", str(root), "checkout", "--detach", "FETCH_HEAD"]]:
                subprocess.run(["git", *command], check=True)
        print(name, root, commit)
    print("python prepare_md.py --md-h5 data/raw/MISATO/data/MD/h5_files/tiny_md.hdf5 "
          "--labels-h5 data/raw/misato-affinity/data/affinity_data.h5")


if __name__ == "__main__": main()
