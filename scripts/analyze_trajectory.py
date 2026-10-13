import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.dynamic.data import read_trajectory


def summarize(trajectory, cutoff=4.5):
    positions = trajectory["positions"].numpy()
    ligand = trajectory["ligand_mask"].numpy()
    reference = trajectory["reference_positions"].numpy()[~ligand]
    reference_centered = reference - reference.mean(0)
    rows = []
    for index, xyz in enumerate(positions):
        p, l = xyz[~ligand], xyz[ligand]
        distances = np.linalg.norm(p[:, None] - l[None], axis=-1)
        # Unweighted Kabsch fit of the selected heavy protein atoms only.
        centered = p - p.mean(0)
        u, _, vt = np.linalg.svd(centered.T @ reference_centered)
        sign = np.linalg.det(u @ vt)
        rotation = u @ np.diag([1., 1., sign]) @ vt
        aligned_p = centered @ rotation
        rmsd = np.sqrt(np.mean(np.sum((aligned_p - reference_centered) ** 2, axis=-1)))
        rows.append({"source_frame": int(trajectory["frame_indices"][index]),
                     "time": float(trajectory["time"][index]), "time_unit": trajectory["time_unit"],
                     "closest_heavy_distance_A": float(distances.min()),
                     "contact_pairs": int((distances <= cutoff).sum()),
                     "selected_protein_rmsd_A": float(rmsd)})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description="Inspect true MD movement/contact changes; RMSD refers to cropped protein atoms")
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("runs/trajectory_analysis"))
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()
    trajectory = read_trajectory(args.trajectory)
    frame = summarize(trajectory)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "dynamics.csv", index=False)
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7, 7))
        for ax, key, label in zip(axes, ["closest_heavy_distance_A", "contact_pairs", "selected_protein_rmsd_A"],
                                  ["Closest protein–ligand\ndistance (Å)", "Contact pairs ≤ 4.5 Å", "Selected protein RMSD (Å)"]):
            ax.plot(frame.time, frame[key], marker="o"); ax.set_ylabel(label)
        axes[-1].set_xlabel("Source frame index" if trajectory["time_unit"] == "frame_index" else "Time (" + trajectory["time_unit"] + ")")
        fig.tight_layout(); fig.savefig(args.output / "dynamics.png", dpi=180); plt.close(fig)
    print(frame.to_string(index=False))


if __name__ == "__main__": main()
