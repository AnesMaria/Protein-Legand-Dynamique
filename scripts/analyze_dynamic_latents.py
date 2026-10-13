import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.utils import save_json


def main():
    p = argparse.ArgumentParser(description="Time-resolved SAE statistics and activation histograms")
    p.add_argument("--input", type=Path, required=True)
    args = p.parse_args()
    with np.load(args.input, allow_pickle=False) as archive:
        z, frames, replicas = archive["z"], archive["frame_index"], archive["trajectory_id"]
        threshold = float(archive["activity_threshold"])
        active = np.abs(z) > threshold
        counts, edges = np.histogram(z, bins=40)
        by_frame = []
        for replica in sorted(set(replicas)):
            for frame in sorted(set(frames[replicas == replica])):
                mask = (replicas == replica) & (frames == frame)
                by_frame.append({"trajectory_id": str(replica), "source_frame": int(frame),
                                 "mean_active_latents": float(active[mask].sum(1).mean()),
                                 "mean_l1_norm": float(np.abs(z[mask]).sum(1).mean())})
    report = {"patches": len(z), "latent_dim": z.shape[1], "fraction_zeros": float((z == 0).mean()),
              "active_ratio": float(active.mean()), "mean_active_latents": float(active.sum(1).mean()),
              "mean_latent_variance": float(z.var(0).mean()),
              "dead_latent_fraction": float((active.mean(0) == 0).mean()),
              "histogram": {"counts": counts.tolist(), "edges": edges.tolist()}, "by_frame": by_frame}
    path = args.input.with_suffix(".analysis.json")
    save_json(report, path)
    print(path)


if __name__ == "__main__": main()
