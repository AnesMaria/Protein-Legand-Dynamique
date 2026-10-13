"""Run inside the original trusted PocketDTA/TorchDrug environment.

Explicitly opt in to loading a trusted original pickle. The baseline itself only
loads safe tensor dictionaries and does not require TorchDrug or PyG.
"""
import argparse
import json
import pickle
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pickle", type=Path, required=True)
    p.add_argument("--index-map", type=Path, required=True, help="JSON: protein_id -> index in original pickle")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--trust-source", action="store_true", help="Allow pickle execution for a trusted source")
    args = p.parse_args()
    if not args.trust_source:
        p.error("Original pickle is executable: provide --trust-source only for a trusted file")
    with args.pickle.open("rb") as f:
        objects = pickle.load(f)
    indices = json.loads(args.index_map.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    mapping = {}
    for protein_id, index in indices.items():
        obj = objects[index]
        graph = {}
        for key in ("node_s", "node_v", "edge_s", "edge_v", "edge_index", "residue_index"):
            value = getattr(obj, key, None)
            if value is not None:
                graph[key] = torch.as_tensor(value).detach().cpu()
        positions = getattr(obj, "node_position", None)
        if positions is not None:
            graph["coords"] = torch.as_tensor(positions).detach().cpu()
        # TorchDrug edge_list has [source, destination, relation] columns.
        if "edge_index" not in graph and getattr(obj, "edge_list", None) is not None:
            graph["edge_index"] = obj.edge_list[:, :2].T.long().cpu()
        if "node_s" not in graph and getattr(obj, "node_feature", None) is not None:
            value = obj.node_feature
            graph["node_s"] = (value.to_dense() if value.is_sparse else value).detach().cpu()
        if not graph:
            raise ValueError(f"No supported tensor fields for {protein_id}; inspect the upstream object")
        # Atom-level graphs are allowed, but do NOT fabricate residue_index.
        path = args.output / f"graph_{len(mapping):05d}.pt"
        torch.save(graph, path)
        mapping[protein_id] = str(path.resolve())
    (args.output / "graph_map.json").write_text(json.dumps(mapping, indent=2))
    print(f"Exported {len(mapping)} graphs. No sequence alignment inferred.")


if __name__ == "__main__":
    main()
