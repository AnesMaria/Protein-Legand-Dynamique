"""Build explicit C-alpha tensor graphs from PocketDTA PDB/P2Rank archives.

This is a baseline adapter, not the original TorchDrug/GearNet featurizer.
"""
import io
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .utils import atomic_save, digest, save_json

AMINO = dict(zip(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
    "ARNDCQEGHILKMFPSTWYV"))


def alpha_carbons(text):
    nodes = {}
    for line in text.splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM  ") or line[12:16].strip() != "CA" or line[16] not in (" ", "A"):
            continue
        key = (line[21], line[22:27].strip())
        nodes.setdefault(key, {"coords": [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                               "aa": AMINO.get(line[17:20], "X"), "bfactor": float(line[60:66])})
    return nodes


def graph_from_pocket(pdb_text, residue_table, sequence, pocket_rank=1, edge_radius=10.):
    nodes = alpha_carbons(pdb_text)
    if not nodes:
        raise ValueError("PDB has no alpha carbons")
    residue_table = residue_table.copy()
    residue_table.columns = residue_table.columns.str.strip()
    for column in ("chain", "residue_label"):
        residue_table[column] = residue_table[column].astype(str).str.strip()
    # Exact chain sequence matching only. No positional mapping across isoforms/mutations.
    chains = {}
    for key in nodes:
        chains.setdefault(key[0], []).append(key)
    matches = [keys for keys in chains.values() if "".join(nodes[k]["aa"] for k in keys) == sequence]
    alignment = {key: i for i, key in enumerate(matches[0])} if len(matches) == 1 else None
    selected = residue_table.loc[residue_table.pocket.eq(pocket_rank)] if pocket_rank > 0 else residue_table.loc[residue_table.pocket.gt(0)]
    keys, features = [], []
    for row in selected.itertuples(index=False):
        key = (str(row.chain), str(row.residue_label))
        if key in nodes:
            keys.append(key)
            features.append([float(row.score), float(row.probability), nodes[key]["bfactor"] / 100.])
    if not keys:
        raise ValueError(f"No matched residues for pocket rank {pocket_rank}")
    coords = torch.tensor([nodes[key]["coords"] for key in keys], dtype=torch.float32)
    distances = torch.cdist(coords, coords)
    edge_index = ((distances < edge_radius) & (distances > 0)).nonzero().T
    displacement = coords[edge_index[1]] - coords[edge_index[0]]
    lengths = displacement.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    graph = {"coords": coords, "node_s": torch.tensor(features, dtype=torch.float32),
             "edge_index": edge_index, "edge_s": lengths / edge_radius,
             "edge_v": (displacement / lengths).unsqueeze(1)}
    if alignment is not None and all(key in alignment for key in keys):
        graph["residue_index"] = torch.tensor([alignment[key] for key in keys], dtype=torch.long)
    return graph


def build_pocket_graphs(frame, source, output, dataset, pocket_rank=1):
    folder = "Davis" if dataset == "davis" else "KIBA"
    source, output = Path(source), Path(output).resolve()
    pdb_path, rank_path = [source / "data" / folder / (name + ".zip") for name in ("pdb", "p2rank")]
    if not pdb_path.exists() or not rank_path.exists():
        warnings.warn("PocketDTA PDB/P2Rank archive missing; using sequence fallback", stacklevel=2)
        return {}
    mapping, failures, aligned = {}, {}, 0
    with zipfile.ZipFile(pdb_path) as pdbs, zipfile.ZipFile(rank_path) as ranks:
        pdb_members = {Path(n).name: n for n in pdbs.namelist() if n.endswith(".pdb")}
        rank_members = {Path(n).name: n for n in ranks.namelist() if n.endswith("_residues.csv")}
        for row in frame[["Target_ID", "Target", "PDB_File"]].drop_duplicates().itertuples(index=False):
            protein, sequence, filename = str(row.Target_ID), row.Target, row.PDB_File
            if protein in mapping:
                continue
            if filename not in pdb_members or filename + "_residues.csv" not in rank_members:
                failures[protein] = "PDB or P2Rank table missing"
                continue
            try:
                pdb_text = pdbs.read(pdb_members[filename]).decode()
                table = pd.read_csv(io.BytesIO(ranks.read(rank_members[filename + "_residues.csv"])))
                graph = graph_from_pocket(pdb_text, table, sequence, pocket_rank)
            except (ValueError, KeyError) as exc:
                failures[protein] = str(exc)
                continue
            path = output / (digest([protein, sequence, filename, pocket_rank]) + ".pt")
            atomic_save(graph, path)
            mapping[protein] = str(path)
            aligned += int("residue_index" in graph)
    save_json({"adapter": "C-alpha pocket, P2Rank rank " + str(pocket_rank),
               "graphs": len(mapping), "exactly_aligned": aligned, "fallbacks": failures},
              output / "graph_provenance.json")
    print(f"Pocket graphs={len(mapping)}, exact ESM alignment={aligned}, fallback proteins={len(failures)}")
    return mapping
