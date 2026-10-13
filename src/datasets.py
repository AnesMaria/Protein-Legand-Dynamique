from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .utils import digest

SPLITS = ("random", "cold_protein", "cold_ligand", "cold_both")


def read_table(path):
    path = Path(path)
    df = pd.read_csv(path, keep_default_na=False)
    required = {"protein_id", "ligand_id", "sequence", "smiles", "affinity"}
    if not required.issubset(df):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")
    if df.empty:
        raise ValueError("Empty dataset")
    for column in ("sequence", "smiles", "protein_id", "ligand_id"):
        if df[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Empty {column}")
    df["affinity"] = pd.to_numeric(df.affinity, errors="raise")
    if not np.isfinite(df.affinity).all():
        raise ValueError("Non-finite affinities")
    # Identity by content prevents aliases with the same sequence/SMILES leaking.
    df["protein_key"] = df.sequence.map(lambda s: digest(str(s)))
    df["ligand_key"] = df.smiles.map(lambda s: digest(str(s)))
    if df.duplicated(["protein_key", "ligand_key"]).any():
        raise ValueError("Duplicate sequence/SMILES pairs; aggregate in prepare_data.py")
    if "graph_path" not in df:
        df["graph_path"] = ""
    df["graph_path"] = df.graph_path.map(
        lambda p: str((path.parent / str(p)).resolve()) if p else ""
    )
    df["row_id"] = np.arange(len(df))
    return df


def _partition(values, fractions, rng):
    values = np.asarray(sorted(set(values)))
    if len(values) < 3:
        raise ValueError("At least 3 distinct entities/pairs required for three splits")
    values = rng.permutation(values)
    n_val = max(1, int(len(values) * fractions[1]))
    n_test = max(1, int(len(values) * fractions[2]))
    if n_val + n_test >= len(values):
        raise ValueError("Fractions leave no training examples")
    n_train = len(values) - n_val - n_test
    return [set(x.tolist()) for x in np.split(values, [n_train, n_train + n_val])]


def make_splits(df, method="random", seed=0, fractions=(0.8, 0.1, 0.1)):
    if method not in SPLITS:
        raise ValueError(f"Unknown split {method}")
    if len(fractions) != 3 or min(fractions) <= 0 or not np.isclose(sum(fractions), 1):
        raise ValueError("Three positive fractions summing to 1 required")
    rng = np.random.default_rng(seed)
    if method == "random":
        groups = _partition(df.row_id, fractions, rng)
        masks = [df.row_id.isin(g) for g in groups]
    elif method == "cold_both":
        proteins = _partition(df.protein_key, fractions, rng)
        ligands = _partition(df.ligand_key, fractions, rng)
        # Cross-partition pairs are excluded, never put into the training set.
        masks = [df.protein_key.isin(p) & df.ligand_key.isin(l)
                 for p, l in zip(proteins, ligands)]
    else:
        key = "protein_key" if method == "cold_protein" else "ligand_key"
        groups = _partition(df[key], fractions, rng)
        masks = [df[key].isin(g) for g in groups]
    result = {name: df.loc[mask].copy().reset_index(drop=True)
              for name, mask in zip(("train", "val", "test"), masks)}
    if any(frame.empty for frame in result.values()):
        raise ValueError("An empty split was generated; adjust fractions or seed explicitly")
    for a, b in combinations(result.values(), 2):
        assert set(a.row_id).isdisjoint(b.row_id)
        if method in ("cold_protein", "cold_both"):
            assert set(a.protein_key).isdisjoint(b.protein_key)
        if method in ("cold_ligand", "cold_both"):
            assert set(a.ligand_key).isdisjoint(b.ligand_key)
    used = sum(map(len, result.values()))
    print(f"Split {method}: " + ", ".join(f"{k}={len(v)}" for k, v in result.items())
          + f", excluded={len(df) - used}")
    return result


class AffinityDataset(Dataset):
    def __init__(self, frame):
        self.records = frame.to_dict("records")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def collate_records(records):
    return records, torch.tensor([r["affinity"] for r in records], dtype=torch.float32)


def make_loader(frame, batch_size, shuffle, seed, num_workers=0):
    # No stochastic transforms run in workers; all encoder calls run in main process.
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(AffinityDataset(frame), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_records, generator=generator, num_workers=num_workers)
