from __future__ import annotations

import hashlib
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from ..datasets import _partition, make_splits
from ..utils import digest


def validate_trajectory(arrays):
    required = {"positions", "atomic_numbers", "ligand_mask", "time", "frame_indices", "units", "time_unit"}
    if not required.issubset(arrays):
        raise ValueError(f"Trajectory fields missing: {required - set(arrays)}")
    xyz, z, ligand = arrays["positions"], arrays["atomic_numbers"], arrays["ligand_mask"]
    if xyz.ndim != 3 or xyz.shape[-1] != 3 or len(xyz) < 2 or not np.isfinite(xyz).all():
        raise ValueError("positions must be finite [T>=2,N,3]; static structures are not MD")
    if z.shape != (xyz.shape[1],) or z.dtype.kind not in "iu" or (z < 1).any() or (z > 118).any():
        raise ValueError("atomic_numbers must be integer [N] in [1,118]")
    if ligand.shape != z.shape or ligand.dtype.kind != "b" or not ligand.any() or ligand.all():
        raise ValueError("Boolean ligand_mask must identify both protein and ligand atoms")
    times, indices = arrays["time"], arrays["frame_indices"]
    if times.shape != (len(xyz),) or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
        raise ValueError("Trajectory time must be finite and strictly increasing")
    if indices.shape != times.shape or indices.dtype.kind not in "iu" or (np.diff(indices) <= 0).any():
        raise ValueError("frame_indices must preserve strictly increasing source frame order")
    if str(arrays["units"]) != "angstrom" or str(arrays["time_unit"]) not in ("ps", "ns", "frame_index"):
        raise ValueError("Coordinates require Angstrom and an explicit time unit")
    if "atom_ids" in arrays and (arrays["atom_ids"].shape != z.shape or len(np.unique(arrays["atom_ids"])) != len(z)):
        raise ValueError("atom_ids must be unique and aligned with the fixed atom axis")
    if "reference_positions" in arrays and (arrays["reference_positions"].shape != xyz.shape[1:]
                                             or not np.isfinite(arrays["reference_positions"]).all()):
        raise ValueError("reference_positions must be finite [N,3]")


def read_trajectory(path, max_frames=None, stride=1):
    if stride < 1 or max_frames is not None and max_frames < 2:
        raise ValueError("Require stride>=1 and max_frames>=2")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    validate_trajectory(arrays)
    selected = np.arange(0, len(arrays["positions"]), stride)
    if max_frames is not None and len(selected) > max_frames:
        selected = selected[np.linspace(0, len(selected) - 1, max_frames).round().astype(int)]
    if len(selected) < 2:
        raise ValueError("Frame selection leaves fewer than two temporal observations")
    result = {"positions": torch.as_tensor(arrays["positions"][selected], dtype=torch.float32),
              "atomic_numbers": torch.as_tensor(arrays["atomic_numbers"], dtype=torch.long),
              "ligand_mask": torch.as_tensor(arrays["ligand_mask"], dtype=torch.bool),
              "time": torch.as_tensor(arrays["time"][selected], dtype=torch.float64),
              "frame_indices": torch.as_tensor(arrays["frame_indices"][selected], dtype=torch.long),
              "atom_ids": torch.as_tensor(arrays.get("atom_ids", np.arange(len(arrays["atomic_numbers"]))), dtype=torch.long),
              "reference_positions": torch.as_tensor(arrays.get("reference_positions", arrays["positions"][0]), dtype=torch.float32),
              "time_unit": str(arrays["time_unit"])}
    return result


def read_manifest(path):
    path = Path(path)
    rows = pd.read_csv(path, keep_default_na=False)
    required = {"complex_id", "trajectory_id", "protein_id", "ligand_id", "trajectory_path", "affinity", "target_type", "source"}
    if not required.issubset(rows) or rows.empty:
        raise ValueError(f"Nonempty MD manifest required; missing {required - set(rows)}")
    for key in ("complex_id", "trajectory_id", "protein_id", "ligand_id", "trajectory_path"):
        rows[key] = rows[key].astype(str)
        if rows[key].str.strip().eq("").any():
            raise ValueError(f"Empty {key}")
    if rows.trajectory_id.duplicated().any():
        raise ValueError("trajectory_id must be unique; do not count a trajectory twice")
    if rows.target_type.nunique() != 1 or rows.target_type.iloc[0] not in ("pKd", "pKi", "pIC50"):
        raise ValueError("Use one experimental endpoint per dataset; no mixed Kd/Ki/IC50 or MMGBSA labels")
    rows["affinity"] = pd.to_numeric(rows.affinity, errors="raise")
    if not np.isfinite(rows.affinity).all():
        raise ValueError("Missing/nonfinite experimental affinity")
    for key in ("sequence", "smiles", "protein_group", "ligand_group"):
        if key not in rows:
            rows[key] = ""
    rows["trajectory_path"] = rows.trajectory_path.map(lambda p: str((path.parent / p).resolve()))
    if rows.trajectory_path.duplicated().any():
        raise ValueError("A trajectory file is reused under several IDs")
    complexes = []
    for complex_id, group in rows.groupby("complex_id", sort=True):
        for key in ("affinity", "target_type", "protein_id", "ligand_id", "sequence", "smiles", "protein_group", "ligand_group"):
            if group[key].nunique() != 1:
                raise ValueError(f"Replicas of {complex_id} disagree on {key}")
        first = group.iloc[0].to_dict()
        # Stable grouping supplied by the provider; canonical sequence/SMILES can override aliases.
        p = first["sequence"] or first["protein_group"] or first["protein_id"]
        l = first["smiles"] or first["ligand_group"] or first["ligand_id"]
        first.update(protein_key=digest(p), ligand_key=digest(l), row_id=len(complexes),
                     protein_identity_verified=bool(first["sequence"] or first["protein_group"]),
                     ligand_identity_verified=bool(first["smiles"] or first["ligand_group"]),
                     trajectories=group.to_dict("records"))
        complexes.append(first)
    return pd.DataFrame(complexes)


def split_complexes(frame, method="random", seed=0, fractions=(.8, .1, .1)):
    if len(fractions) != 3 or min(fractions) <= 0 or not np.isclose(sum(fractions), 1):
        raise ValueError("Require three positive split fractions summing to 1")
    if method in ("cold_protein", "cold_both") and not frame.protein_identity_verified.all():
        raise ValueError("Cold-protein/both requires verified protein_group or sequence, not a structure ID")
    if method in ("cold_ligand", "cold_both") and not frame.ligand_identity_verified.all():
        raise ValueError("Cold-ligand/both requires verified ligand_group or canonical SMILES; generic IDs like '4-mer' are insufficient")
    if method != "random":
        return make_splits(frame, method, seed, fractions)
    # Even ambiguous provider aliases are grouped conservatively in random splits.
    keys = frame.apply(lambda r: r.protein_key + ":" + r.ligand_key, axis=1)
    groups = _partition(keys, fractions, np.random.default_rng(seed))
    splits = {k: frame[keys.isin(g)].copy().reset_index(drop=True)
              for k, g in zip(("train", "val", "test"), groups)}
    for a, b in combinations(splits.values(), 2):
        assert set(a.complex_id).isdisjoint(b.complex_id)
        pa = set(zip(a.protein_key, a.ligand_key)); pb = set(zip(b.protein_key, b.ligand_key))
        assert pa.isdisjoint(pb)
    print("Complex split: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
    return splits


def fingerprint(frame):
    files, complexes = {}, frame.to_dict("records")
    # Hash canonical manifest rows and files, not derived split-validation flags.
    records = [replica for record in complexes for replica in record["trajectories"]]
    for record in complexes:
        for replica in record["trajectories"]:
            p = replica["trajectory_path"]
            if p not in files:
                # Stream large files, never read the full MD dataset into memory.
                sha = hashlib.sha256()
                with open(p, "rb") as f:
                    for block in iter(lambda: f.read(1024 * 1024), b""):
                        sha.update(block)
                files[p] = sha.hexdigest()
    return digest({"records": records, "files": files})


class ComplexDataset(Dataset):
    def __init__(self, frame, config):
        self.records = frame.to_dict("records")
        self.config = config

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = dict(self.records[index])
        record["trajectories"] = [dict(read_trajectory(r["trajectory_path"],
                                     self.config["max_frames"], self.config["frame_stride"]),
                                     trajectory_id=r["trajectory_id"]) for r in record["trajectories"]]
        return record


def collate_complexes(records):
    return records, torch.tensor([r["affinity"] for r in records], dtype=torch.float32)


def loader(frame, config, shuffle=False):
    return DataLoader(ComplexDataset(frame, config["dynamic"]), batch_size=config["training"]["batch_size"],
                      shuffle=shuffle, num_workers=config["training"]["num_workers"], collate_fn=collate_complexes,
                      generator=torch.Generator().manual_seed(config["seed"]))
