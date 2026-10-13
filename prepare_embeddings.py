"""Precompute unique frozen backbone embeddings, not learned patches."""
import argparse

from src.datasets import read_table
from src.ligand_encoder import LigandEncoder
from src.protein_encoder import ProteinEncoder
from src.utils import choose_device, load_config, seed_everything, save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--dataset", choices=["davis", "kiba", "bindingdb"], default="davis")
    p.add_argument("--limit", type=int, help="Debug: limit unique molecules/proteins")
    args = p.parse_args()
    config = load_config(args.config)
    seed_everything(config["seed"])
    device = choose_device(config["device"])
    df = read_table(f"{config['data']['root']}/{args.dataset}/pairs.csv")
    identities = {}
    for name, cls, field in [("protein", ProteinEncoder, "sequence"), ("ligand", LigandEncoder, "smiles")]:
        settings = config["encoders"][name]
        if not settings["freeze"] or not settings.get("cache"):
            raise ValueError("Precomputation requires freeze=true and a cache path")
        encoder = cls(**settings).to(device).eval()
        values = df[field].unique()
        if args.limit:
            values = values[:args.limit]
        for i, value in enumerate(values):
            output = encoder(value)
            if i == 0 or (i + 1) % 25 == 0 or i + 1 == len(values):
                print(f"{name} {i + 1}/{len(values)} shape={tuple(output.shape)}", flush=True)
        identities[name] = encoder.identity
        del encoder
    save_json(identities, "embeddings/backbone_provenance.json")


if __name__ == "__main__":
    main()
