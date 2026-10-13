"""Create a small Davis subset for runtime validation, never a benchmark result."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

from src.utils import load_config, save_json


def main():
    df = pd.read_csv("data/processed/davis/pairs.csv", keep_default_na=False)
    proteins = sorted(df.sequence.unique(), key=lambda s: (len(s), s))[:12]
    ligands = sorted(df.smiles.unique(), key=lambda s: (len(s), s))[:8]
    subset = df.loc[df.sequence.isin(proteins) & df.smiles.isin(ligands)]
    root = Path("data/smoke/davis")
    root.mkdir(parents=True, exist_ok=True)
    subset.to_csv(root / "pairs.csv", index=False)
    save_json({"purpose": "runtime smoke, biased subset, NOT a scientific evaluation",
               "rows": len(subset), "proteins": len(proteins), "ligands": len(ligands)},
              root / "provenance.json")
    config = load_config("config.yaml")
    config["data"]["root"] = "data/smoke"
    config["data"]["fractions"] = [.6, .2, .2]
    config["training"].update(epochs=2, output="runs/davis_smoke", batch_size=8)
    (root.parent / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"Prepared {len(subset)} genuine Davis pairs; use --config data/smoke/config.yaml")


if __name__ == "__main__":
    main()
