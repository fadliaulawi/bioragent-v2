"""
Download Hugging Face dataset vladimire/geneturing and write each config (subset) to disk.

Each subset is a separate "configuration" of the dataset; the public viewer uses split ``test``
for all of them.
"""

from __future__ import annotations

from pathlib import Path

REPO_ID = "vladimire/geneturing"

# Subset names match the dataset configs on the Hub (including typos in *_aligment).
GENETURING_SUBSETS: tuple[str, ...] = (
    "SNP_location",
    "TF_regulation",
    "all",
    "gene_SNP_association",
    "gene_alias",
    "gene_disease_association",
    "gene_location",
    "gene_name_conversion",
    "gene_name_extraction",
    "gene_ontology",
    "human_genome_DNA_aligment",
    "multi-species_DNA_aligment",
    "protein-coding_genes",
)

_DEFAULT_OUT = Path(__file__).resolve().parent / "geneturing"


def _safe_dir_name(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")


def download_subset(subset: str, out_root: Path) -> Path:
    from datasets import load_dataset

    out_root.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(REPO_ID, name=subset, split="test")
    stem = _safe_dir_name(subset)
    path = out_root / f"{stem}.csv"
    ds.to_csv(str(path))
    return path


def main(argv: list[str] | None = None) -> int:
    _ = argv
    paths = []
    for name in GENETURING_SUBSETS:
        paths.append(download_subset(name, _DEFAULT_OUT))

    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
