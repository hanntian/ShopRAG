"""Extract Amazon Reviews 2023 subset by ESCI small_version ASIN set.

Usage:
    python -m ingestion.amazon_reviews All_Beauty Electronics
    python -m ingestion.amazon_reviews All_Beauty --kind meta
"""
import argparse
import gzip
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from ingestion.esci import load_esci_products

OUT_DIR = Path("data/amazon_reviews_2023")


def extract(categories, kind="review", out_dir=OUT_DIR):
    asins = set(load_esci_products(small=True)["product_id"])
    print(f"ESCI small_version ASINs: {len(asins)}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for cat in categories:
        config = f"raw_{kind}_{cat}"
        out = out_dir / f"{kind}_{cat}.jsonl.gz"
        ds = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023", config,
            split="full", streaming=True, trust_remote_code=True,
        )
        n_in = n_out = 0
        with gzip.open(out, "wt") as f:
            for r in tqdm(ds, desc=cat):
                n_in += 1
                if r.get("parent_asin") in asins:
                    f.write(json.dumps(r) + "\n")
                    n_out += 1
        print(f"[{cat}] kept {n_out}/{n_in} -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("categories", nargs="+",
                   help="e.g. All_Beauty Electronics Books")
    p.add_argument("--kind", choices=["review", "meta"], default="review")
    args = p.parse_args()
    extract(args.categories, args.kind)
