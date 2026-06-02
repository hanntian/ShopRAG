"""Extract Amazon Reviews 2023 subset by ESCI small_version ASIN set.

Usage:
    python -m ingestion.amazon_reviews All_Beauty Electronics
    python -m ingestion.amazon_reviews All_Beauty --kind meta
"""
import argparse
import gzip
import json
from pathlib import Path

import requests
from tqdm import tqdm

from ingestion.esci import load_esci_products

OUT_DIR = Path("data/amazon_reviews_2023")
HF_BASE = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw"
# Review timestamp window: 2022-01-01 to 2023-09-30 (UTC), milliseconds.
TS_MIN = 1640995200000  # 2022-01-01
TS_MAX = 1696118400000  # 2023-10-01 (exclusive)


def extract(categories, kind="review", out_dir=OUT_DIR):
    # ESCI product_id == Amazon ASIN; matched against review.parent_asin.
    asins = set(load_esci_products(small=True)["product_id"])
    print(f"ESCI small_version ASINs: {len(asins)}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for cat in categories:
        fname = f"meta_{cat}.jsonl" if kind == "meta" else f"{cat}.jsonl"
        url = f"{HF_BASE}/{kind}_categories/{fname}"
        out = out_dir / f"{kind}_{cat}.jsonl.gz"
        n_in = n_out = 0
        with requests.get(url, stream=True) as resp:
            resp.raise_for_status()
            with gzip.open(out, "wt") as f:
                for line in tqdm(resp.iter_lines(decode_unicode=True), desc=cat):
                    if not line:
                        continue
                    n_in += 1
                    r = json.loads(line)
                    if r.get("parent_asin") not in asins:
                        continue
                    if kind == "review":
                        ts = r.get("timestamp", 0)
                        if not (TS_MIN <= ts < TS_MAX):
                            continue
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
