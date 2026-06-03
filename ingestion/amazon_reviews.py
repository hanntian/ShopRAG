"""Extract Amazon Reviews 2023 subset by ESCI small_version ASIN set.

Usage:
    python -m ingestion.amazon_reviews All_Beauty Electronics
    python -m ingestion.amazon_reviews All_Beauty --kind meta
"""
import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

from ingestion.esci import load_esci_products

PCTS = [0, 25, 50, 75, 90, 95, 99, 100]
LABELS = ["min", "p25", "p50", "p75", "p90", "p95", "p99", "max"]

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


def stats(categories, kind="review", out_dir=OUT_DIR):
    for cat in categories:
        f = out_dir / f"{kind}_{cat}.jsonl.gz"
        if not f.exists():
            continue
        lens, counts = [], Counter()
        with gzip.open(f, "rt") as fp:
            for line in fp:
                r = json.loads(line)
                lens.append(len(r.get("text") or r.get("title") or ""))
                counts[r.get("parent_asin")] += 1
        cols = {"text_len": lens, "per_parent_asin": list(counts.values())}
        print(f"\n=== {f.name}  rows={sum(counts.values())}  parent_asins={len(counts)} ===")
        print(f"{'':<6}" + "".join(f"{k:>18}" for k in cols))
        for lab, p in zip(LABELS, PCTS):
            print(f"{lab:<6}" + "".join(f"{int(np.percentile(v, p)):>18}" for v in cols.values()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("categories", nargs="+",
                   help="e.g. All_Beauty Electronics Books")
    p.add_argument("--kind", choices=["review", "meta"], default="review")
    p.add_argument("--stats-only", action="store_true")
    args = p.parse_args()
    if not args.stats_only:
        extract(args.categories, args.kind)
    stats(args.categories, args.kind)
