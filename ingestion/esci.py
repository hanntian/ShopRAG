import pandas as pd
import requests
from pathlib import Path


ESCI_BASE = "https://github.com/amazon-science/esci-data/raw/main/shopping_queries_dataset"
DATA_DIR = Path("data/esci")


def _download(filename):
    path = DATA_DIR / filename
    if not path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(requests.get(f"{ESCI_BASE}/{filename}").content)
    return path


def load_esci_products(small=True):
    """Load the ESCI products (product catalog)."""
    df = pd.read_parquet(_download("shopping_queries_dataset_products.parquet"))
    df = df[df["product_locale"] == "us"]
    if small:
        examples = pd.read_parquet(_download("shopping_queries_dataset_examples.parquet"))
        ids = examples.loc[examples["small_version"] == 1, "product_id"].unique()
        df = df[df["product_id"].isin(ids)]
    return df.reset_index(drop=True)


def load_esci_examples(small=True, split="test"):
    """Load the ESCI examples (benchmark dataset)."""

    df = pd.read_parquet(_download("shopping_queries_dataset_examples.parquet"))
    df = df[df["product_locale"] == "us"]
    if small:
        df = df[df["small_version"] == 1]
    if split:
        df = df[df["split"] == split]
    return df.reset_index(drop=True)


if __name__ == "__main__":
    products = load_esci_products()
    print("products shape:", products.shape) #(485495, 7)
    print("products columns:", products.columns.tolist())# ['product_id', 'product_title', 'product_description', 'product_bullet_point', 'product_brand', 'product_color', 'product_locale']
    print(products.head(3).to_string())

    print()

    examples = load_esci_examples()
    print("examples shape:", examples.shape) #(181701, 9)
    print("examples columns:", examples.columns.tolist()) #examples columns: ['example_id', 'query', 'query_id', 'product_id', 'product_locale', 'esci_label', 'small_version', 'large_version', 'split']
    print("label counts:\n", examples["esci_label"].value_counts())
    print(examples[["query", "product_id", "esci_label"]].head(5).to_string())

    # ---------------------------------------------------------------
    # Length distribution: decide whether chunking is even needed.
    # Rough rule: 1 token ~= 4 chars (English). 512 tokens ~= 2000 chars.
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Length distribution (chars) per field")
    print("=" * 60)

    text_fields = ["product_title", "product_description", "product_bullet_point"]
    df = products.copy()
    for f in text_fields:
        df[f] = df[f].fillna("")
        df[f + "_len"] = df[f].str.len()

    # Whole-product serialized length (what a single embedding would see)
    df["whole_product_len"] = (
        df["product_title"].str.len()
        + df["product_description"].str.len()
        + df["product_bullet_point"].str.len()
    )

    quantiles = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    summary = df[[f + "_len" for f in text_fields] + ["whole_product_len"]].quantile(quantiles)
    summary.index = ["min", "p25", "p50", "p75", "p90", "p95", "p99", "max"]
    print(summary.round(0).astype(int).to_string())

    print("\nNon-empty rate per field:")
    for f in text_fields:
        rate = (df[f + "_len"] > 0).mean()
        print(f"  {f:25s} {rate:.1%}")

    # Buckets relevant to chunking decisions (whole-product chars).
    print("\nWhole-product length buckets (chars / approx tokens):")
    buckets = [
        ("0-500 chars (~0-125 tok)",     0,    500),
        ("500-2000 chars (~125-500)",    500,  2000),
        ("2000-4000 chars (~500-1k)",    2000, 4000),
        ("4000-8000 chars (~1k-2k)",     4000, 8000),
        ("8000+ chars (>2k tok)",        8000, 10**9),
    ]
    n = len(df)
    for label, lo, hi in buckets:
        cnt = ((df["whole_product_len"] >= lo) & (df["whole_product_len"] < hi)).sum()
        print(f"  {label:30s} {cnt:>7d}  ({cnt / n:.1%})")

    # Bullet-point structure: how many bullets per product (split on newline).
    bullets_per_product = df["product_bullet_point"].str.split("\n").map(
        lambda xs: sum(1 for x in xs if x.strip())
    )
    print("\nBullets per product:")
    print(bullets_per_product.describe(percentiles=[0.5, 0.9, 0.99]).round(1).to_string())
