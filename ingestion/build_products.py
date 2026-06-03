"""Merge ESCI products + AR2023 meta into a single parent catalog.

Rules:
- For overlapping fields (title, description, bullets, brand, color),
  take the longest non-empty candidate across sources.
- Convert meta list fields to strings:
    description: "\\n\\n".join(...)  (preserves paragraph boundary for chunking)
    features:    "\\n".join(...)     (one bullet per line, easy .split later)
- Drop bullets shorter than MIN_BULLET_CHARS.
- Keep all meta-only structured fields (price, rating, categories, ...) as-is.

Usage:
    python -m ingestion.build_products Appliances
"""
import argparse
import gzip
import json
from pathlib import Path

import pyarrow.parquet as pq

DATA_DIR = Path("data")
META_DIR = DATA_DIR / "amazon_reviews_2023"
ESCI_PRODUCTS_PATH = DATA_DIR / "esci" / "shopping_queries_dataset_products.parquet"

MIN_BULLET_CHARS = 30


def _longest(*candidates):
    """Pick the longest string from candidates; missing/None treated as ''."""
    return max((c or "" for c in candidates), key=len)


def _clean_bullets(bullets):
    """Strip + drop bullets shorter than MIN_BULLET_CHARS."""
    return [b.strip() for b in bullets if b and len(b.strip()) >= MIN_BULLET_CHARS]


def _load_esci_subset(asin_set):
    """Stream ESCI products by row group; return {asin: text_fields_dict}."""
    cols = ["product_id", "product_title", "product_description",
            "product_bullet_point", "product_brand", "product_color",
            "product_locale"]
    pf = pq.ParquetFile(ESCI_PRODUCTS_PATH)
    out = {}
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=cols).to_pydict()
        for i, pid in enumerate(t["product_id"]):
            if t["product_locale"][i] != "us":
                continue
            if pid in asin_set and pid not in out:
                out[pid] = {
                    "title":       t["product_title"][i] or "",
                    "description": t["product_description"][i] or "",
                    "bullets":     t["product_bullet_point"][i] or "",
                    "brand":       t["product_brand"][i] or "",
                    "color":       t["product_color"][i] or "",
                }
    return out


def merge_product(esci, meta):
    """Merge one product into a single dict."""
    e = esci or {}
    details = meta.get("details") or {}

    # Title
    title = _longest(e.get("title"), meta.get("title"))

    # Description: ESCI string vs meta list -> join with "\n\n"
    meta_desc = "\n\n".join(
        x.strip() for x in (meta.get("description") or []) if x and x.strip()
    )
    description = _longest(e.get("description"), meta_desc)

    # Bullets: normalize both sides through the same cleaner, then compare
    esci_bullets = _clean_bullets((e.get("bullets") or "").split("\n"))
    meta_bullets = _clean_bullets(meta.get("features") or [])
    bullets = _longest("\n".join(esci_bullets), "\n".join(meta_bullets))

    # Brand: 2 candidates (ESCI brand, meta details.Brand)
    brand = _longest(e.get("brand"), details.get("Brand"))

    return {
        "parent_asin": meta["parent_asin"],
        # text fields (merged)
        "title": title,
        "description": description,
        "features": bullets,
        "brand": brand,
        # "color": e.get("color"),
        # meta-only structured fields (kept as-is for parent payload)
        "main_category": meta.get("main_category", ""),
        "categories": meta.get("categories") or [],
        "details": details,
        "average_rating": meta.get("average_rating"),
        "rating_number": meta.get("rating_number"),
        "price": meta.get("price"),
        "images": meta.get("images") or [],
        "videos": meta.get("videos") or [],
    }


def build(category, out_dir=META_DIR):
    meta_path = META_DIR / f"meta_{category}.jsonl.gz"
    out_path = out_dir / f"merged_meta_{category}.jsonl.gz"

    meta_rows = [json.loads(l) for l in gzip.open(meta_path, "rt")]
    asin_set = {r["parent_asin"] for r in meta_rows}
    print(f"meta rows: {len(meta_rows)}")

    esci = _load_esci_subset(asin_set)
    print(f"ESCI rows matched in subset: {len(esci)}/{len(meta_rows)}")

    n = 0
    with gzip.open(out_path, "wt") as f:
        for r in meta_rows:
            merged = merge_product(esci.get(r["parent_asin"]), r)
            f.write(json.dumps(merged) + "\n")
            n += 1

    print(f"wrote {n} -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("category", help="e.g. Appliances")
    args = p.parse_args()
    build(args.category)
# parent_asin      str    'B083Q6Y54F'
# title            str    'G.a HOMEFAVOR Cold Brew Coffee Infuser 64oz (2 Quart), Stainless Steel Filter Kit for Wide Mason Ja
# description      str    'INCLUDED:<br> - 61oz Stainless steel filter<br> <br> EASY TO USE<br> 1. Pour coffee grounds and wat
# features         str    'COMPLETE COLD BREW SYSTEM: The Cold Brew Kit is designed from an extra heavy-duty 304 Stainless Ste
# brand            str    'G.a HOMEFAVOR'
# main_category    str    'Amazon Home'
# categories       list   ['Small Appliance Parts & Accessories', 'Coffee & Espresso Machine Parts & Accessories', 'Coffee Mac
# details          dict   {'Package Dimensions': '8.46 x 3.43 x 3.39 inches', 'Item Weight': '3.2 ounces', 'Manufacturer': 'Ga
# average_rating   float  4.5
# rating_number    int    326
# price            NoneType None
# images           list   [{'thumb': 'https://m.media-amazon.com/images/I/41EPlLNVx6L._AC_US75_.jpg', 'large': 'https://m.medi
# videos           list   [{'title': 'Trendy Mason Jar Lid Used with Cold Brew Coffee System', 'url': 'https://www.amazon.com/