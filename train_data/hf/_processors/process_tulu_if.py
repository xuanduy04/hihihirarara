"""Deduplicate allenai/tulu-3-sft-personas-instruction-following.

Three passes, each reported separately:
  1. exact duplicates   - same prompt after lower-casing and stripping non-alphanumerics
  2. test leakage       - prompt identical (same normalisation) to an IFEval or IFBench prompt
  3. near duplicates    - optional (--near 0.9): character n-gram TF-IDF cosine >= threshold; the first occurrence is
     kept

Output: jsonl with id, prompt, reference (assistant turn), constraints.

Input files are first looked up under --data-dir (default: ../data next to this script, i.e. train_data/hf/data) as
<repo>/<filename>, e.g.
train_data/hf/data/allenai/tulu-3-sft-personas-instruction-following/data/train-00000-of-00001.parquet; only a file
missing there is downloaded from the HuggingFace Hub.

    pip install pandas pyarrow scikit-learn huggingface_hub tqdm
    python dedup_tulu_if.py --out tulu_if_dedup.jsonl            # exact + leakage
    python dedup_tulu_if.py --out tulu_if_dedup.jsonl --near 0.9 # plus near-duplicates
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

TULU = ("allenai/tulu-3-sft-personas-instruction-following", "data/train-00000-of-00001.parquet")
TESTS = {
    "IFEval": ("google/IFEval", "ifeval_input_data.jsonl"),
    "IFBench": ("allenai/IFBench_test", "data/train-00000-of-00001.parquet"),
}
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def load(repo, filename, data_dir):
    path = os.path.join(data_dir, repo, filename)
    if not os.path.exists(path):
        path = hf_hub_download(repo, filename, repo_type="dataset")
    return pd.read_json(path, lines=True) if filename.endswith(".jsonl") else pd.read_parquet(path)


def near_duplicate_mask(texts: pd.Series, threshold: float, chunk: int = 2000) -> np.ndarray:
    """True for rows whose cosine similarity to an EARLIER row is >= threshold."""
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2).fit_transform(texts)
    dup = np.zeros(X.shape[0], dtype=bool)
    for start in tqdm(range(0, X.shape[0], chunk), desc="near-duplicates", unit="chunk"):
        S = (X[start : start + chunk] @ X.T).tocoo()
        hit = (S.data >= threshold) & (S.col < S.row + start)  # only earlier rows count
        dup[np.unique(S.row[hit]) + start] = True
    return dup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "tulu_if_dedup.jsonl"))
    ap.add_argument(
        "--data-dir", default=DATA_DIR, help="checked first for <repo>/<filename> before the HuggingFace Hub"
    )
    ap.add_argument("--near", type=float, default=None, help="cosine threshold for near-duplicate removal, e.g. 0.9")
    ap.add_argument("--skip-test-check", action="store_true")
    ap.add_argument(
        "--yes",
        "-y",
        action="store_true",
        default=False,
        help="Preemptively confirm overwrite of the output file (default: false).",
    )
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.yes:
        response = input(f"{args.out} already exists. Override? Type [y]es/[n]o: ").strip().lower()
        if response not in ("y", "yes"):
            print("Existing output file will not be overridden; exiting.")
            sys.exit(0)

    df = load(*TULU, args.data_dir)
    df["reference"] = df.messages.map(lambda m: m[-1]["content"])
    df["_n"] = df.prompt.map(norm)
    print(f"rows: {len(df)}")

    before = len(df)
    df = df.drop_duplicates("_n", keep="first")
    print(f"exact duplicates removed: {before - len(df)}")

    if not args.skip_test_check:
        for name, (repo, filename) in TESTS.items():
            test_prompts = set(load(repo, filename, args.data_dir).prompt.map(norm))
            leak = df._n.isin(test_prompts)
            print(f"identical to a {name} prompt: {int(leak.sum())}")
            df = df[~leak]

    if args.near is not None:
        dup = near_duplicate_mask(df.prompt.astype(str), args.near)
        print(f"near duplicates removed (cosine >= {args.near}): {int(dup.sum())}")
        df = df[~dup]

    print(f"kept: {len(df)}")
    df[["id", "prompt", "reference", "constraints"]].reset_index(drop=True).to_json(
        args.out, orient="records", lines=True, force_ascii=False
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
