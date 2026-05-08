"""Data-side EDA: lengths, ChatML structure, and per-class signals."""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

DATA_FILE = "./data/dataset.csv"
TEST_FILE = "./data/test.csv"
MODEL = "Qwen/Qwen2.5-0.5B"
MAX_LEN = 512


def main() -> None:
    df = pd.read_csv(DATA_FILE)
    df_test = pd.read_csv(TEST_FILE)
    tok = AutoTokenizer.from_pretrained(MODEL)

    # Token lengths
    df["prompt_tok"] = df["prompt"].apply(lambda s: len(tok.encode(s)))
    df["response_tok"] = df["response"].apply(lambda s: len(tok.encode(s)))
    df["combined_tok"] = df.apply(
        lambda r: len(tok.encode(f"{r['prompt']}{r['response']}")), axis=1
    )
    df["truncated"] = df["combined_tok"] > MAX_LEN
    df["response_visible_tok"] = df.apply(
        lambda r: max(0, MAX_LEN - r["prompt_tok"]) if r["truncated"] else r["response_tok"],
        axis=1,
    )

    print("=" * 72)
    print(" LENGTH STATISTICS BY CLASS")
    print("=" * 72)
    for label, name in [(0.0, "Truthful"), (1.0, "Hallucinated")]:
        sub = df[df["label"] == label]
        print(f"\n{name:14s} n={len(sub):4d}  ({len(sub)/len(df)*100:.1f}%)")
        for col in ["prompt_tok", "response_tok", "combined_tok", "response_visible_tok"]:
            v = sub[col]
            print(f"  {col:24s} mean={v.mean():6.1f}  median={v.median():6.1f}  "
                  f"p25={v.quantile(0.25):5.1f}  p75={v.quantile(0.75):5.1f}  "
                  f"max={v.max():5.0f}")
        trunc = sub["truncated"].sum()
        print(f"  truncated:               {trunc:4d}  ({trunc/len(sub)*100:.1f}%)")

    # Cross-tab: truncation × class
    print("\n" + "=" * 72)
    print(" TRUNCATION × CLASS")
    print("=" * 72)
    ct = pd.crosstab(df["truncated"], df["label"], margins=True, normalize=False)
    print(ct)
    print("\nProportions within truncated/non-truncated:")
    ct_norm = pd.crosstab(df["truncated"], df["label"], normalize="index")
    print(ct_norm)

    # Test-set length distribution
    print("\n" + "=" * 72)
    print(" TEST SET LENGTHS (unlabeled, n=100)")
    print("=" * 72)
    df_test["prompt_tok"] = df_test["prompt"].apply(lambda s: len(tok.encode(s)))
    df_test["response_tok"] = df_test["response"].apply(lambda s: len(tok.encode(s)))
    df_test["combined_tok"] = df_test.apply(
        lambda r: len(tok.encode(f"{r['prompt']}{r['response']}")), axis=1
    )
    df_test["truncated"] = df_test["combined_tok"] > MAX_LEN
    print(f"prompt_tok    mean={df_test['prompt_tok'].mean():.1f}  "
          f"median={df_test['prompt_tok'].median():.1f}")
    print(f"response_tok  mean={df_test['response_tok'].mean():.1f}  "
          f"median={df_test['response_tok'].median():.1f}")
    print(f"truncated:    {df_test['truncated'].sum()}/{len(df_test)} "
          f"({df_test['truncated'].mean()*100:.1f}%)")

    # Response text patterns
    print("\n" + "=" * 72)
    print(" RESPONSE TEXT PATTERNS")
    print("=" * 72)
    patterns = {
        "I_dont_know": r"\b(I don'?t know|not sure|cannot|can'?t answer|insufficient)",
        "starts_yes_no": r"^\s*(Yes|No)\b",
        "has_numbers": r"\d",
        "starts_uppercase_word": r"^\s*[A-Z]",
        "has_question_mark": r"\?",
        "has_year": r"\b(19|20)\d{2}\b",
        "first_person": r"\b(I|me|my|mine)\b",
        "ends_with_period": r"\.\s*<\|endoftext\|>?\s*$",
    }
    rows = []
    for name, pat in patterns.items():
        regex = re.compile(pat, re.IGNORECASE if name != "starts_uppercase_word" else 0)
        df[f"p_{name}"] = df["response"].apply(lambda s: bool(regex.search(str(s))))
        truthful_rate = df[df["label"] == 0.0][f"p_{name}"].mean()
        hallu_rate = df[df["label"] == 1.0][f"p_{name}"].mean()
        diff = hallu_rate - truthful_rate
        rows.append((name, truthful_rate, hallu_rate, diff))
    print(f"  {'pattern':22s} {'truthful':>10s} {'hallu':>10s} {'diff':>8s}")
    for name, t, h, d in sorted(rows, key=lambda x: -abs(x[3])):
        print(f"  {name:22s} {t*100:9.1f}% {h*100:9.1f}% {d*100:+8.2f}pp")

    # First-token of response per class
    print("\n" + "=" * 72)
    print(" FIRST 5 RESPONSE TOKENS BY CLASS (top distinctive)")
    print("=" * 72)
    df["resp_first_token"] = df["response"].apply(
        lambda s: str(s).strip().split()[0] if str(s).strip() else "<EMPTY>"
    )
    fw_counts = df.groupby(["resp_first_token", "label"]).size().unstack(fill_value=0)
    fw_counts["total"] = fw_counts.sum(axis=1)
    fw_counts = fw_counts[fw_counts["total"] >= 10]
    if len(fw_counts) > 0:
        fw_counts["hallu_rate"] = fw_counts.get(1.0, 0) / fw_counts["total"]
        baseline = (df["label"] == 1.0).mean()
        fw_counts["lift"] = fw_counts["hallu_rate"] - baseline
        print(f"baseline hallu rate: {baseline*100:.1f}%")
        print(fw_counts.sort_values("lift").head(10))
        print("...")
        print(fw_counts.sort_values("lift", ascending=False).head(10))

    # Prompt structure: count <|im_start|> markers
    print("\n" + "=" * 72)
    print(" CHATML STRUCTURE")
    print("=" * 72)
    for label, name in [(0.0, "truthful"), (1.0, "hallu")]:
        sub = df[df["label"] == label]["prompt"]
        n_user_turns = sub.apply(lambda s: str(s).count("<|im_start|>user")).mean()
        n_assistant_turns = sub.apply(lambda s: str(s).count("<|im_start|>assistant")).mean()
        print(f"{name:10s}  avg user-turns={n_user_turns:.2f}  "
              f"avg assistant-turns={n_assistant_turns:.2f}")

    # Save with all features for later use
    df.to_csv("eda_dataset_with_features.csv", index=False)
    print("\nSaved: eda_dataset_with_features.csv")


if __name__ == "__main__":
    main()
