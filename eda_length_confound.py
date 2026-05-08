"""Quantify the length confound: how much accuracy comes from length alone?"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer

DATA_FILE = "./data/dataset.csv"
TEST_FILE = "./data/test.csv"
MODEL = "Qwen/Qwen2.5-0.5B"
MAX_LEN = 512


def main() -> None:
    df = pd.read_csv(DATA_FILE)
    df_test = pd.read_csv(TEST_FILE)
    tok = AutoTokenizer.from_pretrained(MODEL)

    df["prompt_tok"] = df["prompt"].apply(lambda s: len(tok.encode(s)))
    df["response_tok"] = df["response"].apply(lambda s: len(tok.encode(s)))
    df["combined_tok"] = df["prompt_tok"] + df["response_tok"]
    df["truncated"] = df["combined_tok"] > MAX_LEN
    df["response_visible"] = df.apply(
        lambda r: max(0, MAX_LEN - r["prompt_tok"]) if r["truncated"]
        else r["response_tok"],
        axis=1,
    )
    df["log_response"] = np.log1p(df["response_tok"])

    y = df["label"].astype(int).to_numpy()

    # ---- 1. Length-only baseline (single feature) ----
    print("=" * 72)
    print(" LENGTH-ONLY BASELINE (5-fold CV)")
    print("=" * 72)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for feat_name in ["response_tok", "log_response", "combined_tok", "truncated"]:
        X = df[[feat_name]].to_numpy(dtype=float)
        accs, aurocs = [], []
        for tr, te in skf.split(X, y):
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf.fit(X[tr], y[tr])
            pred = clf.predict(X[te])
            prob = clf.predict_proba(X[te])[:, 1]
            accs.append(accuracy_score(y[te], pred))
            aurocs.append(roc_auc_score(y[te], prob))
        print(f"  {feat_name:18s}  Acc={np.mean(accs)*100:5.2f}%  "
              f"AUROC={np.mean(aurocs)*100:5.2f}%")

    # All 4 length features combined
    print("\n  -- combined 4 length features --")
    X = df[["response_tok", "log_response", "combined_tok", "truncated"]].to_numpy(dtype=float)
    accs, aurocs = [], []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(X[tr], y[tr])
        accs.append(accuracy_score(y[te], clf.predict(X[te])))
        aurocs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    print(f"  combined            Acc={np.mean(accs)*100:5.2f}%  "
          f"AUROC={np.mean(aurocs)*100:5.2f}%")

    # ---- 2. Optimal threshold on response length alone ----
    print("\n" + "=" * 72)
    print(" THRESHOLD-ON-LENGTH OPTIMAL ACCURACY (no model at all)")
    print("=" * 72)
    for col in ["response_tok", "combined_tok"]:
        best_t, best_acc = -1, 0
        for t in np.unique(df[col].values):
            pred = (df[col] >= t).astype(int)
            acc = accuracy_score(y, pred)
            if acc > best_acc:
                best_acc = acc
                best_t = t
        print(f"  {col:18s}  best_threshold>={best_t:5.0f}  Acc={best_acc*100:.2f}%")

    # Reverse direction
    for col in ["response_tok", "combined_tok"]:
        best_t, best_acc = -1, 0
        for t in np.unique(df[col].values):
            pred = (df[col] < t).astype(int)
            acc = accuracy_score(y, pred)
            if acc > best_acc:
                best_acc = acc
                best_t = t

    # ---- 3. Test-set length distribution match ----
    print("\n" + "=" * 72)
    print(" TRAIN vs TEST LENGTH DISTRIBUTION")
    print("=" * 72)
    df_test["prompt_tok"] = df_test["prompt"].apply(lambda s: len(tok.encode(s)))
    df_test["response_tok"] = df_test["response"].apply(lambda s: len(tok.encode(s)))
    df_test["combined_tok"] = df_test["prompt_tok"] + df_test["response_tok"]
    df_test["truncated"] = df_test["combined_tok"] > MAX_LEN

    for col in ["prompt_tok", "response_tok", "combined_tok"]:
        print(f"\n  {col}:")
        for name, sub in [("train_truthful", df[df["label"]==0.0]),
                          ("train_hallu", df[df["label"]==1.0]),
                          ("train_all", df),
                          ("test_all", df_test)]:
            v = sub[col]
            print(f"    {name:18s}  mean={v.mean():6.1f}  "
                  f"median={v.median():6.0f}  p25={v.quantile(0.25):5.0f}  "
                  f"p75={v.quantile(0.75):5.0f}")

    print(f"\n  Truncation rates:")
    print(f"    train_truthful: {(df[df['label']==0.0]['truncated']).mean()*100:.1f}%")
    print(f"    train_hallu:    {(df[df['label']==1.0]['truncated']).mean()*100:.1f}%")
    print(f"    test_all:       {df_test['truncated'].mean()*100:.1f}%")

    # ---- 4. Test response length: estimate prior on test ----
    print("\n" + "=" * 72)
    print(" TEST: predict the test set using train-fitted length probe")
    print("=" * 72)
    X_train = df[["response_tok", "log_response", "combined_tok", "truncated"]].to_numpy(dtype=float)
    df_test["log_response"] = np.log1p(df_test["response_tok"])
    X_test = df_test[["response_tok", "log_response", "combined_tok", "truncated"]].to_numpy(dtype=float)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train, y)
    pred_test = clf.predict(X_test)
    print(f"  Test predicted hallu rate (length-only): {pred_test.mean()*100:.1f}%")
    print(f"  Train  hallu rate (true):                 {y.mean()*100:.1f}%")

    # ---- 5. Stratified CV by length: does within-length-bin probing add ANY signal? ----
    print("\n" + "=" * 72)
    print(" WITHIN-LENGTH-BIN HALLU RATE")
    print("=" * 72)
    bins = pd.qcut(df["response_tok"], q=5, duplicates="drop")
    by_bin = df.groupby(bins)["label"].agg(["count", "mean"])
    print(by_bin)

    # ---- 6. Hidden-state probe vs length probe ----
    print("\n" + "=" * 72)
    print(" CONCLUSION:")
    print("=" * 72)
    print(" If length-only AUROC is close to our 75.7% — the probe basically learns length.")
    print(" If it's much lower — there's real factuality signal.")


if __name__ == "__main__":
    main()
