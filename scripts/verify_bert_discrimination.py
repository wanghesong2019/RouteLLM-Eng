#!/usr/bin/env python3
"""BERT 路由器区分度验证 —— 在真实 benchmark 数据上检验 win rate 是否有效。

核心问题: win rate 是否真的与"弱模型能否答对"相关?

数据格式要求（来自 routellm/evals/）:
    GSM8K: 单 CSV, 含列 prompt / <weak_model> / <strong_model> (True|False)
    MMLU:  responses/ 目录下每学科一个 CSV, 同样列结构

用法:
    # GSM8K 同质数据（预期: 无区分度）
    python verify_bert_discrimination.py --model <ckpt> \
        --csv ../../routellm/evals/gsm8k/gsm8k_responses.csv

    # MMLU 跨学科（预期: 强负相关）
    python verify_bert_discrimination.py --model <ckpt> \
        --csv-dir ../../routellm/evals/mmlu/responses

实测记录（2026-09-16）:
    GSM8K   : 差值 +0.0004      → 无区分度
    MMLU(57): corr = -0.7123    → 区分度强
"""
import argparse
import csv
import glob
import os
import random

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

NUM_LABELS = 3
WEAK_COL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
STRONG_COL = "gpt-4-1106-preview"


def build_win_rate_fn(model, tokenizer, device):
    def win_rate(prompt):
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits.cpu().numpy()[0]
        exp = np.exp(logits - np.max(logits))
        sm = exp / np.sum(exp)
        return float(1 - np.sum(sm[-2:]))

    return win_rate


def read_csv(fp):
    with open(fp, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_gsm8k(win_rate, fp, sample_size):
    """按弱模型对错分组对比。"""
    rows = read_csv(fp)
    print(f"loaded {len(rows)} rows from {os.path.basename(fp)}")
    correct = [r for r in rows if str(r[WEAK_COL]).strip() == "True"]
    wrong = [r for r in rows if str(r[WEAK_COL]).strip() == "False"]
    random.seed(42)
    random.shuffle(correct)
    random.shuffle(wrong)

    res = {"weak_CORRECT": [], "weak_WRONG": []}
    for r in correct[:sample_size]:
        res["weak_CORRECT"].append(win_rate(r["prompt"]))
    for r in wrong[:sample_size]:
        res["weak_WRONG"].append(win_rate(r["prompt"]))

    print(f"weak correct 总数: {len(correct)}, weak wrong 总数: {len(wrong)}")
    print(f"抽样: {sample_size} + {sample_size}\n")
    print(f"{'group':<16} {'n':>4} {'mean':>8} {'std':>7} {'min':>7} {'max':>7}")
    for g, vals in res.items():
        a = np.array(vals)
        print(f"{g:<16} {len(a):>4} {a.mean():>8.4f} {a.std():>7.4f} {a.min():>7.4f} {a.max():>7.4f}")

    c, w = np.array(res["weak_CORRECT"]), np.array(res["weak_WRONG"])
    diff = c.mean() - w.mean()
    print(f"\nmean(weak_CORRECT) - mean(weak_WRONG) = {diff:+.4f}")
    for thr in (0.3, 0.4, 0.5):
        print(f"  threshold {thr}: weak_CORRECT 送强模型 {(c >= thr).sum()}/{len(c)}, "
              f"weak_WRONG 送强模型 {(w >= thr).sum()}/{len(w)}")
    return diff


def run_mmlu(win_rate, csv_dir, per_subject):
    """按学科算弱模型正确率 vs 平均 win rate。"""
    files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    print(f"发现 {len(files)} 个学科文件\n")

    per = []
    for fp in files:
        subject = os.path.basename(fp).replace("mmlu_", "").replace(".csv", "")
        rows = read_csv(fp)
        if not rows:
            continue
        step = max(1, len(rows) // per_subject)
        samp = rows[::step][:per_subject]
        wrs = [win_rate(r["prompt"]) for r in samp]
        weak_acc = float(np.mean([str(r[WEAK_COL]).strip() == "True" for r in samp]))
        strong_acc = float(np.mean([str(r[STRONG_COL]).strip() == "True" for r in samp]))
        per.append((subject, len(rows), weak_acc, strong_acc, float(np.mean(wrs))))

    per.sort(key=lambda x: x[2])
    print(f"{'subject':<40} {'n':>5} {'weak_acc':>9} {'strong_acc':>11} {'mean_wr':>9}")
    for s, tot, wa, sa, wr in per:
        print(f"{s:<40} {tot:>5} {wa:>9.3f} {sa:>11.3f} {wr:>9.4f}")

    wa_arr = np.array([x[2] for x in per])
    wr_arr = np.array([x[4] for x in per])
    corr = float(np.corrcoef(wa_arr, wr_arr)[0, 1])
    print(f"\n学科数: {len(per)}")
    print(f"weak_acc 范围: {wa_arr.min():.3f} ~ {wa_arr.max():.3f}  (std {wa_arr.std():.3f})")
    print(f"mean_wr  范围: {wr_arr.min():.4f} ~ {wr_arr.max():.4f}  (std {wr_arr.std():.4f})")
    print(f"相关系数 corr(weak_acc, mean_wr) = {corr:+.4f}")

    order = np.argsort(wa_arr)
    q = max(1, len(order) // 4)
    lowq, highq = wr_arr[order[:q]], wr_arr[order[-q:]]
    print(f"\n弱模型最不擅长的 1/4 学科, 平均 win rate = {lowq.mean():.4f}")
    print(f"弱模型最擅长的   1/4 学科, 平均 win rate = {highq.mean():.4f}")
    print(f"差值 = {highq.mean() - lowq.mean():+.4f}")
    return corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--labels", type=int, default=NUM_LABELS)
    ap.add_argument("--csv", help="GSM8K 单文件模式")
    ap.add_argument("--csv-dir", help="MMLU 目录模式")
    ap.add_argument("--sample-size", type=int, default=25, help="GSM8K 每组抽样数")
    ap.add_argument("--per-subject", type=int, default=30, help="MMLU 每学科抽样数")
    args = ap.parse_args()

    if not args.csv and not args.csv_dir:
        ap.error("至少提供 --csv 或 --csv-dir")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== loading ({device}) ===")
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=args.labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = model.to(device).eval()
    print("ok:", model.config.model_type, "\n")

    win_rate = build_win_rate_fn(model, tokenizer, device)

    if args.csv:
        diff = run_gsm8k(win_rate, args.csv, args.sample_size)
        print()
        if diff > 0.02:
            print("RESULT: 方向正确 —— 弱模型答对的题 win rate 更高")
        elif diff < -0.02:
            print("RESULT: 方向相反 —— 需检查标签语义")
        else:
            print("RESULT: 几乎无区分度 —— 同质化数据集上路由器可能无能为力")
    if args.csv_dir:
        corr = run_mmlu(win_rate, args.csv_dir, args.per_subject)
        print()
        if corr < -0.3:
            print("RESULT: 区分度强 —— 弱模型越不擅长的学科 win rate 越高（应路由到强模型）")
        elif corr > 0.3:
            print("RESULT: 方向相反 —— 需检查标签语义")
        else:
            print("RESULT: 区分度弱 —— 需排查")


if __name__ == "__main__":
    main()
