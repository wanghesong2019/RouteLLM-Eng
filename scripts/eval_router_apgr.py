#!/usr/bin/env python
"""按论文指标（APGR）评测 bert 路由在 MMLU / GSM8K 上的表现。

论文指标说明（routellm/evals/evaluate.py）：
    APGR = (router_AUC - weak_AUC) / (strong_AUC - weak_AUC)
    0 = 与全走弱模型相同；1 = 达到理论最优（先知路由）

流程：
    1. 从 MMLU/GSM8K 数据读 prompt + 强弱模型的对错标记（预计算，无需真实 API）
    2. 通过 6070 服务批量计算每个 prompt 的 win_rate
    3. 扫多个 threshold，计算准确率曲线与 AUC
    4. 输出 APGR + 与 baseline（random）对比

用法：
    python scripts/eval_router_apgr.py \
        --bert-url http://127.0.0.1:6070 \
        --mmlu-dir routellm/evals/mmlu/responses \
        --gsm8k-csv routellm/evals/gsm8k/gsm8k_responses.csv \
        --out-dir results/apgr
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WEAK = "mistralai/Mixtral-8x7B-Instruct-v0.1"
STRONG = "gpt-4-1106-preview"


def read_csv_rows(fp):
    with open(fp, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def score_prompts(url, prompts, batch=64):
    """调推理服务批量评分。"""
    out = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i : i + batch]
        payload = json.dumps({"prompts": chunk}).encode()
        req = urllib.request.Request(
            f"{url}/v1/score", data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            d = json.loads(resp.read().decode())
        out.extend(r["win_rate"] for r in d["results"])
    return np.array(out)


def compute_apgr(win_rates, weak_correct, strong_correct, thresholds):
    """按论文公式计算 APGR 与准确率曲线。"""
    # 全弱 / 全强的准确率
    weak_acc = float(np.mean(weak_correct))
    strong_acc = float(np.mean(strong_correct))

    percents, accs = [], []
    for th in thresholds:
        route_strong = win_rates >= th
        # 路由到强模型时用强模型的正确性，否则用弱的
        acc = float(
            np.mean(np.where(route_strong, strong_correct, weak_correct))
        )
        pct = float(np.mean(route_strong))
        percents.append(pct)
        accs.append(acc)

    percents = np.array(percents)
    accs = np.array(accs)
    order = np.argsort(percents)
    percents, accs = percents[order], accs[order]

    router_auc = float(np.trapz(accs, percents))
    weak_auc = weak_acc * 1.0  # 常数积分
    strong_auc = strong_acc * 1.0
    apgr = (router_auc - weak_auc) / (strong_auc - weak_auc) if strong_auc != weak_auc else float("nan")

    return {
        "weak_accuracy": round(weak_acc, 4),
        "strong_accuracy": round(strong_acc, 4),
        "router_auc": round(router_auc, 4),
        "apgr": round(apgr, 4),
        "curve": [{"strong_pct": round(float(p), 4), "accuracy": round(float(a), 4)}
                  for p, a in zip(percents, accs)],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bert-url", default="http://127.0.0.1:6070")
    ap.add_argument("--mmlu-dir", default=None)
    ap.add_argument("--gsm8k-csv", default=None)
    ap.add_argument("--sample-per-subject", type=int, default=100,
                    help="每个 MMLU 学科最多取多少题（0=全部）")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    thresholds = np.concatenate([
        np.arange(0.0, 1.001, 0.01),
    ])
    results = {}

    # ---- MMLU ----
    if args.mmlu_dir:
        files = sorted(glob.glob(os.path.join(args.mmlu_dir, "*.csv")))
        print(f"[MMLU] {len(files)} 个学科文件")
        all_prompts, all_weak, all_strong, subjects = [], [], [], []
        for fp in files:
            rows = read_csv_rows(fp)
            if args.sample_per_subject and len(rows) > args.sample_per_subject:
                idx = np.linspace(0, len(rows) - 1, args.sample_per_subject).astype(int)
                rows = [rows[i] for i in idx]
            subj = os.path.basename(fp).replace("mmlu_", "").replace(".csv", "")
            for r in rows:
                all_prompts.append(r["prompt"])
                all_weak.append(str(r[WEAK]).strip() == "True")
                all_strong.append(str(r[STRONG]).strip() == "True")
                subjects.append(subj)
        print(f"       共 {len(all_prompts)} 题")

        t0 = time.time()
        wrs = score_prompts(args.bert_url, all_prompts)
        print(f"       评分耗时 {time.time()-t0:.1f}s")

        res = compute_apgr(wrs, np.array(all_weak), np.array(all_strong), thresholds)
        results["mmlu"] = res
        print(f"\n[MMLU 结果]")
        print(f"  全弱模型准确率 : {res['weak_accuracy']}")
        print(f"  全强模型准确率 : {res['strong_accuracy']}")
        print(f"  路由器 AUC     : {res['router_auc']}")
        print(f"  ★ APGR         : {res['apgr']}")

    # ---- GSM8K ----
    if args.gsm8k_csv and os.path.exists(args.gsm8k_csv):
        print(f"\n[GSM8K] {args.gsm8k_csv}")
        rows = read_csv_rows(args.gsm8k_csv)
        prompts = [r["prompt"] for r in rows]
        weak = np.array([str(r[WEAK]).strip() == "True" for r in rows])
        strong = np.array([str(r[STRONG]).strip() == "True" for r in rows])
        print(f"       共 {len(prompts)} 题")

        t0 = time.time()
        wrs = score_prompts(args.bert_url, prompts)
        print(f"       评分耗时 {time.time()-t0:.1f}s")

        res = compute_apgr(wrs, weak, strong, thresholds)
        results["gsm8k"] = res
        print(f"\n[GSM8K 结果]")
        print(f"  全弱模型准确率 : {res['weak_accuracy']}")
        print(f"  全强模型准确率 : {res['strong_accuracy']}")
        print(f"  路由器 AUC     : {res['router_auc']}")
        print(f"  ★ APGR         : {res['apgr']}")

    # ---- 输出 ----
    print("\n" + "=" * 60)
    print("★ 论文指标 APGR 汇总（0=全弱模型, 1=理论最优）")
    print("=" * 60)
    print(f"  {'数据集':<10}{'APGR':>10}{'全弱':>10}{'全强':>10}")
    for name, r in results.items():
        print(f"  {name:<10}{r['apgr']:>10.4f}{r['weak_accuracy']:>10.4f}{r['strong_accuracy']:>10.4f}")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        out = os.path.join(args.out_dir, "apgr_bert.json")
        with open(out, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已写入 {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
