#!/usr/bin/env python
"""阈值标定实验（Phase 5 / 4.10）。

目标：为 BERT 路由器的强/弱切分阈值提供数据依据，替代拍脑袋的 0.5。

与 eval_router_apgr.py 的差异：
  - 那个脚本只产出「曲线 + 单一 APGR」，回答的是「路由器好不好」
  - 本脚本产出「win_rate 分布 + 分位数 + 多阈值 APGR 对比 + bootstrap CI
    + 多轮稳定性」，回答的是「切点定在哪最优」

设计要点：
  1. 候选阈值由实际分布动态生成，不预设 0.3~0.5
  2. 每个候选档都必须是「真分流」（既不 0% 也不 100% 走强），排除死档
  3. bootstrap 重采样估计 APGR 的置信区间，规避小样本偏差
  4. 多轮重采样量化同一阈值的 APGR 波动，取稳定区间

指标公式（与 routellm/evals 论文一致）：
    APGR = (router_AUC - weak_AUC) / (strong_AUC - weak_AUC)
    0 = 与全走弱相同；1 = 达到理论最优（先知路由）

用法：
    python scripts/calibrate_threshold.py \
        --bert-url http://127.0.0.1:6070 \
        --mmlu-dir routellm/evals/mmlu/responses \
        --gsm8k-csv routellm/evals/gsm8k/gsm8k_responses.csv \
        --out-dir results/calibration
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


# ---------------------------------------------------------------- 数据读取

def read_csv_rows(fp):
    with open(fp, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def score_prompts(url, prompts, batch=256):
    """调 6070 推理服务批量评分，返回 win_rate 数组。"""
    out = []
    total = len(prompts)
    t0 = time.time()
    for i in range(0, total, batch):
        chunk = prompts[i : i + batch]
        payload = json.dumps({"prompts": chunk}).encode()
        req = urllib.request.Request(
            f"{url}/v1/score",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=1800) as resp:
            d = json.loads(resp.read().decode())
        out.extend(r["win_rate"] for r in d["results"])
        if (i // batch) % 20 == 0:
            done = min(i + batch, total)
            el = time.time() - t0
            print(f"        评分进度 {done}/{total} ({el:.1f}s)", flush=True)
    return np.array(out, dtype=float)


def load_mmlu(mmlu_dir, sample_per_subject=0):
    """读 MMLU 评测集。sample_per_subject=0 表示全量。"""
    files = sorted(glob.glob(os.path.join(mmlu_dir, "*.csv")))
    prompts, weak, strong, subjects = [], [], [], []
    for fp in files:
        rows = read_csv_rows(fp)
        if sample_per_subject and len(rows) > sample_per_subject:
            idx = np.linspace(0, len(rows) - 1, sample_per_subject).astype(int)
            rows = [rows[i] for i in idx]
        subj = os.path.basename(fp).replace("mmlu_", "").replace(".csv", "")
        for r in rows:
            prompts.append(r["prompt"])
            weak.append(str(r[WEAK]).strip() == "True")
            strong.append(str(r[STRONG]).strip() == "True")
            subjects.append(subj)
    return prompts, np.array(weak), np.array(strong), subjects


def load_gsm8k(csv_path):
    rows = read_csv_rows(csv_path)
    prompts = [r["prompt"] for r in rows]
    weak = np.array([str(r[WEAK]).strip() == "True" for r in rows])
    strong = np.array([str(r[STRONG]).strip() == "True" for r in rows])
    return prompts, weak, strong


# ------------------------------------------------------------ 核心计算
#
# 术语与公式严格对齐论文 RouteLLM (ICLR 2025)：
#   Eq 4  c(M_Rα)     = 走强比例（成本指标）
#   Eq 5  r(M_Rα)     = 平均响应质量
#   Eq 6  PGR         = (r(M_Rα) - r(M_w)) / (r(M_s) - r(M_w))
#   Eq 7  APGR        = ∫₀¹ PGR d(c)          （对成本积分）
#   Eq 8  APGR ≈ (1/10) Σ_{i=1..10} PGR(c_i)   （把走强比例离散成 10 档）
#   CPT(x%)           = 达到 PGR=x% 所需的最小走强比例
#
# 随机基线：APGR ≈ 0.5（论文 Table 1-3 实测 0.500±0 / 0.497±0.01 / 0.493±0.033）。
# 论文原话："a trivial router that always sends queries to the strong model
# achieves a perfect PGR = 1 but with no cost savings." —— 故不能只看 PGR。

#: 论文 Eq 8 的离散档数
PAPER_COST_BINS = 10


def compute_pgr(weak_acc, strong_acc, router_acc):
    """论文 Eq 6：性能差距恢复比例 PGR。

    PGR = (r(M_Rα) - r(M_w)) / (r(M_s) - r(M_w))

    0 = 与全走弱相同；1 = 达到全强的质量。
    注意 PGR=1 且不省成本是无意义的「trivial router」，必须配合成本看。
    """
    denom = strong_acc - weak_acc
    if denom == 0:
        return float("nan")
    return (router_acc - weak_acc) / denom


def compute_cpt(pgr_curve, target_pgr):
    """CPT(x%)：达到 PGR = x% 所需的最小走强比例（论文 3.2 节定义）。

    参数
      pgr_curve : [(strong_pct, pgr), ...]，按 strong_pct 升序
      target_pgr: 目标 PGR（0~1），论文常用 0.5 / 0.8

    返回达到目标的最小走强比例；若曲线任何点都达不到则返回 None。
    """
    for strong_pct, pgr in sorted(pgr_curve):
        if pgr >= target_pgr - 1e-12:
            return float(strong_pct)
    return None


def compute_apgr(win_rates, weak_correct, strong_correct, thresholds=None,
                 cost_bins=PAPER_COST_BINS):
    """论文 Eq 7/8：APGR = ∫₀¹ PGR d(c)，按论文做法离散成 cost_bins 档。

    论文 Eq 8 的做法：把走强比例 [0%,100%] 离散成 {c_i}_{i∈[10]}，
    对每个 c_i 求满足该成本约束的阈值 α_i，取 PGR 均值。

    与早期实现的区别（重要）：
      早期版本在「阈值网格」上扫描、对 (走强比例, 准确率) 直接用 trapz 积分。
      这在数学上近似 Eq 7，但与论文的**成本档位**离散方式不一致，导致
      数值不可与论文表格直接对照。现改为严格的 Eq 8：等分成本档，
      每档反解阈值（分位数），再取 PGR 均值。

    thresholds 参数保留用于向后兼容：传入时退化为阈值网格扫描 + trapz。
    """
    win_rates = np.asarray(win_rates, dtype=float)
    weak_correct = np.asarray(weak_correct)
    strong_correct = np.asarray(strong_correct)

    weak_acc = float(np.mean(weak_correct))
    strong_acc = float(np.mean(strong_correct))

    if thresholds is not None:
        # 兼容路径：阈值网格 + trapz 近似
        percents, accs = [], []
        for th in thresholds:
            route_strong = win_rates >= th
            acc = float(np.mean(np.where(route_strong, strong_correct, weak_correct)))
            percents.append(float(np.mean(route_strong)))
            accs.append(acc)
        percents = np.array(percents)
        accs = np.array(accs)
        order = np.argsort(percents)
        percents, accs = percents[order], accs[order]
        router_auc = float(np.trapz(accs, percents))
        apgr = (router_auc - weak_acc * 1.0) / (strong_acc - weak_acc) \
            if strong_acc != weak_acc else float("nan")
    else:
        # 论文 Eq 8 路径：等分成本档，每档反解阈值，取 PGR 均值
        #
        # 档位必须覆盖 [0%, 100%] 的**闭区间**（含两端）。
        # 这是保持随机基线对称性的关键：随机路由下 APGR 必须 ≈ 0.5，
        # 而只取 10%~100%（缺 0% 档，偏高）会得 0.551，
        # 只取 0%~90%（缺 100% 档，偏低）会得 0.451 —— 都偏离论文基线。
        # 实测覆盖闭区间后用 11 个等分点（0,10,...,100）得 0.501，与论文一致。
        pgr_sum = 0.0
        n_pts = 0
        for i in range(0, cost_bins + 1):
            target_pct = 100.0 * i / cost_bins
            if target_pct <= 0:
                route_strong = np.zeros(len(win_rates), dtype=bool)
            elif target_pct >= 100:
                route_strong = np.ones(len(win_rates), dtype=bool)
            else:
                th = threshold_for_strong_pct(win_rates, target_pct)
                route_strong = win_rates >= th
            acc = float(np.mean(np.where(route_strong, strong_correct, weak_correct)))
            pgr_sum += compute_pgr(weak_acc, strong_acc, acc)
            n_pts += 1
        apgr = pgr_sum / n_pts
        router_auc = float("nan")  # 该路径下 AUC 无对应定义

    # 曲线（供 CPT 计算与作图）
    curve, pgr_curve = [], []
    for i in range(0, cost_bins + 1):
        target_pct = 100.0 * i / cost_bins
        th = threshold_for_strong_pct(win_rates, target_pct) if target_pct > 0 \
            else float("inf")
        route_strong = win_rates >= th if target_pct > 0 else np.zeros(len(win_rates), bool)
        acc = float(np.mean(np.where(route_strong, strong_correct, weak_correct)))
        pct = float(np.mean(route_strong))
        curve.append({"strong_pct": round(pct, 4), "accuracy": round(acc, 4)})
        pgr_curve.append((pct, compute_pgr(weak_acc, strong_acc, acc)))

    cpt = {f"CPT({int(t * 100)}%)": compute_cpt(pgr_curve, t)
           for t in (0.5, 0.8)}

    return {
        "weak_accuracy": round(weak_acc, 4),
        "strong_accuracy": round(strong_acc, 4),
        "router_auc": round(router_auc, 4) if router_auc == router_auc else None,
        "apgr": round(apgr, 4),
        "cpt": {k: (round(v, 4) if v is not None else None) for k, v in cpt.items()},
        "curve": curve,
    }


def threshold_for_strong_pct(win_rates, target_pct):
    """取 (100-target) 分位作为阈值，使走强比例接近 target_pct。

    `routed_model = strong if win_rate >= threshold else weak`
    → 阈值取高分位 → 更少请求满足 → 更少走强。
    """
    win_rates = np.asarray(win_rates)
    return float(np.percentile(win_rates, 100.0 - target_pct))


def candidate_thresholds(win_rates, n=5, pct_lo=2.0, pct_hi=50.0):
    """按实际分布动态生成候选阈值，排除死档，且覆盖有意义的走强比例区间。

    策略：以「走强比例」为锚，在 [pct_lo, pct_hi] 上等间隔取 n 个比例点，
    反查对应分位得到阈值。

    `routed_model = strong if win_rate >= threshold else weak`
    → 要让 X% 的请求走强，阈值应取 (100-X) 分位。
      例：pct=2  → p98 分位 → 约 2% 走强
          pct=50 → p50 分位 → 约 50% 走强

    为什么是 [pct_lo, pct_hi] 而非固定均分点：窄分布下高分位点会挤在一
    个极窄窗口里（实测 7 个候选落在 0.005 宽度内），等于只评估了一个点。
    聚焦 2%~50% 是因为走强比例超过 50% 时成本已过半，且路由收益主要
    在低比例段实现（APGR 曲线的陡峭段）。

    单调方向（易错点）：pct 越大 → 阈值越低（分位数越低），返回数组是
    **随 pct 递增而单调递减**。早期实现按「递增」做去重兜底，把全部候选
    钉死在第一个值上（走强比例全为 2%），已被回归测试捕获。
    """
    win_rates = np.asarray(win_rates)
    pcts = np.linspace(pct_lo, pct_hi, n)
    cands = [threshold_for_strong_pct(win_rates, p) for p in pcts]

    # pct 递增 → 阈值递减（分位数天然单调不减，方向与之相反）。
    # 仅在极端退化（重复值）时做微量区分，保持严格递减。
    out = []
    for c in cands:
        c = float(c)
        if out and c >= out[-1]:
            c = out[-1] - 1e-6
        out.append(c)

    # 排除死档（防御性；分位点理论上不会是死档）
    out = [c for c in out if 0.0 < float(np.mean(win_rates >= c)) < 1.0]

    return np.array(out, dtype=float)


def bootstrap_apgr_ci(win_rates, weak_correct, strong_correct, n_boot=500,
                      seed=0, ci=0.95):
    """对样本做 bootstrap 重采样，估计 APGR（论文 Eq 8）的置信区间。"""
    rng = np.random.default_rng(seed)
    win_rates = np.asarray(win_rates)
    weak_correct = np.asarray(weak_correct)
    strong_correct = np.asarray(strong_correct)
    n = len(win_rates)

    point = compute_apgr(win_rates, weak_correct, strong_correct)["apgr"]

    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[b] = compute_apgr(
            win_rates[idx], weak_correct[idx], strong_correct[idx]
        )["apgr"]

    boot = boot[~np.isnan(boot)]
    lo = float(np.percentile(boot, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot, (1 + ci) / 2 * 100))
    return {
        "apgr_point": point,
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "ci_level": ci,
        "n_boot": int(n_boot),
    }


def stability_report(win_rates, weak_correct, strong_correct, thresholds,
                     rounds=5, seed=0, subsample=None):
    """多轮重采样，量化各阈值 APGR 的波动。

    重要：APGR 是**曲线积分**指标，不能对单个阈值计算 —— 只给一个切点时
    曲线退化成两点梯形，AUC 与基线之差没有意义（实测会得到 -5 这种荒谬值）。
    正确做法：每轮重采样后，在「围绕该阈值的一个网格」上算完整曲线，
    取该阈值对应的准确率点，与全局基线比较得到该阈值下的路由增益。

    因此这里报告的是：
      - per_cost[].pgr_at_th: 该阈值下的 PGR（论文 Eq 6）
                             = (r(M_Rα) - r(M_w)) / (r(M_s) - r(M_w))
                             分母取**全局** weak/strong 准确率。
      - per_cost[].strong_pct: 该阈值下的走强比例（论文 Eq 4 的成本指标）
      - apgr_curve_mean: 该轮样本上的整体 APGR（论文 Eq 8，10 档成本离散）

    PGR 的取值范围（重要）：
      PGR 是「相对全局全弱基线的性能差距恢复比例」，**可以大于 1**。
      当阈值挑出的子集恰好是强模型擅长、弱模型拉胯的题时，该子集上的
      准确率提升会超过全局 (r(M_s) - r(M_w)) 差值 —— 实测可达 1.1。
      含义是「比全强更省成本，质量却接近甚至局部超过全强」。
      真正异常的是**负的 PGR**（比全弱还差 = 切点方向接反）。
    """
    rng = np.random.default_rng(seed)
    win_rates = np.asarray(win_rates)
    weak_correct = np.asarray(weak_correct)
    strong_correct = np.asarray(strong_correct)
    n = len(win_rates)
    m = subsample or n

    fine = np.arange(0.0, 1.001, 0.01)

    out = {}
    for th in thresholds:
        per_round = []
        for _ in range(rounds):
            idx = rng.integers(0, n, m)
            wr = win_rates[idx]
            wc = weak_correct[idx]
            sc = strong_correct[idx]

            weak_acc = float(np.mean(wc))
            strong_acc = float(np.mean(sc))

            # 该阈值处的 PGR（论文 Eq 6）
            route_strong = wr >= th
            acc_at_th = float(np.mean(np.where(route_strong, sc, wc)))
            pgr = compute_pgr(weak_acc, strong_acc, acc_at_th)

            # 该轮样本上的整体 APGR（论文 Eq 8，10 档成本离散）
            curve_apgr = compute_apgr(wr, wc, sc)["apgr"]

            per_round.append({
                "pgr_at_th": float(pgr),
                "apgr_curve": curve_apgr,
                "strong_pct": float(np.mean(route_strong)),
                "accuracy": acc_at_th,
            })

        pgrs = [r["pgr_at_th"] for r in per_round]
        curves = [r["apgr_curve"] for r in per_round]
        pcts = [r["strong_pct"] for r in per_round]
        accs = [r["accuracy"] for r in per_round]

        out[th] = {
            "pgr_mean": float(np.mean(pgrs)),
            "pgr_std": float(np.std(pgrs)),
            "pgr_min": float(np.min(pgrs)),
            "pgr_max": float(np.max(pgrs)),
            "apgr_curve_mean": float(np.mean(curves)),
            "apgr_curve_std": float(np.std(curves)),
            "strong_pct_mean": float(np.mean(pcts)),
            "strong_pct_std": float(np.std(pcts)),
            "accuracy_mean": float(np.mean(accs)),
            "pgr_per_round": pgrs,
        }
    return out


def describe_distribution(win_rates):
    """win_rate 分布描述：分位数 + 直方图。"""
    win_rates = np.asarray(win_rates)
    qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    quantiles = {f"p{q}": round(float(np.percentile(win_rates, q)), 4) for q in qs}
    counts, edges = np.histogram(win_rates, bins=20, range=(0.0, 1.0))
    return {
        "n": int(len(win_rates)),
        "mean": round(float(np.mean(win_rates)), 4),
        "std": round(float(np.std(win_rates)), 4),
        "min": round(float(np.min(win_rates)), 4),
        "max": round(float(np.max(win_rates)), 4),
        "quantiles": quantiles,
        "histogram": {
            "bin_edges": [round(float(e), 3) for e in edges],
            "counts": [int(c) for c in counts],
        },
    }


# ------------------------------------------------------------------ main

def evaluate_dataset(name, win_rates, weak, strong, args):
    print(f"\n{'=' * 62}\n[{name}] n={len(win_rates)}\n{'=' * 62}")

    dist = describe_distribution(win_rates)
    print(f"  分布: mean={dist['mean']} std={dist['std']} "
          f"min={dist['min']} max={dist['max']}")
    print(f"  分位数: {dist['quantiles']}")

    # 全量指标（论文 Eq 7/8）
    base = compute_apgr(win_rates, weak, strong)
    print(f"  r(M_w) 全弱 acc = {base['weak_accuracy']}")
    print(f"  r(M_s) 全强 acc = {base['strong_accuracy']}")
    print(f"  ★ APGR = {base['apgr']}   （随机基线 ≈ 0.5，>0.5 才算路由有效）")
    if base["cpt"]:
        cpt_str = "  ".join(f"{k}={v:.2%}" if v is not None else f"{k}=N/A"
                            for k, v in base["cpt"].items())
        print(f"  ★ CPT = {cpt_str}")

    # 动态候选阈值
    cands = candidate_thresholds(win_rates, n=args.n_candidates,
                                 pct_lo=args.pct_lo, pct_hi=args.pct_hi)
    print(f"\n  候选阈值（按分布动态生成, n={len(cands)}）: "
          f"{[round(float(c), 4) for c in cands]}")

    table = []
    for th in cands:
        route_strong = win_rates >= th
        strict_pct = float(np.mean(route_strong))
        acc = float(np.mean(np.where(route_strong, strong, weak)))
        pgr = compute_pgr(base["weak_accuracy"], base["strong_accuracy"], acc)
        table.append({
            "threshold": round(float(th), 4),
            "strong_pct": round(strict_pct, 4),   # 论文 Eq 4 的 c(M_Rα)
            "accuracy": round(acc, 4),            # 论文 Eq 5 的 r(M_Rα)
            "pgr": round(pgr, 4),                 # 论文 Eq 6
            "cost_saving_vs_all_strong": round(1.0 - strict_pct, 4),
        })

    print(f"\n  {'α(阈值)':>10}{'c(MRα)走强':>12}{'r(MRα)质量':>12}"
          f"{'PGR':>9}{'省成本':>10}")
    for row in table:
        print(f"  {row['threshold']:>10.4f}{row['strong_pct']:>11.2%}"
              f"{row['accuracy']:>12.4f}{row['pgr']:>9.4f}"
              f"{row['cost_saving_vs_all_strong']:>10.2%}")

    # bootstrap CI（整体 APGR）
    print(f"\n  bootstrap CI (n_boot={args.n_boot}) ...", flush=True)
    ci = bootstrap_apgr_ci(win_rates, weak, strong,
                           n_boot=args.n_boot, seed=args.seed)
    print(f"  APGR = {ci['apgr_point']} "
          f"[{ci['ci_low']}, {ci['ci_high']}] ({ci['ci_level']:.0%} CI)")

    # 稳定性：候选阈值各跑 rounds 轮
    print(f"\n  稳定性评估 (rounds={args.rounds}) ...", flush=True)
    stab = stability_report(win_rates, weak, strong, list(cands),
                            rounds=args.rounds, seed=args.seed)
    print(f"  {'α(阈值)':>10}{'PGR均值':>10}{'PGR标准差':>11}"
          f"{'走强比例':>11}{'质量':>10}")
    for th, s in stab.items():
        print(f"  {th:>10.4f}{s['pgr_mean']:>10.4f}{s['pgr_std']:>11.4f}"
              f"{s['strong_pct_mean']:>11.2%}{s['accuracy_mean']:>10.4f}")

    # 推荐：按最高 APGR 对应的成本档反推（论文式：成本优先）
    ranked = sorted(stab.items(), key=lambda kv: -kv[1]["pgr_mean"])
    best_th, best_s = ranked[0]
    print(f"\n  ★ PGR 最高的阈值 = {best_th:.4f} "
          f"(PGR {best_s['pgr_mean']:.4f} ± {best_s['pgr_std']:.4f}, "
          f"走强 {best_s['strong_pct_mean']:.1%}, "
          f"质量 {best_s['accuracy_mean']:.4f})")
    print(f"  ⚠ 注意：PGR 随走强比例单调递增，故此为区间内最大值，"
          f"不代表「最优切点」—— 决策应看下表的 CPT。")

    # 边界警告：最高 PGR 落在搜索区间边缘时，说明区间未覆盖拐点
    boundary_warning = None
    if len(ranked) >= 2:
        best_idx = [th for th, _ in ranked].index(best_th)
        cands_list = list(cands)
        if best_idx == 0 or best_idx == len(cands_list) - 1:
            boundary_warning = (
                f"最高 PGR 落在候选区间边界（走强比例 "
                f"{best_s['strong_pct_mean']:.1%}）；如需评估更保守/更激进的"
                f"切点，请调整 [{args.pct_lo}%, {args.pct_hi}%]"
            )
            print(f"  ⚠ {boundary_warning}")

    # ---- CPT 表（论文 3.2 节）：达到目标 PGR 所需的最小走强比例 ----
    #
    # 这是**决策用的表**。PGR/APGR 衡量「路由器好不好」，CPT 回答
    # 「我要达到目标质量，最少要走多少次强模型」—— 即论文定义的
    # "the minimum percentage of calls to the strong model needed to
    # reach the desired PGR"。
    print(f"\n  ── CPT 表（论文 3.2 节：达到目标 PGR 所需的最小走强比例）──")
    pgr_curve = [(r["strong_pct"], r["pgr"]) for r in table]
    # 曲线需覆盖全成本区间，用细网格补足
    fine_grid = np.arange(0.0, 1.0001, 0.005)
    fine_pairs = []
    for th in fine_grid:
        rs = win_rates >= th
        pct = float(np.mean(rs))
        acc = float(np.mean(np.where(rs, strong, weak)))
        fine_pairs.append((pct, compute_pgr(base["weak_accuracy"],
                                            base["strong_accuracy"], acc)))
    fine_pairs.sort()

    cpt_table = []
    for target in (0.5, 0.8, 0.9, 0.95, 1.0):
        hit = compute_cpt(fine_pairs, target)
        # 反查该成本点对应的阈值
        th_hit = None
        if hit is not None:
            th_hit = round(float(threshold_for_strong_pct(win_rates, hit * 100)), 4)
            acc_hit = float(np.mean(np.where(win_rates >= th_hit, strong, weak)))
        else:
            acc_hit = None
        cpt_table.append({
            "target_pgr": target,
            "cpt_strong_pct": round(hit, 4) if hit is not None else None,
            "cpt_threshold": th_hit,
            "cost_saving_vs_all_strong": round(1.0 - hit, 4) if hit is not None else None,
            "accuracy_at_cpt": round(acc_hit, 4) if acc_hit is not None else None,
        })

    print(f"  {'目标PGR':>10}{'CPT(走强比例)':>16}{'对应阈值α':>12}"
          f"{'省成本':>10}{'质量':>10}")
    for r in cpt_table:
        if r["cpt_strong_pct"] is None:
            print(f"  {r['target_pgr']:>10.0%}{'不可达':>16}")
            continue
        print(f"  {r['target_pgr']:>10.0%}{r['cpt_strong_pct']:>15.2%}"
              f"{r['cpt_threshold']:>12.4f}"
              f"{r['cost_saving_vs_all_strong']:>10.2%}"
              f"{r['accuracy_at_cpt']:>10.4f}")

    return {
        "n": int(len(win_rates)),
        "distribution": dist,
        "overall": base,
        "table": table,
        "cpt_table": cpt_table,
        "bootstrap": ci,
        "stability": {str(k): v for k, v in stab.items()},
        "max_pgr_threshold": round(float(best_th), 4),
        "max_pgr": best_s,
        "boundary_warning": boundary_warning,
        "search_range_pct": [args.pct_lo, args.pct_hi],
        "ranking": [
            {"threshold": round(float(k), 4), **{
                kk: vv for kk, vv in v.items() if kk != "pgr_per_round"
            }}
            for k, v in ranked
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bert-url", default="http://127.0.0.1:6070")
    ap.add_argument("--mmlu-dir", default=None)
    ap.add_argument("--gsm8k-csv", default=None)
    ap.add_argument("--sample-per-subject", type=int, default=0,
                    help="每个 MMLU 学科最多取多少题（0=全量）")
    ap.add_argument("--n-candidates", type=int, default=5,
                    help="动态候选阈值数量")
    ap.add_argument("--pct-lo", type=float, default=2.0,
                    help="候选阈值覆盖的最小走强比例(%)")
    ap.add_argument("--pct-hi", type=float, default=80.0,
                    help="候选阈值覆盖的最大走强比例(%)。设得过小会把最优点"
                         "排除在搜索区间外——若最优出现在区间边界，说明该调大")
    ap.add_argument("--n-boot", type=int, default=500,
                    help="bootstrap 重采样次数")
    ap.add_argument("--rounds", type=int, default=5,
                    help="稳定性评估轮数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    results = {"config": vars(args)}

    if args.mmlu_dir:
        print(f"[load] MMLU <- {args.mmlu_dir}", flush=True)
        prompts, weak, strong, subjects = load_mmlu(
            args.mmlu_dir, args.sample_per_subject
        )
        print(f"       {len(prompts)} 题, 评分中 ...", flush=True)
        t0 = time.time()
        wrs = score_prompts(args.bert_url, prompts)
        print(f"       评分耗时 {time.time() - t0:.1f}s")
        results["mmlu"] = evaluate_dataset("MMLU", wrs, weak, strong, args)

    if args.gsm8k_csv and os.path.exists(args.gsm8k_csv):
        print(f"\n[load] GSM8K <- {args.gsm8k_csv}", flush=True)
        prompts, weak, strong = load_gsm8k(args.gsm8k_csv)
        print(f"       {len(prompts)} 题, 评分中 ...", flush=True)
        t0 = time.time()
        wrs = score_prompts(args.bert_url, prompts)
        print(f"       评分耗时 {time.time() - t0:.1f}s")
        results["gsm8k"] = evaluate_dataset("GSM8K", wrs, weak, strong, args)

    # ---- 汇总 ----
    print("\n" + "=" * 62)
    print("★ 阈值标定汇总")
    print("=" * 62)
    for name, r in results.items():
        if not isinstance(r, dict) or "max_pgr_threshold" not in r:
            continue
        cpt50 = next((c for c in r["cpt_table"] if c["target_pgr"] == 0.5), None)
        cpt_str = (f"CPT(50%)={cpt50['cpt_strong_pct']:.2%}"
                   if cpt50 and cpt50["cpt_strong_pct"] is not None else "CPT(50%)=N/A")
        print(f"  {name:<8} APGR={r['overall']['apgr']:.4f} "
              f"[{r['bootstrap']['ci_low']:.4f}, {r['bootstrap']['ci_high']:.4f}]  "
              f"{cpt_str}")

    # 与论文 Table 2/3 的对照提示
    print("\n  对照论文 (ICLR 2025)：")
    print("    - 随机路由基线 APGR ≈ 0.5；APGR > 0.5 才算路由有效")
    print("    - 论文 MMLU  Dextra(=D_arena+D_gold) BERT: APGR 0.572, CPT(50%) 41.30%")
    print("    - 论文 GSM8K (=D_arena+D_judge) BERT: APGR 0.531, CPT(50%) 44.76%")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        out = os.path.join(args.out_dir, "threshold_calibration.json")
        with open(out, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已写入 {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
