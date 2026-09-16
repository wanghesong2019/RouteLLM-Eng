#!/usr/bin/env python3
"""BERT 路由模型加载与基础行为验证。

复刻上游 routellm/routers/routers.py :: BERTRouter.calculate_strong_win_rate

用法:
    python verify_bert_basic.py --model <ckpt_path>

实测记录（2026-09-16, RTX 4090）:
    加载 1.2s / 推理 avg 6.1ms / 同输入结果确定性一致
"""
import argparse
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

NUM_LABELS = 3

PROMPTS = [
    "What is 1+1?",
    "hi",
    "Explain the trade-offs between consistency and availability in distributed systems, "
    "and how Raft handles network partitions.",
    "Write a poem about the sea.",
    "Prove that the square root of 2 is irrational and discuss its implications for "
    "constructible numbers in Galois theory.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="BERT router checkpoint path")
    ap.add_argument("--labels", type=int, default=NUM_LABELS)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== loading model from {args.model} (device={device}) ===")
    t = time.time()
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=args.labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = model.to(device).eval()
    print(f"loaded in {time.time() - t:.1f}s")
    print("config:", model.config.model_type, "| num_labels:", model.config.num_labels)

    def strong_win_rate(prompt):
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits.cpu().numpy()[0]
        exp_scores = np.exp(logits - np.max(logits))
        softmax_scores = exp_scores / np.sum(exp_scores)
        binary_prob = np.sum(softmax_scores[-2:])
        return 1 - binary_prob, softmax_scores

    print()
    print("=== win rate 计算 ===")
    print(f"{'win_rate':>10}  {'softmax':<34} prompt")
    for p in PROMPTS:
        wr, sm = strong_win_rate(p)
        sm_s = "[" + ", ".join(f"{x:.3f}" for x in sm) + "]"
        print(f"{wr:>10.4f}  {sm_s:<34} {p[:60]}")

    print()
    print("=== 确定性检查 ===")
    a, _ = strong_win_rate(PROMPTS[0])
    b, _ = strong_win_rate(PROMPTS[0])
    print(f"run1={a:.10f}  run2={b:.10f}  identical={a == b}")

    print()
    print("=== 推理耗时（预热后 10 次）===")
    times = []
    for _ in range(10):
        s = time.time()
        strong_win_rate("Latency probe prompt for benchmarking purposes.")
        times.append((time.time() - s) * 1000)
    print(f"min={min(times):.1f}ms  avg={sum(times)/len(times):.1f}ms  max={max(times):.1f}ms")


if __name__ == "__main__":
    main()
