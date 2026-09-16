#!/usr/bin/env python3
"""端到端验证：RouteLLM（33 号机） → 推理服务（43 号机） → 路由决策。

验证整条链路真实可用，而非仅 mock 通过。

链路：
    Controller(remote_bert) --HTTP--> host:43:6070 /v1/score --> xlm-roberta

对照：
    - 33 上直接加载 BERT 的基准值（来自实验记录）
    - 43 推理服务的 /selfcheck
"""
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SERVICE = os.environ.get("ROUTELLM_INFERENCE_URL", "http://115.233.223.42:6070")

# 环境记录中的基准值（43 号机推理服务 /selfcheck）
BASELINE = {
    "What is 1+1?": 0.2969909906,
    "hi": 0.4007,
    "Write a poem about the sea.": 0.1993,
}

print("=" * 70)
print("RouteLLM → 推理服务 端到端验证")
print("=" * 70)
print(f"推理服务: {SERVICE}")

from routellm.controller import Controller
from routellm.routers.remote import RemoteBERTRouter, RemoteInferenceError

# ---------------------------------------------------------------- 1. 连通性
print()
print("1. 推理服务连通性")
router = RemoteBERTRouter(base_url=SERVICE, timeout=30.0)
try:
    h = router.health()
    print(f"   status={h.get('status')}  model_loaded={h.get('model_loaded')}")
    print(f"   arch={h.get('arch')}  device={h.get('device')}  labels={h.get('num_labels')}")
    print(f"   load_seconds={h.get('load_seconds')}")
except RemoteInferenceError as e:
    print(f"   FAILED: {e}")
    sys.exit(1)

# ---------------------------------------------------------------- 2. 一致性
print()
print("2. win_rate 一致性（服务 vs 直接加载模型的基准）")
print(f"   {'prompt':<34} {'service':>10} {'baseline':>10}  {'匹配':>5}")
all_ok = True
for prompt, baseline in BASELINE.items():
    wr = router.calculate_strong_win_rate(prompt)
    ok = abs(wr - baseline) < 1e-3
    all_ok &= ok
    print(f"   {prompt[:33]:<34} {wr:>10.4f} {baseline:>10.4f}  {'✓' if ok else '✗'}")
print(f"   → {'全部一致' if all_ok else '存在不一致！'}")

# ---------------------------------------------------------------- 3. 路由决策
print()
print("3. 路由决策（Controller + remote_bert，threshold=0.5）")
with_controller = Controller(
    routers=["remote_bert"],
    strong_model="openai/qwen3.7-max",
    weak_model="openai/qwen3.5-flash",
    config={"remote_bert": {"base_url": SERVICE}},
)

CASES = [
    ("What is 1+1?", 0.5, "weak-model"),
    ("Write a poem about the sea.", 0.5, "weak-model"),
    ("Prove that the square root of 2 is irrational using Galois theory.", 0.5, "?"),
]
print(f"   {'prompt':<52} {'win_rate':>9} {'阈值':>5} {'路由到':<12}")
for prompt, thr, _exp in CASES:
    wr = with_controller.routers["remote_bert"].calculate_strong_win_rate(prompt)
    routed = with_controller.route(prompt, "remote_bert", thr)
    short = "strong" if routed == "openai/qwen3.7-max" else "weak"
    print(f"   {prompt[:51]:<52} {wr:>9.4f} {thr:>5} {short:<12}")

# ---------------------------------------------------------------- 4. 批量
print()
print("4. 批量评分（100 条，验证 batch 接口）")
import time

prompts = [f"Batch end-to-end probe number {i} for validation." for i in range(100)]
t = time.time()
wrs = router.calculate_strong_win_rate_batch(prompts)
elapsed = (time.time() - t) * 1000
print(f"   {len(wrs)} 条，耗时 {elapsed:.1f}ms（{elapsed/len(wrs):.2f}ms/条）")
print(f"   win_rate 范围: {min(wrs):.4f} ~ {max(wrs):.4f}")

# ---------------------------------------------------------------- 5. 错误处理
print()
print("5. 错误处理（指向不存在的服务）")
bad = RemoteBERTRouter(base_url="http://127.0.0.1:59999", timeout=3.0)
try:
    bad.calculate_strong_win_rate("test")
    print("   ✗ 应当报错但未报错")
except RemoteInferenceError as e:
    print(f"   ✓ 正确抛出 RemoteInferenceError: {str(e)[:70]}")

print()
print("=" * 70)
print("E2E_DONE" if all_ok else "E2E_FAILED")
