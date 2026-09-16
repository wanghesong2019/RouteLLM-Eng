#!/usr/bin/env python3
"""端到端验证：RouteLLM（开发机） → 推理服务（执行机） → 路由决策。

验证整条链路真实可用，而非仅 mock 通过。

链路：
    Controller(remote_bert) --HTTP--> host:6070 /v1/score --> xlm-roberta

对照：
    - 直接加载 BERT 的基准值（来自实验记录）
    - 推理服务的 /selfcheck

用法：
    python scripts/test_e2e_remote_router.py
    ROUTELLM_INFERENCE_URL=http://<host>:6070 python scripts/test_e2e_remote_router.py

注意：本脚本需要真实推理服务在线，不是 pytest 用例。
      所有逻辑封装在 main() 内，import 时无副作用（避免被测试收集器导入时
      执行网络请求并 sys.exit，导致 pytest INTERNALERROR）。
"""
import os
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main() -> int:
    # 默认指向本机推理服务；跨机验证时通过环境变量覆盖，不硬编码地址。
    service = os.environ.get("ROUTELLM_INFERENCE_URL", "http://127.0.0.1:6070")

    # 环境记录中的基准值（推理服务 /selfcheck）
    baseline_map = {
        "What is 1+1?": 0.2969909906,
        "hi": 0.4007,
        "Write a poem about the sea.": 0.1993,
    }

    from routellm.controller import Controller
    from routellm.routers.remote import RemoteBERTRouter, RemoteInferenceError

    print("=" * 70)
    print("RouteLLM → 推理服务 端到端验证")
    print("=" * 70)
    print(f"推理服务: {service}")

    # ------------------------------------------------------------ 1. 连通性
    print()
    print("1. 推理服务连通性")
    router = RemoteBERTRouter(base_url=service, timeout=30.0)
    try:
        h = router.health()
        print(f"   status={h.get('status')}  model_loaded={h.get('model_loaded')}")
        print(f"   arch={h.get('arch')}  device={h.get('device')}  labels={h.get('num_labels')}")
        print(f"   load_seconds={h.get('load_seconds')}")
    except RemoteInferenceError as e:
        print(f"   FAILED: {e}")
        return 1

    # ------------------------------------------------------------ 2. 一致性
    print()
    print("2. win_rate 一致性（服务 vs 直接加载模型的基准）")
    print(f"   {'prompt':<34} {'service':>10} {'baseline':>10}  {'匹配':>5}")
    all_ok = True
    for prompt, baseline in baseline_map.items():
        wr = router.calculate_strong_win_rate(prompt)
        ok = abs(wr - baseline) < 1e-3
        all_ok &= ok
        print(f"   {prompt[:33]:<34} {wr:>10.4f} {baseline:>10.4f}  {'✓' if ok else '✗'}")
    print(f"   → {'全部一致' if all_ok else '存在不一致！'}")

    # ------------------------------------------------------------ 3. 路由决策
    print()
    print("3. 路由决策（Controller + remote_bert，threshold=0.5）")
    strong_model = os.environ.get("ROUTELLM_STRONG_MODEL", "openai/qwen3.7-max")
    weak_model = os.environ.get("ROUTELLM_WEAK_MODEL", "openai/qwen3.5-flash")
    with_controller = Controller(
        routers=["remote_bert"],
        strong_model=strong_model,
        weak_model=weak_model,
        config={"remote_bert": {"base_url": service}},
    )

    cases = [
        ("What is 1+1?", 0.5, "weak-model"),
        ("Write a poem about the sea.", 0.5, "weak-model"),
        ("Prove that the square root of 2 is irrational using Galois theory.", 0.5, "?"),
    ]
    print(f"   {'prompt':<52} {'win_rate':>9} {'阈值':>5} {'路由到':<12}")
    for prompt, thr, _exp in cases:
        wr = with_controller.routers["remote_bert"].calculate_strong_win_rate(prompt)
        routed = with_controller.route(prompt, "remote_bert", thr)
        short = "strong" if routed == strong_model else "weak"
        print(f"   {prompt[:51]:<52} {wr:>9.4f} {thr:>5} {short:<12}")

    # ------------------------------------------------------------ 4. 批量
    print()
    print("4. 批量评分（100 条，验证 batch 接口）")
    prompts = [f"Batch end-to-end probe number {i} for validation." for i in range(100)]
    t = time.time()
    wrs = router.calculate_strong_win_rate_batch(prompts)
    elapsed = (time.time() - t) * 1000
    print(f"   {len(wrs)} 条，耗时 {elapsed:.1f}ms（{elapsed/len(wrs):.2f}ms/条）")
    print(f"   win_rate 范围: {min(wrs):.4f} ~ {max(wrs):.4f}")

    # ------------------------------------------------------------ 5. 错误处理
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
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
