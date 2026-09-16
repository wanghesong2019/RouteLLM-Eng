#!/usr/bin/env python3
"""推理服务端到端测试：batch、延迟、并发、错误处理。

对应 ADR-001 的三项验证要求：
  1. win rate 与直接加载模型逐位一致  → /selfcheck 已验证
  2. HTTP 往返延迟对路由总延迟的影响可忽略
  3. 支持 batch（MMLU 14000 题逐条请求不现实）
"""
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "http://127.0.0.1:6070"


def post(path, payload, timeout=120):
    req = urllib.request.Request(f"{BASE}{path}", data=json.dumps(payload).encode())
    req.add_header("Content-Type", "application/json")
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode())
    return body, (time.time() - t) * 1000


def get(path, timeout=30):
    with urllib.request.urlopen(f"{BASE}{path}", timeout=timeout) as r:
        return json.loads(r.read().decode())


print("=" * 66)
print("1. 单条延迟（含 HTTP 开销，20 次）")
print("=" * 66)
lat = []
for _ in range(20):
    _, ms = post("/v1/score", {"prompts": ["Latency probe for benchmarking."]})
    lat.append(ms)
lat.sort()
print(f"  min={lat[0]:.1f}ms  p50={lat[len(lat)//2]:.1f}ms  p95={lat[int(len(lat)*0.95)]:.1f}ms  max={lat[-1]:.1f}ms")

print()
print("=" * 66)
print("2. Batch 扩展性")
print("=" * 66)
print(f"  {'batch':>6} {'total_ms':>10} {'per_item_ms':>12} {'speedup':>8}")
base_per = None
for n in (1, 8, 32, 128):
    prompts = [f"Batch probe item number {i}, please answer." for i in range(n)]
    body, ms = post("/v1/score", {"prompts": prompts})
    per = ms / n
    if base_per is None:
        base_per = per
    print(f"  {n:>6} {ms:>10.1f} {per:>12.2f} {base_per/per:>7.2f}x")
    assert body["count"] == n, f"count mismatch: {body['count']} != {n}"

print()
print("=" * 66)
print("3. 并发（10 线程 × 每条 4 prompt）")
print("=" * 66)
probe = {"prompts": ["Concurrent probe, explain distributed consensus."] * 4}


def one_call(_):
    b, ms = post("/v1/score", probe)
    return b["count"], ms


t = time.time()
with ThreadPoolExecutor(max_workers=10) as ex:
    results = list(ex.map(one_call, range(10)))
wall = (time.time() - t) * 1000
total_items = sum(r[0] for r in results)
print(f"  10 并发请求 × 4 prompt = {total_items} items")
print(f"  墙钟总耗时: {wall:.1f}ms")
print(f"  平均单请求耗时: {sum(r[1] for r in results)/len(results):.1f}ms")
print(f"  吞吐: {total_items / (wall/1000):.1f} items/s")

print()
print("=" * 66)
print("4. 与 MMLU 评测场景的匹配度（14000 题）")
print("=" * 66)
body, ms = post("/v1/score", {"prompts": [f"MMLU probe {i}" for i in range(500)]})
per_item = ms / 500
print(f"  batch=500 实测: {ms:.1f}ms  ({per_item:.2f}ms/item)")
print(f"  推算 14000 题（分 28 批 × 500）: {per_item*14000/1000:.1f}s")
print()
print("=" * 66)
print("5. 错误处理")
print("=" * 66)


def try_post(payload, label):
    try:
        b, _ = post("/v1/score", payload, timeout=20)
        print(f"  {label}: OK count={b.get('count')}")
    except urllib.error.HTTPError as e:
        detail = json.loads(e.read().decode()).get("detail", "")
        print(f"  {label}: HTTP {e.code} — {str(detail)[:70]}")
    except Exception as e:
        print(f"  {label}: {type(e).__name__} — {str(e)[:70]}")


try_post({"prompts": []}, "空列表")
try_post({}, "缺 prompts 字段")
try_post({"prompts": ["ok"], "return_softmax": True}, "正确请求")

print()
print("=== 服务统计 ===")
h = get("/health")
print(f"  infer_count={h['stats']['infer_count']}  avg_ms={h['stats']['avg_ms']}")
print("TEST_DONE")
