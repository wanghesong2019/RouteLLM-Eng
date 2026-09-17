#!/usr/bin/env python3
"""端到端验证：强弱分侧配置 + 原始模型名 + 分侧连通性预检。

在 43 上对**真实运行中的容器**执行，验证三件事：

1. 弱模型也有完整的 base_url / api_key 配置项，且强弱各自生效
2. 模型名以**原始名**存储/回显（前缀由网关调用时自动补）
3. 两个「测试连接」按钮各测各的（verify?tier=strong / weak）

用法（在 43 上执行）：
    # 走面板链路（推荐，验证转发层）
    GW=http://127.0.0.1:8080 ROUTELLM_GATEWAY_API_KEY=<面板容器的key> \
        python3 verify_per_tier_config.py
    # 直连网关（仅验证网关侧）
    GW=http://127.0.0.1:6060 ROUTELLM_GATEWAY_API_KEY=<网关key> \
        python3 verify_per_tier_config.py

在**面板容器内**执行时用 http://127.0.0.1:8080（容器内端口）；
在宿主机执行时用 http://127.0.0.1:8092（映射端口）。

注意：本脚本会**修改配置**（写入分侧字段），执行前会先备份现有配置，
结束后恢复原值。
"""

import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("GW", "http://127.0.0.1:8092")
GW_KEY = os.environ.get("ROUTELLM_GATEWAY_API_KEY", "")


def call(method, path, body=None):
    return call_to(BASE, method, path, body)


def call_to(base, method, path, body=None):
    """向指定 base 发请求，返回 (status, json)。

    不抛异常：4xx/5xx 也返回状态码与解析后的响应体，
    便于脚本断言"应当被拒绝"这类预期失败。
    """
    url = base.rstrip("/") + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if GW_KEY:
        req.add_header("Authorization", f"Bearer {GW_KEY}")
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, {"raw": raw[:400]}
    except Exception as e:  # noqa: BLE001
        return 0, {"error": f"{type(e).__name__}: {e}"}


def main():
    print("=" * 70)
    print(f"端到端验证 · 目标 {BASE}")
    print("=" * 70)

    # ---------------------------------------------------------- 0. 备份
    st, before = call("GET", "/api/config")
    if not isinstance(before, dict):
        before = {"raw": str(before)}
    if st != 200 or before.get("proxy_error"):
        print(f"[!] 无法读取配置 (HTTP {st}): {json.dumps(before, ensure_ascii=False)[:300]}")
        return 1
    print(f"\n[0] 当前配置（密钥已掩码）")
    for k in ("strong_model", "weak_model", "api_base", "strong_api_base", "weak_api_base"):
        print(f"    {k:18} = {before.get(k, '')!r}")
    print(f"    api_key            = {before.get('api_key','')!r}")
    print(f"    strong_api_key     = {before.get('strong_api_key','')!r}")
    print(f"    weak_api_key       = {before.get('weak_api_key','')!r}")
    print(f"    single_model_mode  = {before.get('single_model_mode')}")

    backup = {k: before.get(k, "") for k in (
        "strong_model", "weak_model", "api_base",
        "strong_api_base", "weak_api_base")}

    fail = 0

    # ---------------------------------------------------------- 1. 分侧可编辑
    print(f"\n[1] 分侧字段是否可编辑")
    for f in ("strong_api_base", "strong_api_key", "weak_api_base", "weak_api_key"):
        ok = f in (before.get("editable_fields") or [])
        print(f"    {'OK ' if ok else 'FAIL'} {f} 在 editable_fields 中")
        if not ok:
            fail += 1

    # ---------------------------------------------------------- 2. 原始名存储
    print(f"\n[2] 模型名以原始名存储（无 openai/ 前缀）")
    st, r = call("PUT", "/api/config", {"strong_model": "test-org/Test-Model-Raw"})
    print(f"    PUT strong_model=test-org/Test-Model-Raw -> HTTP {st}, ok={r.get('ok')}")
    st, after = call("GET", "/api/config")
    got = after.get("strong_model")
    ok = got == "test-org/Test-Model-Raw"
    print(f"    {'OK ' if ok else 'FAIL'} 回显 = {got!r}（应无 openai/ 前缀）")
    if not ok:
        fail += 1

    # ---------------------------------------------------------- 3. 分侧各自生效
    print(f"\n[3] 强弱各自配置（隔离性）")
    st, r = call("PUT", "/api/config", {
        "strong_api_base": "https://strong.example.com/v1",
        "weak_api_base": "https://weak.example.com/v1",
    })
    print(f"    PUT 分侧 base_url -> HTTP {st}, changed={r.get('changed')}")
    st, after = call("GET", "/api/config")
    ok_s = after.get("strong_api_base") == "https://strong.example.com/v1"
    ok_w = after.get("weak_api_base") == "https://weak.example.com/v1"
    print(f"    {'OK ' if ok_s else 'FAIL'} strong_api_base = {after.get('strong_api_base')!r}")
    print(f"    {'OK ' if ok_w else 'FAIL'} weak_api_base   = {after.get('weak_api_base')!r}")
    if not (ok_s and ok_w):
        fail += 1

    # ---------------------------------------------------------- 4. 分侧预检
    print(f"\n[4] 两个「测试连接」各测各的")
    for tier in ("strong", "weak"):
        st, r = call("POST", f"/api/config/verify?tier={tier}")
        probed = r.get("probed") or {}
        ok = r.get("tier") == tier and probed.get("tier") == tier
        print(f"    [tier={tier}] HTTP {st}")
        print(f"        ok       = {r.get('ok')}")
        print(f"        detail   = {r.get('detail')}")
        print(f"        测的模型  = {probed.get('model')!r}")
        print(f"        测的地址  = {probed.get('api_base')!r}")
        print(f"        凭据来源  = {probed.get('source')!r}")
        print(f"        api_key  = {probed.get('api_key')!r}")
        print(f"        {'OK ' if ok else 'FAIL'} 响应明确标识了被测对象为 {tier}")
        if not ok:
            fail += 1

    # 非法 tier 应被拒绝（Phase 3.6 后：面板保留状态码，不再包成 200）
    st, r = call("POST", "/api/config/verify?tier=medium")
    ok = st >= 400
    detail = r.get("detail") if isinstance(r, dict) else r
    print(f"    {'OK ' if ok else 'FAIL'} 非法 tier=medium -> HTTP {st}（应 >=400）")
    print(f"        detail = {str(detail)[:120]}")
    if not ok:
        fail += 1

    # ---------------------------------------------------------- 5. 单模型兜底
    print(f"\n[5] 只配一个模型的兜底（不报错、不路由到空模型）")
    st, r = call("PUT", "/api/config", {"weak_model": ""})
    print(f"    PUT weak_model='' -> HTTP {st}, ok={r.get('ok')}")
    st, after = call("GET", "/api/config")
    print(f"    single_model_mode     = {after.get('single_model_mode')}")
    print(f"    effective_strong_model= {after.get('effective_strong_model')!r}")
    print(f"    effective_weak_model  = {after.get('effective_weak_model')!r}")
    ok = (after.get("single_model_mode") is True
          and after.get("effective_weak_model") == after.get("effective_strong_model")
          and after.get("effective_weak_model") != "")
    print(f"    {'OK ' if ok else 'FAIL'} 单模型模式下弱档并入已配档（非空）")
    if not ok:
        fail += 1

    # ---------------------------------------------------------- 6. 真实请求
    # 注意：/v1/chat/completions 只在网关（6060）上，面板不代理该端点。
    # 因此这一步直连网关，验证「单模型模式下不会带空模型名去调下游」。
    print(f"\n[6] 单模型模式下真实请求（直连网关，验证不会带空模型名）")
    gw = os.environ.get("ROUTELLM_GATEWAY_DIRECT", "http://127.0.0.1:6060")
    payload = {
        "model": "router-bert",
        "messages": [{"role": "user", "content": "hello, say hi in one word"}],
        "max_tokens": 8,
    }
    st, r = call_to(gw, "POST", "/v1/chat/completions", payload)
    body = json.dumps(r, ensure_ascii=False)
    if st == 200:
        print(f"    OK   HTTP 200 · 实际使用模型 = {r.get('model', '?')}")
    elif "空" in body and "模型" in body:
        print(f"    FAIL HTTP {st}: {body[:200]}")
        print(f"         出现了「空模型名」类错误 —— 兜底未生效")
        fail += 1
    else:
        print(f"    ?    HTTP {st}: {body[:220]}")
        print(f"         （下游 provider 不可达属预期；关键是**不是**空模型名报错）")

    # ---------------------------------------------------------- 恢复
    print(f"\n[7] 恢复原配置")
    restore = {k: v for k, v in backup.items()}
    restore["strong_api_base"] = before.get("strong_api_base", "")
    restore["weak_api_base"] = before.get("weak_api_base", "")
    st, r = call("PUT", "/api/config", restore)
    print(f"    PUT 恢复 -> HTTP {st}, changed={r.get('changed')}")
    st, final = call("GET", "/api/config")
    print(f"    恢复后 strong_model = {final.get('strong_model')!r}, "
          f"weak_model = {final.get('weak_model')!r}")

    print("\n" + "=" * 70)
    print(f"结果：{'全部通过' if fail == 0 else f'{fail} 项未通过'}")
    print("=" * 70)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
