"""sw_ranking host 推理服务的单元测试（TDD RED 阶段）。

背景：
    sw_ranking 需要 bge-m3 + arena 向量 + Elo 回归三样东西。若全塞进网关容器，
    会打破 ADR-001「模型推理下沉到 host」的轻量网关设计（镜像 675MB 无 torch）。
    因此把整个 sw_ranking 推理做成 host 侧服努，接口与现有 BERT 服务
    (services/inference_server.py) 的 /v1/score 对齐 —— 容器侧只需把路由名
    从 remote_bert 换成 remote_sw_ranking，保持零挂载。

设计约束（照搬 inference_server.py 的既有约定）：
    - 独立实现，**不 import routellm 包**，保持解耦
    - /health 返回 model_loaded + 关键配置 + 统计，供容器探针使用
    - /v1/score 支持 batch（MMLU 14000 题场景）
    - /selfcheck 用固定输入验证一致性

运行：
    pytest tests/test_sw_ranking_server.py -v
"""

import inspect
import os

import numpy as np
import pytest

BGE_M3_PATH = os.environ.get("ROUTELLM_BGE_M3_PATH", "")


# --------------------------------------------------------------- 模块契约


def test_module_exists_and_is_importable():
    """服务模块应存在且可导入。"""
    import importlib

    mod = importlib.import_module("services.sw_ranking_server")
    assert hasattr(mod, "app"), "缺少 FastAPI app"


def test_does_not_import_routellm_package():
    """服务须独立实现，不 import routellm 包（保持解耦，可单独部署）。"""
    src = inspect.getsource(__import__("services.sw_ranking_server", fromlist=["x"]))
    # 允许在注释/docstring 提及，但不得有实际 import 语句
    bad = [
        ln
        for ln in src.splitlines()
        if ln.strip().startswith(("import routellm", "from routellm"))
    ]
    assert not bad, f"不应 import routellm 包：{bad}"


def test_score_endpoint_matches_bert_service_shape():
    """接口形状须与现有 BERT 服务的 /v1/score 一致，容器侧才能复用同一调用逻辑。"""
    from services.sw_ranking_server import ScoreRequest, ScoreResponse

    req_fields = set(ScoreRequest.model_fields)
    assert "prompts" in req_fields

    resp_fields = set(ScoreResponse.model_fields)
    for f in ("results", "model_type", "count", "elapsed_ms"):
        assert f in resp_fields, f"响应缺少字段 {f}"


def test_routes_registered():
    """/health、/v1/score、/selfcheck 三个端点须齐全。"""
    from services.sw_ranking_server import app

    paths = {r.path for r in app.routes}
    for p in ("/health", "/v1/score", "/selfcheck"):
        assert p in paths, f"缺少端点 {p}"


# --------------------------------------------------------------- 推理正确性


def _config():
    """构造服务所需的路径配置。

    judge 数据可选：提供时走官方完整配置（arena + judge 拼接），
    未提供时仅 arena（会与官方有系统偏差，仅供链路测试）。
    """
    return {
        "model_path": BGE_M3_PATH,
        "battles_csv": os.environ.get("ROUTELLM_ARENA_CSV", ""),
        "embeddings_npy": os.environ.get("ROUTELLM_ARENA_EMBEDDINGS", ""),
        "judge_parquet": os.environ.get("ROUTELLM_JUDGE_PARQUET") or None,
        "judge_embeddings": os.environ.get("ROUTELLM_JUDGE_EMBEDDINGS") or None,
    }


def _load(cfg):
    """按配置加载（含可选 judge 数据集）。"""
    from services.sw_ranking_server import load_router

    load_router(
        cfg["model_path"],
        cfg["battles_csv"],
        cfg["embeddings_npy"],
        judge_parquet=cfg["judge_parquet"],
        judge_embeddings=cfg["judge_embeddings"],
    )


needs_real = pytest.mark.skipif(
    not (BGE_M3_PATH and os.environ.get("ROUTELLM_ARENA_CSV")),
    reason="需 ROUTELLM_BGE_M3_PATH / ROUTELLM_ARENA_CSV / ROUTELLM_ARENA_EMBEDDINGS",
)


@needs_real
def test_load_and_score():
    """加载真实数据后，score 应返回 [0,1] 的 win_rate。"""
    from services.sw_ranking_server import score_batch

    cfg = _config()
    _load(cfg)

    results = score_batch(["What is the capital of France?"], return_detail=False)
    assert len(results) == 1
    wr = results[0]["win_rate"]
    assert 0.0 <= wr <= 1.0, f"win_rate 越界: {wr}"


@needs_real
def test_batch_scoring():
    """batch 应一次处理多条，条数与输入一致。"""
    from services.sw_ranking_server import score_batch

    cfg = _config()
    _load(cfg)

    prompts = [f"Test prompt number {i}, reasonably long for encoding." for i in range(5)]
    results = score_batch(prompts, return_detail=False)
    assert len(results) == 5
    assert all(0.0 <= r["win_rate"] <= 1.0 for r in results)


@needs_real
def test_deterministic():
    """同一 prompt 多次调用结果须一致（服务不得引入随机性）。"""
    from services.sw_ranking_server import score_batch

    cfg = _config()
    _load(cfg)

    p = "Explain the trade-offs between consistency and availability."
    vals = [score_batch([p], return_detail=False)[0]["win_rate"] for _ in range(3)]
    assert len(set(f"{v:.10f}" for v in vals)) == 1, f"结果不稳定: {vals}"


@needs_real
def test_health_reports_loaded():
    """加载后 /health 应报告 model_loaded=True 与数据规模。"""
    from services.sw_ranking_server import health

    cfg = _config()
    _load(cfg)

    h = health()
    assert h["status"] == "online"
    assert h["model_loaded"] is True
    assert h["dimension"] == 1024
    # 数据规模：仅 arena 时 55361；含 judge 时 164462
    assert h["rows"] in (55361, 164462), f"意外的行数 {h['rows']}"
    if h["rows"] == 164462:
        names = [d["name"] for d in h.get("datasets", [])]
        assert names == ["arena", "judge"], f"datasets 应为 arena+judge，实际 {names}"


@needs_real
def test_not_loaded_raises_503():
    """模型未加载时 score 应报 503（供容器探针判断）。"""
    from fastapi import HTTPException

    import services.sw_ranking_server as srv

    saved = srv.EMBEDDER
    try:
        srv.EMBEDDER = None
        with pytest.raises(HTTPException) as ei:
            srv.score(srv.ScoreRequest(prompts=["hi"]))
        assert ei.value.status_code == 503
    finally:
        srv.EMBEDDER = saved


@needs_real
def test_selfcheck_has_baseline():
    """/selfcheck 应返回固定 probe 与其 win_rate（可跨版本比对一致性）。"""
    from services.sw_ranking_server import selfcheck

    cfg = _config()
    _load(cfg)

    sc = selfcheck()
    assert "probe" in sc and "win_rate" in sc
    assert 0.0 <= sc["win_rate"] <= 1.0
