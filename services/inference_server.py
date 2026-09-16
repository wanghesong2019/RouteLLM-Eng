"""RouteLLM 路由模型推理服务。

把 RouteLLM 的路由分类模型（BERT / CausalLLM）从 RouteLLM 进程内下沉到 host
独立服务，RouteLLM 容器通过 HTTP 调用。设计见 docs/decisions/ADR-001。

独立实现，不 import routellm 包 —— 只依赖 transformers + fastapi，保持解耦。

接口设计依据
------------
- 返回 win_rate + softmax 分布：softmax 用于诊断（判断模型是否异常），
  实验记录见 docs/experiments/2026-09-16-bert-router-validation.md
- 支持 batch：MMLU 评测有 14000+ 题，逐条 HTTP 请求不现实
- 语义：win_rate = 1 - sum(softmax(logits)[-2:])，表示"应路由到强模型的程度"
  win_rate >= threshold → 路由到强模型，否则弱模型（与上游 Router.route 一致）

启动
----
    python inference_server.py --model-type bert --model-path <ckpt> --port 6070

自检
----
    curl -s localhost:6070/selfcheck
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("routellm.inference")

# ---------------------------------------------------------------------------
# 全局状态（进程内单例）
# ---------------------------------------------------------------------------
MODEL: Optional[torch.nn.Module] = None
TOKENIZER: Optional[Any] = None
DEVICE: str = "cpu"
CONFIG: Dict[str, Any] = {}
START_TS: float = 0.0
INFER_COUNT: int = 0
INFER_TOTAL_MS: float = 0.0

# CausalLLM 接口占位：当前未实现（权重 17GB，待链路验证后接入）
SUPPORTED_TYPES = ("bert", "causal_llm")


# ---------------------------------------------------------------------------
# 请求 / 响应模型
# ---------------------------------------------------------------------------
class ScoreRequest(BaseModel):
    prompts: List[str] = Field(..., min_length=1, description="待评分 prompt 列表")
    return_softmax: bool = Field(False, description="是否返回完整 softmax 分布")


class ScoreResult(BaseModel):
    win_rate: float
    softmax: Optional[List[float]] = None


class ScoreResponse(BaseModel):
    results: List[ScoreResult]
    model_type: str
    count: int
    elapsed_ms: float


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------
def load_model(model_type: str, model_path: str, num_labels: int = 3) -> None:
    global MODEL, TOKENIZER, DEVICE, CONFIG, START_TS

    if model_type == "causal_llm":
        # 接口占位：CausalLLM 需要处理词表末尾追加的 5 个特殊 token
        # （见模型文件 new_embeddings.safetensors 与上游 CausalLLMClassifier），
        # 实现复杂度高，待 BERT 链路验证完成后再接入。
        raise NotImplementedError(
            "causal_llm 推理尚未实现。当前仅支持 bert。"
            "计划：链路验证完成后接入（需处理词表追加的特殊 token）。"
        )

    if model_type != "bert":
        raise ValueError(f"未知 model_type={model_type}，支持: {SUPPORTED_TYPES}")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("loading %s from %s (device=%s)", model_type, model_path, DEVICE)
    t = time.time()

    MODEL = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
    TOKENIZER = AutoTokenizer.from_pretrained(model_path)
    MODEL = MODEL.to(DEVICE).eval()

    elapsed = time.time() - t
    CONFIG = {
        "model_type": model_type,
        "model_path": model_path,
        "arch": MODEL.config.model_type,
        "num_labels": MODEL.config.num_labels,
        "device": DEVICE,
        "load_seconds": round(elapsed, 2),
    }
    START_TS = time.time()
    logger.info("model loaded in %.1fs: %s", elapsed, CONFIG)


# ---------------------------------------------------------------------------
# 核心推理（复刻上游 BERTRouter.calculate_strong_win_rate）
# ---------------------------------------------------------------------------
@torch.no_grad()
def score_batch(prompts: List[str], return_softmax: bool) -> List[Dict[str, Any]]:
    inputs = TOKENIZER(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    logits = MODEL(**inputs).logits.cpu().numpy()

    out: List[Dict[str, Any]] = []
    for row in logits:
        exp = np.exp(row - np.max(row))
        softmax = exp / np.sum(exp)
        # 与上游一致：binary_prob 取末尾两类（tie / 弱模型胜），win_rate = 1 - binary_prob
        binary_prob = float(np.sum(softmax[-2:]))
        item: Dict[str, Any] = {"win_rate": float(1 - binary_prob)}
        if return_softmax:
            item["softmax"] = [float(x) for x in softmax]
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RouteLLM Inference Service",
    description="RouteLLM 路由分类模型的独立推理服务（host 侧）",
    version="0.1.0",
)


@app.get("/health")
def health() -> Dict[str, Any]:
    """健康检查。模型未加载时返回 not_ready，供容器启动探针使用。"""
    ready = MODEL is not None
    return {
        "status": "online" if ready else "not_ready",
        "model_loaded": ready,
        "uptime_seconds": round(time.time() - START_TS, 1) if ready else 0,
        **CONFIG,
        "stats": {
            "infer_count": INFER_COUNT,
            "avg_ms": round(INFER_TOTAL_MS / INFER_COUNT, 2) if INFER_COUNT else 0.0,
        },
    }


@app.get("/v1/models")
def list_models() -> Dict[str, Any]:
    """列出可用路由模型（对齐 OpenAI /v1/models 格式）。"""
    return {
        "object": "list",
        "data": [
            {
                "id": f"router-model-{CONFIG.get('model_type', 'unknown')}",
                "object": "model",
                "owned_by": "routellm-inference",
                "created": int(START_TS) if START_TS else 0,
            }
        ],
    }


@app.post("/v1/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> Dict[str, Any]:
    """批量计算 prompt 的 win rate。

    win_rate 语义：应路由到强模型的程度（与上游 Router.route 一致）。
    """
    global INFER_COUNT, INFER_TOTAL_MS

    if MODEL is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    t = time.time()
    try:
        results = score_batch(req.prompts, req.return_softmax)
    except Exception as e:  # noqa: BLE001
        logger.exception("inference failed")
        raise HTTPException(status_code=500, detail=f"inference error: {e}") from e

    elapsed_ms = (time.time() - t) * 1000
    INFER_COUNT += len(req.prompts)
    INFER_TOTAL_MS += elapsed_ms

    return {
        "results": results,
        "model_type": CONFIG.get("model_type", "unknown"),
        "count": len(results),
        "elapsed_ms": round(elapsed_ms, 2),
    }


@app.get("/selfcheck")
def selfcheck() -> Dict[str, Any]:
    """自检：用固定输入验证服务与上游实现的一致性。

    上游基准值由 scripts/verify_bert_basic.py 产出，用于确认服务未引入偏差。
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    probe = "What is 1+1?"
    res = score_batch([probe], return_softmax=True)[0]
    return {
        "probe": probe,
        "win_rate": res["win_rate"],
        "softmax": res["softmax"],
        "expected": {
            "win_rate": 0.2969909906,
            "note": "上游 BERTRouter 基准值，来自 docs/experiments/2026-09-16-bert-router-validation.md",
        },
        "matches": abs(res["win_rate"] - 0.2969909906) < 1e-6,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="RouteLLM 推理服务")
    ap.add_argument("--model-type", default="bert", choices=SUPPORTED_TYPES)
    ap.add_argument("--model-path", required=True, help="模型 checkpoint 路径")
    ap.add_argument("--port", type=int, default=6070)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--gpu", type=int, default=None, help="指定 GPU 序号（设置 CUDA_VISIBLE_DEVICES）")
    ap.add_argument("--num-labels", type=int, default=3)
    args = ap.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    load_model(args.model_type, args.model_path, args.num_labels)
    logger.info("starting server on %s:%s", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
