"""远程推理路由器：把模型推理委托给 host 上的独立推理服务。

背景与设计依据见 docs/decisions/ADR-001-inference-service-on-host.md。

上游的 BERTRouter / CausalLLMRouter 在进程内用 transformers 加载模型，
导致容器需要 CUDA torch + 模型权重。本模块提供 HTTP 版本，把推理下沉到
host 服务（services/inference_server.py），容器只保留路由逻辑。

契约与上游完全一致：实现 Router 抽象接口，win_rate 语义相同
（1 - sum(softmax[-2:])，表示应路由到强模型的程度），因此可被
Controller 无缝替换。
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from routellm.routers.routers import Router

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30.0
DEFAULT_BATCH_SIZE = 64


class RemoteInferenceError(Exception):
    """推理服务调用失败，或返回了不可用的响应。"""


class RemoteBERTRouter(Router):
    """通过 HTTP 调用 host 推理服务完成路由评分。

    Args:
        base_url: 推理服务地址，如 ``http://127.0.0.1:6070``
        timeout: 单次 HTTP 请求超时（秒）
        batch_size: 批量评分的分片大小；超过则拆成多次请求
        model_type: 服务侧的模型类型，默认 ``bert``

    用法::

        router = RemoteBERTRouter(base_url="http://host.docker.internal:6070")
        win_rate = router.calculate_strong_win_rate(prompt)
    """

    NO_PARALLEL = True  # 走 HTTP 调用，pandarallel 多进程无意义且引入依赖

    def __init__(
        self,
        base_url: str,
        timeout: float = DEFAULT_TIMEOUT,
        batch_size: int = DEFAULT_BATCH_SIZE,
        model_type: str = "bert",
    ):
        if not base_url:
            raise ValueError("base_url is required")
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.batch_size = int(batch_size)
        self.model_type = model_type

        # 懒加载的 opener，便于测试与复用连接
        self._opener = urllib.request.build_opener()

    # ------------------------------------------------------------------
    # HTTP 层
    # ------------------------------------------------------------------
    def _post_score(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST /v1/score，返回解析后的 JSON。

        独立成方法，便于测试时 mock。
        """
        url = f"{self.base_url}/v1/score"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data)
        req.add_header("Content-Type", "application/json")

        try:
            with self._opener.open(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8")[:300]
            except Exception:  # noqa: BLE001
                pass
            raise RemoteInferenceError(
                f"inference service returned HTTP {e.code} for {url}: {body}"
            ) from e
        except urllib.error.URLError as e:
            raise RemoteInferenceError(
                f"cannot reach inference service at {url}: {e.reason}"
            ) from e
        except json.JSONDecodeError as e:
            raise RemoteInferenceError(
                f"inference service returned invalid JSON: {e}"
            ) from e
        except Exception as e:  # noqa: BLE001
            raise RemoteInferenceError(
                f"unexpected error calling {url}: {type(e).__name__}: {e}"
            ) from e

    @staticmethod
    def _extract_win_rates(resp: Dict[str, Any], expected: int) -> List[float]:
        """从响应中提取 win_rate 列表，校验结构。"""
        if not isinstance(resp, dict) or "results" not in resp:
            raise RemoteInferenceError(
                f"malformed response, missing 'results': {str(resp)[:200]}"
            )
        results = resp["results"]
        if not isinstance(results, list):
            raise RemoteInferenceError(
                f"malformed response, 'results' is not a list: {type(results).__name__}"
            )
        if len(results) != expected:
            raise RemoteInferenceError(
                f"result count mismatch: expected {expected}, got {len(results)}"
            )
        out: List[float] = []
        for i, item in enumerate(results):
            if not isinstance(item, dict) or "win_rate" not in item:
                raise RemoteInferenceError(
                    f"malformed result at index {i}, missing 'win_rate': {str(item)[:120]}"
                )
            out.append(float(item["win_rate"]))
        return out

    # ------------------------------------------------------------------
    # 异步 HTTP 层（方案文档 4.4）
    #
    # 现状：同步 urllib 调用。在 FastAPI 的 async 入口里直接调它会阻塞事件
    # 循环 —— 单 worker 下一个慢请求让所有并发排队。
    #
    # 改造：改用 httpx.AsyncClient + 连接池复用。
    # 为什么必须复用 client 而不是每次新建：新建 client 每次都建立 TCP/TLS
    # 连接，并发反而退化为串行（连接建立是同步等待）。这里懒加载单例，
    # 由 aclose() 释放。
    # ------------------------------------------------------------------
    async def _get_async_client(self):
        """取（或懒加载）复用的 httpx.AsyncClient。"""
        client = getattr(self, "_async_client", None)
        if client is None:
            import httpx

            client = httpx.AsyncClient(
                timeout=self.timeout,
                limits=httpx.Limits(
                    max_connections=max(10, self.batch_size),
                    max_keepalive_connections=10,
                ),
            )
            self._async_client = client
        return client

    async def aclose(self) -> None:
        """关闭异步连接池（服务关停时调用；未创建过则无操作）。"""
        client = getattr(self, "_async_client", None)
        if client is not None:
            try:
                await client.aclose()
            finally:
                self._async_client = None

    async def _async_score_prompts(self, prompts: List[str]) -> List[float]:
        """异步 POST /v1/score。可被测试替换。

        异常语义与同步版一致：统一包装成 RemoteInferenceError，
        让上层（含容错降级链）只面对一种异常类型。
        """
        url = f"{self.base_url}/v1/score"
        try:
            client = await self._get_async_client()
            resp = await client.post(url, json={"prompts": prompts})
            if resp.status_code >= 400:
                raise RemoteInferenceError(
                    f"inference service returned HTTP {resp.status_code} "
                    f"for {url}: {resp.text[:300]}"
                )
            data = resp.json()
        except RemoteInferenceError:
            raise
        except Exception as e:  # noqa: BLE001
            raise RemoteInferenceError(
                f"cannot reach inference service at {url}: "
                f"{type(e).__name__}: {e}"
            ) from e

        return self._extract_win_rates(data, len(prompts))

    async def calculate_strong_win_rate_batch_async(
        self, prompts: List[str]
    ) -> List[float]:
        """异步批量评分，按 batch_size 分片且保持顺序。"""
        if not prompts:
            return []

        out: List[float] = []
        for start in range(0, len(prompts), self.batch_size):
            chunk = prompts[start : start + self.batch_size]
            out.extend(await self._async_score_prompts(chunk))
        return out

    async def calculate_strong_win_rate_async(self, prompt: str) -> float:
        """异步单条评分。"""
        res = await self.calculate_strong_win_rate_batch_async([prompt])
        return res[0]

    # ------------------------------------------------------------------
    # Router 契约
    # ------------------------------------------------------------------
    def calculate_strong_win_rate(self, prompt: str) -> float:
        """单条评分。走批量接口，避免两套代码路径。"""
        return self.calculate_strong_win_rate_batch([prompt])[0]

    def calculate_strong_win_rate_batch(self, prompts: List[str]) -> List[float]:
        """批量评分。超长列表按 batch_size 分片，顺序保持不变。"""
        if not prompts:
            return []

        out: List[float] = []
        for start in range(0, len(prompts), self.batch_size):
            chunk = prompts[start : start + self.batch_size]
            resp = self._post_score({"prompts": chunk})
            out.extend(self._extract_win_rates(resp, len(chunk)))
        return out

    # ------------------------------------------------------------------
    # 运维辅助
    # ------------------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        """查询推理服务健康状态。"""
        url = f"{self.base_url}/health"
        try:
            with self._opener.open(url, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            raise RemoteInferenceError(f"health check failed for {url}: {e}") from e

    def __repr__(self) -> str:
        return (
            f"RemoteBERTRouter(base_url={self.base_url!r}, "
            f"timeout={self.timeout}, batch_size={self.batch_size}, "
            f"model_type={self.model_type!r})"
        )


class RemoteSWRankingRouter(RemoteBERTRouter):
    """通过 HTTP 调用 host 侧的 sw_ranking 推理服务。

    服务实现见 ``services/sw_ranking_server.py``（端口 6071）—— 把
    bge-m3 编码 + 55k 向量相似度 + Elo 回归整条链路下沉到 host，
    容器保持零挂载、无 torch 的轻量形态（设计见 ADR-001）。

    与 :class:`RemoteBERTRouter` 的唯一差别是默认端口与 model_type：
    接口形状（``POST /v1/score``，请求 ``{"prompts": [...]}``，
    响应 ``{"results": [{"win_rate": ...}]}``）完全一致，
    故直接继承复用全部 HTTP 与分片逻辑。

    win_rate 语义与前两者相同：应路由到强模型的程度，可直接被
    Controller 替换使用。

    Args:
        base_url: 服务地址，默认 ``http://host.docker.internal:6071``
        timeout: 单次 HTTP 请求超时（秒）
        batch_size: 批量评分的分片大小
        model_type: 服务侧模型类型，默认 ``sw_ranking``

    用法::

        router = RemoteSWRankingRouter(
            base_url="http://host.docker.internal:6071"
        )
        win_rate = router.calculate_strong_win_rate(prompt)
    """

    DEFAULT_BASE_URL = "http://host.docker.internal:6071"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        batch_size: int = DEFAULT_BATCH_SIZE,
        model_type: str = "sw_ranking",
    ):
        super().__init__(
            base_url=base_url,
            timeout=timeout,
            batch_size=batch_size,
            model_type=model_type,
        )

    def __repr__(self) -> str:
        return (
            f"RemoteSWRankingRouter(base_url={self.base_url!r}, "
            f"timeout={self.timeout}, batch_size={self.batch_size}, "
            f"model_type={self.model_type!r})"
        )
