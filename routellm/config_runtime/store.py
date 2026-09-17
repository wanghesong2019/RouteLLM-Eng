"""运行时配置存储（配置热更新的核心）。

设计依据：方案文档 4.8 节 —— 方案C「持久化配置中心」

要解决的问题（问题7）
--------------------
下游 LLM 的 base_url / api_key / 模型名原先通过环境变量在启动时注入
（`Settings.from_env()`），任何变更都需重建容器（服务中断 ~30s）。
本模块让这些配置可**运行时修改并立即生效**。

核心设计
--------
1. **不可变配置对象（frozen dataclass）**
   配置值整体封装在一个 frozen 对象里。请求处理时读到的一定是某个
   完整版本，绝不会读到"半新半旧"的组合。

2. **原子替换**
   `update()` 构造**新对象**并一次性替换引用，而非就地修改旧对象。
   CPython 中对象引用赋值是原子的，因此读取方**无需加锁**。

3. **无锁读**
   `load()` 只是读一个引用（O(1)），不引入任何竞争。这是选择
   "不可变 + 替换"而非"可变 + 加锁"的关键收益 —— 请求路径零开销。

4. **JSON 持久化 + 600 权限**
   配置文件挂载在卷上，容器重建不丢。文件含密钥，权限收为 600。

5. **回落链**
   配置文件存在 → 用它；不存在/损坏 → 回落环境变量（兼容既有部署）。

为何不做加密存储
----------------
容器内无 KMS/密钥管理服务，自实现加密（密钥随代码同处）安全性提升有限
却增加复杂度。当前采取：**文件 600 权限 + API 掩码回显 + 日志脱敏**。
真要在多租户环境加固，应引入外部密钥管理 —— 列为演进项。
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "/data/runtime_config.json"

# 掩码：保留前缀与后缀各若干字符，中间打码
_MASK_HEAD = 3
_MASK_TAIL = 4
_MASK = "***"


@dataclass(frozen=True)
class RuntimeConfig:
    """下游 LLM 的运行时配置（不可变）。

    frozen=True 是原子替换的基础 —— 任何"修改"都必须构造新对象。
    """

    strong_model: str
    weak_model: str
    api_base: str
    api_key: str


def mask_secret(value: str) -> str:
    """遮蔽密钥：保留少量首尾字符便于辨识，中间打码。

    短字符串（不足以保留首尾）直接整体打码，避免泄漏。
    """
    if not value:
        return ""
    if len(value) <= _MASK_HEAD + _MASK_TAIL:
        return _MASK
    return f"{value[:_MASK_HEAD]}{_MASK}{value[-_MASK_TAIL:]}"


class RuntimeConfigStore:
    """运行时配置存储（进程内单例式使用）。

    Args:
        path: JSON 配置文件路径。默认 /data/runtime_config.json（容器挂载卷）。

    线程安全：`load()` 无锁（读引用）；`update()` 用写锁串行化落盘，
    但配置对象的替换本身是原子的。
    """

    def __init__(self, path: Optional[str] = None, env: Optional[dict] = None):
        self.path = path or os.environ.get("ROUTELLM_RUNTIME_CONFIG", DEFAULT_CONFIG_PATH)
        self._env = env if env is not None else os.environ
        self._lock = threading.Lock()
        self._config: RuntimeConfig = self._load_initial()

    # ------------------------------------------------------------------ 初始化

    def _from_env(self) -> RuntimeConfig:
        """从环境变量构造（既有部署的兼容路径）。"""
        e = self._env

        def g(name: str, default: str = "") -> str:
            return e.get(f"ROUTELLM_{name}") or default

        return RuntimeConfig(
            strong_model=g("STRONG_MODEL"),
            weak_model=g("WEAK_MODEL"),
            api_base=g("API_BASE"),
            api_key=g("API_KEY"),
        )

    def _load_initial(self) -> RuntimeConfig:
        """优先读配置文件；不存在或损坏则回落环境变量。"""
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as f:
                    data = json.load(f)
                cfg = RuntimeConfig(
                    strong_model=data.get("strong_model", ""),
                    weak_model=data.get("weak_model", ""),
                    api_base=data.get("api_base", ""),
                    api_key=data.get("api_key", ""),
                )
                logger.info("运行时配置已从 %s 加载", self.path)
                return cfg
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "配置文件 %s 读取失败（%s），回落到环境变量", self.path, e
                )

        cfg = self._from_env()
        logger.info("运行时配置由环境变量初始化（strong=%s）", cfg.strong_model or "<空>")
        return cfg

    # ------------------------------------------------------------------ 读

    def load(self) -> RuntimeConfig:
        """读取当前配置（无锁，O(1)）。

        返回的对象的字段组合一定是一致的 —— 因为配置只在 update 时
        整体替换，不会就地修改。
        """
        return self._config

    def masked(self) -> Dict[str, Any]:
        """掩码视图（供 GET 接口回显，不含密钥明文）。"""
        d = asdict(self._config)
        d["api_key"] = mask_secret(self._config.api_key)
        d["api_key_masked"] = True
        return d

    # ------------------------------------------------------------------ 写

    def update(self, **fields: Any) -> RuntimeConfig:
        """更新配置并立即生效（原子替换 + 落盘）。

        只更新传入的字段；未传入的保持原值。传入 None 表示"不修改"
        （避免前端表单未填的字段覆盖已有值）。

        Returns:
            新的 RuntimeConfig 对象（也是 load() 之后会返回的对象）。
        """
        cleaned = {k: v for k, v in fields.items() if v is not None}

        with self._lock:
            current = self._config
            # dataclasses.replace 构造新对象 —— 不修改旧的
            new_cfg = replace(current, **cleaned)
            self._config = new_cfg  # 原子替换
            self._persist(new_cfg)

        changed = ", ".join(sorted(cleaned.keys())) or "(无)"
        logger.info("运行时配置已更新: %s", changed)
        return new_cfg

    def _persist(self, cfg: RuntimeConfig) -> None:
        """写盘。用临时文件 + rename 保证原子性（避免读到写一半的文件）。"""
        try:
            d = os.path.dirname(os.path.abspath(self.path))
            if d:
                os.makedirs(d, exist_ok=True)

            fd, tmp = tempfile.mkstemp(dir=d or ".", prefix=".rtcfg-", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)
                os.chmod(tmp, 0o600)  # 含密钥
                os.replace(tmp, self.path)  # 原子替换
            except Exception:
                # 清理临时文件后抛出
                if os.path.exists(tmp):
                    os.unlink(tmp)
                raise
        except Exception as e:  # noqa: BLE001
            # 落盘失败不应影响内存中的热更新（服务仍可用，重启会丢）
            logger.error("配置落盘失败（内存配置已生效，重启将丢失）: %s", e)
