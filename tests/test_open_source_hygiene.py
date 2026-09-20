"""开源卫生守卫的回归测试（TDD）。

背景
----
守卫原本只覆盖「内网 IP 字面量 / 内部主机名 / 私有域名 / 凭据」四类，
但实际评审发现另一类泄漏长期漏检：**裸主机编号 + 部署拓扑描述**。
典型如 ``nginx → 33:16060 → 隧道 → 43:6060`` —— 这里没有内网 IP、
没有主机名、没有私有域名，却是完整可复现的内部访问链路。

本测试锁定新增规则的行为：
1. 裸主机编号（33/43 等 ``<ip-or-host>:<port>`` 形式）应被报告
2. 拓扑描述词（隧道 / NAT hairpin / 传输镜像 等）应被报告
3. 正常文本（测试计数「43 例」、版本号、时间戳之类）不得误报
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GUARD_PATH = REPO_ROOT / "scripts" / "check_open_source_hygiene.py"


def _load_guard():
    """把守卫脚本作为模块加载（它不在 installable package 里）。"""
    spec = importlib.util.spec_from_file_location("hygiene_guard", GUARD_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hygiene_guard"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def guard():
    return _load_guard()


def _scan_text(guard, tmp_path: Path, text: str):
    """在临时目录写入 text，跑守卫的内容检查，返回违规类别集合。"""
    (tmp_path / "sample.md").write_text(text, encoding="utf-8")
    violations = guard.check_content(str(tmp_path))
    return {v.category for v in violations}


@pytest.mark.parametrize(
    "leaky",
    [
        # 完整内部链路：主机编号出现在 <host>:<port> 位置
        "背景：公网通过 nginx → 33:16060 → 隧道 → 43:6060(网关) 访问 dashboard。",
        # 单侧主机编号 + 端口
        "端到端验证（33 → SSH 隧道 → 43:6070）：",
        # 无端口但显式指代为节点
        "改在 33 构建后传输镜像到部署机 43。",
    ],
)
def test_bare_host_number_with_port_is_reported(guard, tmp_path, leaky):
    """RED 阶段：裸主机编号泄漏必须被报告（当前守卫应漏检）。"""
    assert "topology_fingerprint" in _scan_text(guard, tmp_path, leaky), (
        f"未报告主机编号泄漏: {leaky!r}"
    )


@pytest.mark.parametrize(
    "leaky",
    [
        "部署机防火墙仅开放 SSH 端口，故采用 SSH 隧道 + 反向代理暴露。",
        "私有域名被 DNS 解析到本机公网 IP，不可达（NAT hairpin 行为）。",
        "镜像 675MB（构建机构建 → docker save → 传输 → 部署机 docker load）。",
    ],
)
def test_internal_topology_phrasing_is_reported(guard, tmp_path, leaky):
    """拓扑描述词（与其配套的主机编号一起）应被报告。"""
    # 这些句子单独看可能只是通用描述；与主机编号组合才是指纹。
    combined = "43 部署机：" + leaky
    assert "topology_fingerprint" in _scan_text(guard, tmp_path, combined), (
        f"未报告拓扑指纹: {combined!r}"
    )


@pytest.mark.parametrize(
    "benign",
    [
        "测试：新增 43 例，全量 238 passed, 17 skipped。",   # 计数
        "换阿里云镜像源 → 43 秒",                          # 耗时
        "模型版本 v2.5，batch=128 → 68.78x 加速",           # 版本/倍率
        "端口 6060 是网关默认端口，可在 compose 中覆盖。",   # 通用端口说明
        "使用 172.17.0.1 之外的 host.docker.internal 网关。",  # 通用 docker 写法
    ],
)
def test_benign_text_is_not_flagged(guard, tmp_path, benign):
    """GREEN 阶段的反向约束：正常文本不得误报（避免误报疲劳）。"""
    assert "topology_fingerprint" not in _scan_text(guard, tmp_path, benign), (
        f"误报: {benign!r}"
    )
