"""前端面板测试：自适应阈值展示 + error_rate 卡片不得回归。

前端没有浏览器测试框架，这里用**源码断言**守住两条契约：
  1. index.html 必须含自适应阈值面板，并真的 fetch /api/adaptive-threshold
  2. index.html 不得再出现 error_rate 卡片
     （该卡片已由 opensource 分支的 3bceee7 移除；本分支从 main 起，
      需同等移除，否则等于把已修的问题带回来）
"""

from pathlib import Path

INDEX = Path(__file__).resolve().parents[1] / (
    "routellm/monitoring/dashboard/static/index.html"
)


def _html() -> str:
    return INDEX.read_text(encoding="utf-8")


def test_index_exists():
    assert INDEX.is_file(), f"面板首页缺失: {INDEX}"


class TestAdaptiveThresholdPanel:
    def test_fetches_adaptive_threshold_endpoint(self):
        """面板必须真的去取阈值状态，而不是只写一句静态文案。"""
        html = _html()
        assert "/api/adaptive-threshold" in html, (
            "index.html 未调用 /api/adaptive-threshold —— 阈值面板不会有数据"
        )

    def test_has_dedicated_panel_container(self):
        """需要一块独立的展示区（id 供 JS 定位）。"""
        html = _html()
        assert "adaptive" in html.lower()
        # 至少有一个 id 含 adaptive 的元素，供 JS 渲染
        assert 'id="adaptive"' in html or "id='adaptive'" in html

    def test_shows_effective_tau_and_bounds(self):
        """必须展示生效阈值与上下限 —— 验收标准要求"可见 τ 从 0.5 上调"。"""
        html = _html()
        assert "effective_tau" in html, "未使用 effective_tau 字段"
        assert "tau_min" in html and "tau_max" in html, "未展示 τ 区间"

    def test_shows_budget_and_cost_rate(self):
        """预算与成本速率是判断 τ 为何变动的依据，需可见。"""
        html = _html()
        assert "cost_rate" in html
        assert "budget_tokens_per_min" in html

    def test_renders_inside_loadall_cycle(self):
        """面板应随 loadAll() 自动刷新，而不是只在首次加载时渲染一次。"""
        html = _html()
        # loadAll 里必须出现 adaptive 相关调用
        idx = html.find("async function loadAll")
        assert idx != -1, "未找到 loadAll"
        tail = html[idx:]
        assert "adaptive" in tail.lower(), (
            "loadAll() 未刷新自适应阈值面板 —— 10s 自动刷新对它无效"
        )


class TestErrorRateNotReintroduced:
    def test_error_rate_card_absent(self):
        """error_rate 卡片已在 opensource 分支移除，本分支不得带回来。"""
        html = _html()
        assert "错误率" not in html, (
            "index.html 又出现了『错误率』卡片。该卡片已由 3bceee7 移除，"
            "本分支从 main 起需同等移除（否则是跨分支回归）。"
        )
        assert "s.error_rate" not in html, (
            "renderCards 仍在读取 s.error_rate"
        )
