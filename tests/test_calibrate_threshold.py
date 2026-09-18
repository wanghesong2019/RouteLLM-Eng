"""阈值标定脚本的单元测试（TDD RED 阶段）。

被测模块：scripts/calibrate_threshold.py

核心待验证行为：
1. 分位数计算：给定 win_rate 数组与「希望走强的比例 X%」，选出的阈值应使
   实际走强比例最接近 X%（阈值取 (100-X) 分位）。
2. 候选阈值生成：依据实际分布动态生成，不含「永不走强」或「永远走强」的死档。
3. APGR 计算与 routellm/evals 论文公式一致：(router_auc - weak_auc)/(strong_auc - weak_auc)。
4. bootstrap 置信区间：对同一分布多次重采样，CI 下界 <= 点估计 <= CI 上界。
5. 稳定性评估：多轮采样下同一阈值的 APGR 波动可量化（返回 std）。
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_REPO_ROOT, "scripts", "calibrate_threshold.py")


def _load_module():
    """从路径加载标定脚本模块（scripts/ 不是包）。"""
    spec = importlib.util.spec_from_file_location("calibrate_threshold", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cal():
    return _load_module()


def test_script_exists():
    """脚本必须存在于 scripts/ 下。"""
    assert os.path.isfile(_SCRIPT), f"missing calibration script: {_SCRIPT}"


def test_threshold_at_quantile_achieves_target_strong_pct(cal):
    """阈值取 (100-X) 分位时，实际走强比例应接近 X%。"""
    rng = np.random.default_rng(0)
    win_rates = np.clip(rng.normal(0.45, 0.08, 5000), 0.0, 1.0)
    for target_pct in (10, 30, 50):
        th = cal.threshold_for_strong_pct(win_rates, target_pct)
        actual = float(np.mean(win_rates >= th)) * 100
        assert abs(actual - target_pct) <= 3.0, (
            f"target {target_pct}% -> actual {actual:.1f}% (threshold={th})"
        )


def test_candidate_thresholds_stay_inside_distribution(cal):
    """候选阈值必须落在分布范围内，不得产生永不生效的死档。"""
    rng = np.random.default_rng(1)
    win_rates = np.clip(rng.normal(0.42, 0.06, 3000), 0.0, 1.0)
    cands = cal.candidate_thresholds(win_rates, n=5)

    assert len(cands) == 5
    # 降序（pct 递增 -> 阈值递减）
    assert list(cands) == sorted(cands, reverse=True)
    lo, hi = float(win_rates.min()), float(win_rates.max())
    for th in cands:
        assert lo <= th <= hi, f"threshold {th} outside distribution [{lo}, {hi}]"
        # 每个候选档都必须真的分流（既不 0% 也不 100% 走强）
        pct = float(np.mean(win_rates >= th))
        assert 0.0 < pct < 1.0, f"threshold {th} is a dead band (strong pct={pct})"


def test_apgr_matches_paper_formula(cal):
    """APGR 手算校验：构造一个完美先知路由，APGR 应达到该构造下的理论上限。

    注意论文公式的结构性上限：APGR = (router_AUC - weak_AUC)/(strong_AUC - weak_AUC)。
    理论上限出现在「先知路由」—— 逐步把强模型擅长的题先收进来，曲线是上凸的
    积分；而 win_rate 完美单调可分时时，任何切点都只能按 win_rate 直线切分，
    曲线退化为对角线，router_AUC 上限 = strong_AUC/2（weak_acc=0 时）。
    因此这里断言的是 > 0.49 而非接近 1 —— 0.5 是此构造下的正确值。
    """
    # 50 题：weak 全错，strong 全对；win_rate 与「强对」完全一致
    strong_correct = np.ones(50, dtype=bool)
    weak_correct = np.zeros(50, dtype=bool)
    win_rates = np.linspace(0.0, 1.0, 50)

    res = cal.compute_apgr(win_rates, weak_correct, strong_correct)

    assert res["weak_accuracy"] == pytest.approx(0.0)
    assert res["strong_accuracy"] == pytest.approx(1.0)
    # 对角线积分 = 0.5
    assert res["apgr"] == pytest.approx(0.5, abs=0.02), (
        f"diagonal router under this construction should be ~0.5, got {res['apgr']}"
    )


def test_apgr_increases_with_router_informativeness(cal):
    """路由越有信息，APGR 越高 —— 用「有信息 vs 无信息」的相对关系校验。

    这里对比两种极端：win_rate 与强模型正确性完全相关（有信息）vs 完全独立
    （无信息）。

    关键结构事实：无信息路由（win_rate 独立于正确性）时，所有切点的准确率
    都等于同一个混合基线，APGR 曲线退化为从 (0, weak_acc) 到 (1, strong_acc)
    的对角线，其 APGR ≈ 0.5 —— 这是该公式的**基线值，不是 0**。
    「APGR=0」对应的是比随机更差的情形（例如切点方向接反）。
    因此判据是：有信息 > 随机基线 0.5，且随机路由落在 0.5 附近。
    """
    rng = np.random.default_rng(7)
    n = 6000
    weak_correct = rng.random(n) < 0.5
    strong_correct = rng.random(n) < 0.9

    # 有信息：win_rate 与 strong_correct 强相关
    informative = np.where(strong_correct, 0.75, 0.25) + rng.normal(0, 0.05, n)
    res_info = cal.compute_apgr(np.clip(informative, 0, 1),
                                weak_correct, strong_correct)

    # 无信息：win_rate 与两类正确性均独立 -> 应落在 0.5 基线附近
    uninformative = rng.random(n)
    res_rand = cal.compute_apgr(uninformative, weak_correct, strong_correct)

    assert res_rand["apgr"] == pytest.approx(0.5, abs=0.05), (
        f"uninformative router should sit at the 0.5 baseline, "
        f"got {res_rand['apgr']}"
    )
    assert res_info["apgr"] > res_rand["apgr"] + 0.05, (
        f"informative APGR {res_info['apgr']} should clearly beat "
        f"uninformative {res_rand['apgr']}"
    )


def test_bootstrap_ci_brackets_point_estimate(cal):
    """bootstrap 置信区间必须包住点估计。"""
    rng = np.random.default_rng(3)
    n = 2000
    win_rates = np.clip(rng.normal(0.45, 0.07, n), 0, 1)
    weak_correct = rng.random(n) < 0.5
    strong_correct = rng.random(n) < 0.9

    ci = cal.bootstrap_apgr_ci(win_rates, weak_correct, strong_correct,
                               n_boot=50, seed=0)
    assert ci["ci_low"] <= ci["apgr_point"] <= ci["ci_high"], ci
    assert ci["ci_high"] > ci["ci_low"]


def test_stability_report_quantifies_variance(cal):
    """稳定性评估应返回各轮增益及其标准差，且增益非负。

    回归防护：曾经用 compute_apgr(...,[th]) 单点计算，得到 -5 这类荒谬值。
    PGR 的正当范围是「>= 0」（0=等同全弱基线；>1 表示该子集收益超过
    全局基线差，是合法结果，见 stability_report docstring）。
    负值必然是算法错误。
    """
    rng = np.random.default_rng(5)
    n = 3000
    win_rates = np.clip(rng.normal(0.44, 0.07, n), 0, 1)
    weak_correct = rng.random(n) < 0.5
    strong_correct = rng.random(n) < 0.9

    rep = cal.stability_report(win_rates, weak_correct, strong_correct,
                               thresholds=[0.4, 0.45], rounds=4, seed=0)
    assert set(rep.keys()) == {0.4, 0.45}
    for th, s in rep.items():
        assert len(s["pgr_per_round"]) == 4
        assert s["pgr_std"] >= 0.0
        assert abs(s["pgr_mean"] - float(np.mean(s["pgr_per_round"]))) < 1e-9
        # PGR 不得为负：负值意味着比全弱还差（切点方向接反）
        assert s["pgr_mean"] >= 0.0, (
            f"threshold {th} PGR {s['pgr_mean']} < 0 — 单点计算错误"
        )
        assert s["pgr_min"] >= -1e-6, f"pgr_min {s['pgr_min']} < 0"


def test_stability_report_rejects_single_point_apgr(cal):
    """回归测试：单点阈值必须走「增益」路径，而不是丢给曲线积分函数。

    若实现把单点阈值传给 compute_apgr，AUC 会退化成两点梯形并产出
    -5 这类荒谬值。这里断言增益非负且曲线 APGR 在合理范围内。
    """
    rng = np.random.default_rng(11)
    n = 4000
    weak_correct = rng.random(n) < 0.5
    strong_correct = rng.random(n) < 0.9
    win_rates = np.clip(
        np.where(strong_correct, 0.7, 0.3) + rng.normal(0, 0.1, n), 0, 1
    )

    rep = cal.stability_report(win_rates, weak_correct, strong_correct,
                               thresholds=[0.5], rounds=3, seed=0)
    s = rep[0.5]
    # 曲线 APGR：细网格积分，应在 [0,1]
    assert 0.0 <= s["apgr_curve_mean"] <= 1.0, (
        f"curve APGR {s['apgr_curve_mean']} out of [0,1]"
    )
    # 单点增益：可为 0~1.x，但绝不能是荒谬负数
    assert s["pgr_mean"] >= 0.0, (
        f"single-point PGR {s['pgr_mean']} < 0 — 走了曲线积分路径"
    )
    # 有信息的路由，增益应明显高于 0
    assert s["pgr_mean"] > 0.1, f"informative router PGR too low: {s['pgr_mean']}"


def test_candidate_thresholds_cover_meaningful_range(cal):
    """候选阈值应覆盖有实用价值的走强比例区间，而非挤在一个点上。"""
    rng = np.random.default_rng(13)
    # 模拟真实 MMLU 的宽分布
    win_rates = np.clip(rng.normal(0.55, 0.18, 14000), 0, 1)

    cands = cal.candidate_thresholds(win_rates, n=7)
    assert len(cands) == 7
    # pct 递增 -> 阈值递减，因此返回数组是降序（见 docstring）
    assert list(cands) == sorted(cands, reverse=True), f"not descending: {cands}"

    pcts = [float(np.mean(win_rates >= c)) for c in cands]
    # 走强比例应跨越相当宽的范围（默认 2%~50%）
    assert max(pcts) - min(pcts) > 0.3, (
        f"candidate thresholds too clustered: strong_pct range "
        f"{min(pcts):.2%}~{max(pcts):.2%}"
    )
    assert max(pcts) <= 0.55, f"should not include very-high-cost points: {max(pcts)}"
    assert min(pcts) >= 0.0


def test_candidate_thresholds_narrow_distribution_still_distinct(cal):
    """窄分布下候选阈值仍须互相可区分（不塌缩成一个点）。"""
    rng = np.random.default_rng(17)
    win_rates = np.clip(rng.normal(0.45, 0.02, 5000), 0, 1)  # 极窄

    cands = cal.candidate_thresholds(win_rates, n=5)
    assert len(cands) == 5
    assert len(set(np.round(cands, 6))) == 5, f"collapsed: {cands}"
    # 每个候选都必须真分流
    for c in cands:
        pct = float(np.mean(win_rates >= c))
        assert 0.0 < pct < 1.0, f"threshold {c} is a dead band (pct={pct})"


# ---------------------------------------------------------------------------
# 论文对齐测试（RouteLLM ICLR 2025）
# ---------------------------------------------------------------------------

def test_compute_pgr_matches_paper_eq6(cal):
    """论文 Eq 6：PGR = (r(M_Rα) - r(M_w)) / (r(M_s) - r(M_w))。

    手算检验三个锚点：全弱 -> 0；全强 -> 1；中点 -> 0.5。
    """
    weak_acc, strong_acc = 0.60, 0.80
    assert cal.compute_pgr(weak_acc, strong_acc, 0.60) == pytest.approx(0.0)
    assert cal.compute_pgr(weak_acc, strong_acc, 0.80) == pytest.approx(1.0)
    assert cal.compute_pgr(weak_acc, strong_acc, 0.70) == pytest.approx(0.5)
    # 强弱相同 -> 分母 0 -> nan（不应抛异常）
    assert cal.compute_pgr(0.7, 0.7, 0.7) != cal.compute_pgr(0.7, 0.7, 0.7)


def test_cpt_is_min_strong_pct_reaching_target(cal):
    """CPT(x%) = 达到该 PGR 所需的最小走强比例（论文 3.2 节）。"""
    # 曲线：(走强 10%, PGR 0.2), (30%, 0.5), (60%, 0.9)
    curve = [(0.10, 0.2), (0.30, 0.5), (0.60, 0.9)]
    assert cal.compute_cpt(curve, 0.5) == pytest.approx(0.30)
    assert cal.compute_cpt(curve, 0.2) == pytest.approx(0.10)
    assert cal.compute_cpt(curve, 0.9) == pytest.approx(0.60)
    # 达不到 -> None
    assert cal.compute_cpt(curve, 0.95) is None


def test_apgr_uses_paper_eq8_cost_bins(cal):
    """APGR 必须走论文 Eq 8 的成本档位离散，而不是阈值网格 trapz。

    判据：结果应等于「等分 10 个成本档、每档反解阈值后取 PGR 均值」的手算值。
    """
    rng = np.random.default_rng(23)
    n = 5000
    weak_correct = rng.random(n) < 0.5
    strong_correct = rng.random(n) < 0.9
    win_rates = np.clip(
        np.where(strong_correct, 0.7, 0.3) + rng.normal(0, 0.1, n), 0, 1
    )

    res = cal.compute_apgr(win_rates, weak_correct, strong_correct)
    assert "cpt" in res and "CPT(50%)" in res["cpt"]

    # 手算 Eq 8（覆盖 [0%,100%] 闭区间，共 11 个等分点）
    weak_acc = float(np.mean(weak_correct))
    strong_acc = float(np.mean(strong_correct))
    total = 0.0
    n = len(win_rates)
    for i in range(0, 11):
        pct = 10 * i
        if pct <= 0:
            rs = np.zeros(n, dtype=bool)
        elif pct >= 100:
            rs = np.ones(n, dtype=bool)
        else:
            th = float(np.percentile(win_rates, 100 - pct))
            rs = win_rates >= th
        acc = float(np.mean(np.where(rs, strong_correct, weak_correct)))
        total += (acc - weak_acc) / (strong_acc - weak_acc)
    assert res["apgr"] == pytest.approx(total / 11, abs=1e-3), (
        f"APGR {res['apgr']} != paper Eq8 hand-calc {total / 11}"
    )


def test_apgr_baseline_is_half_for_random_router(cal):
    """论文基线：随机路由 APGR ≈ 0.5（Table 1-3 实测 0.493~0.500）。

    这是判断「路由是否有效」的基准线，必须与论文一致。
    """
    rng = np.random.default_rng(29)
    n = 20000
    weak_correct = rng.random(n) < 0.5
    strong_correct = rng.random(n) < 0.9
    win_rates = rng.random(n)  # 与正确性完全独立 = 随机路由

    res = cal.compute_apgr(win_rates, weak_correct, strong_correct)
    assert res["apgr"] == pytest.approx(0.5, abs=0.05), (
        f"random router APGR should be ~0.5 per paper, got {res['apgr']}"
    )


def test_apgr_thresholds_path_still_supported(cal):
    """向后兼容：显式传 thresholds 时走旧的阈值网格 + trapz 路径。"""
    rng = np.random.default_rng(31)
    n = 3000
    weak_correct = rng.random(n) < 0.5
    strong_correct = rng.random(n) < 0.9
    win_rates = np.clip(
        np.where(strong_correct, 0.7, 0.3) + rng.normal(0, 0.1, n), 0, 1
    )

    grid = np.arange(0.0, 1.001, 0.01)
    res = cal.compute_apgr(win_rates, weak_correct, strong_correct, grid)
    assert 0.0 <= res["apgr"] <= 1.0
    # 兼容路径下 router_auc 有值；Eq8 路径下为 None
    assert res["router_auc"] is not None
