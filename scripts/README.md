# scripts/

辅助脚本。**不属于 `routellm` 包，不参与打包。**

## 清单

| 脚本 | 用途 |
|---|---|
| `eval_router_apgr.py` | 按论文指标（APGR）评测路由在 MMLU / GSM8K 上的表现 |
| `calibrate_threshold.py` | 阈值标定：win-rate 分布 + CPT 表 + bootstrap 置信区间 |
| `check_open_source_hygiene.py` | 开源卫生守卫：检查凭据 / 内网指纹 / 部署留档产物 |

## 开源卫生守卫

提交前运行，确认仓库不含不应公开的内容：

```bash
python scripts/check_open_source_hygiene.py        # 扫工作区
python scripts/check_open_source_hygiene.py --all  # 同时扫 git 历史
python scripts/check_open_source_hygiene.py --quiet
```

检查项：凭据（`sk-*` / `hf_*` / Bearer / 硬编码赋值）、环境指纹（内网 IP、
内部主机名、私有域名）、部署留档产物、`.gitignore` 护栏。

退出码 0 = 通过，1 = 发现违规。脚本只读，不修改任何文件。

## 评测与标定

需要先启动推理服务（见 `services/`）。

```bash
# 论文指标 APGR（复现路径）
python scripts/eval_router_apgr.py \
    --bert-url http://127.0.0.1:6070 \
    --mmlu-dir routellm/evals/mmlu/responses \
    --gsm8k-csv routellm/evals/gsm8k/gsm8k_responses.csv \
    --out-dir results/apgr

# 阈值标定（CPT 框架）
python scripts/calibrate_threshold.py \
    --bert-url http://127.0.0.1:6070 \
    --mmlu-dir routellm/evals/mmlu/responses \
    --gsm8k-csv routellm/evals/gsm8k/gsm8k_responses.csv \
    --out-dir results/calibration
```

标定脚本的核心结论：**APGR 不能用来选阈值**（PGR 随走强比例单调递增，
只最大化 APGR 的答案永远是「全走强」）。选阈值必须用 **CPT** ——
先定质量目标，再反解最小走强比例。

依赖：`numpy`；`eval_router_apgr.py` 与 `calibrate_threshold.py` 通过 HTTP
调用推理服务，本地不需要 torch。
