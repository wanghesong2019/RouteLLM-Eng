# RouteLLM-Eng 交接状态（2026-09-17 17:35）

> 会话中断后从这里接上。稳定的环境事实已写入长期记忆，此处只记**进度与待办**。

## 一句话现状

Phase 3.5/3.6 全部完成并推送；网关已通过 SSH 隧道接入 Hermes（保活就绪）；
下一件事是**阈值标定实验**（已登记进方案文档，安排在收尾时做）。

---

## 一、代码仓库状态

### RouteLLM-Eng（33: /data/home/hermes/srv/projects/RouteLLM-Eng）

    远端   ssh://git@172.17.17.50:2424/large-model-security-governance/RouteLLM-Eng.git
    分支   main（单分支，无 dev）
    HEAD   6273901  feat(config): 强弱分侧配置 + 原始模型名 + 分侧连通性预检（Phase 3.6）
    状态   已推送、工作区干净

    最近提交
      6273901  Phase 3.6（本次）
      37164cc  Phase 3.5 运行时配置热更新
      119c66f  网关鉴权 + Dashboard 独立容器 + compose
      d199f44  监控体系 + Dashboard
      453bb38  多级缓存层

    测试   238 passed, 17 skipped（33 本机 .venv）
    守卫   scripts/check_open_source_hygiene.py 报 21 项，全在 docs/experiments/
           （既有环境指纹，本次新增文件零告警）

    43 部署副本  /mnt/data/wanghesong/routellm/RouteLLM-Eng/（手工 scp 维护，非 checkout）
                 本次已 rsync 同步 routellm/ tests/ scripts/
                 容器已重建（镜像含 Phase 3.6 代码）

### jobfinding（33: /data/home/hermes/projects/jobfinding）

    方案文档  projects/RouteLLM-优化改造方案.md
    HEAD      664da86  docs(RouteLLM): 登记阈值标定实验（4.10 节）
    状态      已推送

    本次在方案文档新增：
      4.9 节   强弱分侧配置 + 分侧预检（Phase 3.6，7 个小节）
      4.10 节  阈值标定实验（⏳ 待做，收尾时执行）
      功能列表/Phase 表/实施计划表（总计 12.2→14.0 天）/技术亮点清单/面试叙事模板
              均已同步更新

---

## 二、运行环境状态

### 43 网关（compose 双容器）

    routellm-deploy     0.0.0.0:6060  （需 Bearer key）
    routellm-dashboard  0.0.0.0:8092 → 容器 8080（免鉴权）
    状态                两者 healthy
    热更新配置          /data/runtime_config.json（600）

    当前生效配置（用户 17:2x 在前端设置）
      strong_model = openai/zai-org/GLM-5.2
      weak_model   = openai/Qwen/Qwen3.5-4B      ← 用户最后改的
      api_base     = https://api.siliconflow.cn/v1
      分侧 base/key 均填了硅基流动

    注：模型名带了 openai/ 前缀仍可用（白名单识别已带前缀则原样保留），
        但按新设计只需填原始名（zai-org/GLM-5.2），前缀会自动补。

### SSH 隧道（33 本地 → 43:6060）

    脚本     /data/home/hermes/scripts/tunnel_routellm.sh
             （start|stop|restart|status|ensure）
    保活     用户 crontab：每分钟 `ensure`
    本地端口 16060
    日志     /tmp/routellm_tunnel.log（带 1MB 轮转 + 连续失败告警）
    当前     UP，PID 记录在 /tmp/routellm_tunnel.pid
    验证要点 PPID=1 且 SID==PID 才算真脱离 Hermes 会话

    日志防护（本次新增）
      - 正常运行时零写入（仅异常才写）
      - 超 1MB 自动轮转成 .1，只留最近一份
      - 连续 3 次拉起失败 → 写 ***ALERT*** 一条（只在跨阈值时写，防刷屏）
      - status 显示日志大小 + 连续失败次数

### Hermes 配置（profile agent-engineer）

    custom_providers 新增（未改主模型，主模型仍是 deepseek）：
      name     : routellm-gateway
      base_url : http://127.0.0.1:16060/v1
      api_key  : <网关自身的 ROUTELLM_GATEWAY_API_KEY，55 字符>
      model    : router-remote_bert-0.5
      api_mode : chat_completions

    备份     config.yaml.pre-routellm

---

## 三、已完成的验证（可复述的证据）

    ① 强弱分侧配置
       PUT 分侧 base_url → 回显各自独立；弱侧未配则回落顶层
    ② 原始模型名
       PUT test-org/Test-Model-Raw → 回显无前缀；调用时补 openai/
       容器内实测：deepseek-ai/DeepSeek-V4-Pro → openai/deepseek-ai/DeepSeek-V4-Pro
    ③ 分侧测试连接
       verify?tier=strong/weak 各测各的，响应回显 model/api_base/source
       非法 tier → HTTP 422（修复前被面板包成 200）
    ④ 单模型兜底
       只配一个模型 → effective_strong == effective_weak（非空），不报错
    ⑤ 热更新对客户端立即生效（用户实测）
       弱模型 deepseek-ai/DeepSeek-V4-Flash → 改 Qwen/Qwen3.5-4B
       同一请求输出随之改变；网关容器 StartedAt 未变（59 分钟无重启）
       Hermes 侧 model 字段全程未改
    ⑥ 隧道保活
       kill -9 → ensure 自愈 → 新 pid → /health 200
    ⑦ 孤儿端口回收（修复的 bug）
       造孤儿占 16060 + pidfile 指向死进程 → ensure 识别并回收 → UP
    ⑧ 日志防护
       1.1MB → 轮转 .1；正常时 ensure 两次日志字节数不变；失败计数 2/3/5 场景正确

---

## 四、待办

### 下一步：阈值标定实验（已登记，收尾时做）

见方案文档 **4.10 节**。要点：

    背景   实测 14 条请求 win_rate 集中 0.297~0.539，无样本 ≥0.55
           → 阈值 ≥0.55 等于"永远走弱"
           当前 0.5 下走强 4/14（偏弱）
    方向   阈值调高→更多走弱；调低→更多走强（与"偏保守就调高"的直觉相反）
    方法   用真实评测集（MMLU/GSM8K，见 scripts/eval_router_apgr.py）
           按分位数标定 + APGR 验证
    技巧   阈值是请求级参数（router-remote_bert-<t>），可零风险对比

### 其他未完成的（方案文档既有清单）

    Phase 3   容错 + 异步化（熔断器、退避重试；路由器 async 化）  未做
    Phase 4   Hermes 适配（已部分完成：/v1/models 已补）         部分
    Phase 5   MT-Bench 评测                                      未做

### 可选的小事

    - fullstack-docker-workflow skill 的 Pitfall 编号有重复
      （多个 13/14/15/16/17，Pitfall 33 出现两处）—— 纯可读性问题
    - 43 上网关若停掉，隧道仍在但请求会失败（缺"远端服务不可达"的监控）
    - 用户最后把弱模型改成了 Qwen/Qwen3.5-4B（是否改回 DeepSeek-V4-Flash 待定）

---

## 五、技能沉淀现状（本轮已核对，无需重复写入）

    经验已完整存在，勿重复添加：

    fullstack-docker-workflow
      Pitfall 34  转发层吞 4xx → UI 分不清成功与拒绝
      Pitfall 35  容器内 127.0.0.1 是自己；兄弟服务用 service name
      Pitfall 36  网络不通时在已部署镜像内验证核心逻辑

    codebase-onboarding/references/gateway-auth-dashboard-and-hot-config.md
      §5b 分侧凭据+全局兜底  §5c 单模型兜底  §5d 原始名与前缀
      §5e 分侧预检  §5f 既有测试语义变更  §6b 手工变异验证  §9 「兜底」歧义

    remote-ssh-access/references/durable-tunnel-without-sudo.md
      §1 环境探测  §2 守护脚本  §2a 日志轮转  §2b is_running 不可用可变配置
      §3 cron  §4 验证脱离  §5 故障测试  §5a 测试别变成事故  §6 接入 Hermes

    codebase-onboarding/references/deploying-a-change-into-a-live-container.md
      §8 部署机解释器≠测试环境（rag-dev 无 litellm）
