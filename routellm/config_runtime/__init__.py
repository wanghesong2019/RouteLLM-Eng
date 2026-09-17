"""运行时配置（热更新）。

方案文档 4.8 节 —— 让下游强弱模型的 base_url / api_key / 模型名可
**不重启网关**即可编辑并立即生效。

模块：
    store.py   RuntimeConfigStore —— 不可变配置对象 + 原子替换 + JSON 持久化
    api.py     GET/PUT /api/config —— 编辑接口（掩码回显 + 鉴权）

用法：
    from routellm.config_runtime import RuntimeConfigStore, CONFIG_STORE

    cfg = CONFIG_STORE.load()        # 请求路径：无锁 O(1)
    ... cfg.api_base, cfg.api_key, cfg.strong_model

注：本包**不做热切导入重模块**，各子模块按路径显式导入（与 monitoring
包同样的约定，保证轻量消费者不被拖入网关重依赖）。
"""

__all__ = ["store", "api"]
