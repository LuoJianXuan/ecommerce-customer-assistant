"""应用配置：单一激活配置，全部来自 .env（pydantic-settings）。

HF 环境变量必须在任何 huggingface_hub 导入**之前**就位——它在校验时把
ENDPOINT/OFFLINE 固化成模块常量，后设无效（ch05 实测：切分器先导入时
模型加载会绕过镜像直连 huggingface.co 重试到挂死）。所以放在模块级。
"""

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(_PROJECT_ROOT / ".hf_cache"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 模型已全量缓存；缺失时 Embedder 会临时关掉回退下载
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from functools import lru_cache  # noqa: E402

from pydantic_settings import BaseSettings, SettingsConfigDict  # noqa: E402


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # 模型接入三件套：换供应商只改这三行
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-flash"
    llm_temperature: float = 0.7
    # DeepSeek 新模型默认开启思考模式；当前 Agent/结构化输出依赖强制 tool_choice，
    # 因此默认关闭。需要纯文本思考任务时可在独立模型实例中再开启。
    llm_thinking_enabled: bool = False

    # 多轮历史 token 预算（发送窗口上限）
    history_token_budget: int = 2000

    # System Prompt 模板变量：客服所属平台名
    platform_name: str = "云优选"

    # 数据库（Docker MySQL，compose 默认 3307:3306）
    db_host: str = "127.0.0.1"
    db_port: int = 3307
    db_user: str = "root"
    db_password: str = ""  # 通过本地 .env 配置，源码不保存数据库口令
    db_name: str = "kefu"

    # 工具执行
    tool_timeout_seconds: float = 5.0
    tool_max_retries: int = 1

    # 单次模型调用超时（秒）：路由调用与收敛调用各自限时，防上游悬死
    llm_call_timeout_seconds: float = 60.0

    # ch03 知识库
    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_collection: str = "knowledge"
    embed_model_name: str = "BAAI/bge-m3"
    # 两个 0.6B 本地模型同时常驻时，BF16 可将权重内存约减半；CPU 推理已验证可用。
    local_model_dtype: str = "bfloat16"
    # 模型缓存落 D 盘（C 盘红线；相对项目目录）
    hf_home: str = ".hf_cache"
    retrieval_top_k: int = 3
    # 防御性过滤：即使 Milvus 中残留旧向量，也不让未审核对话挖掘块进入召回。
    retrieval_excluded_categories: str = "对话挖掘"
    # 挖知识定时任务（小时，<=0 关闭）
    mine_interval_hours: float = 6.0

    # ch04 检索策略：dense | bm25 | hybrid | hybrid_rerank
    retrieval_strategy: str = "hybrid_rerank"
    # 重排模型（点名选型）
    rerank_model_name: str = "BAAI/bge-reranker-v2-m3"
    # 混合检索：两路各召回多少再 RRF 融合
    recall_top_k: int = 50
    # 重排：融合结果取 Top-N 进 reranker，精排留 Top-K
    rerank_candidates: int = 20
    rerank_top_k: int = 10
    # Query 理解开关（口语改写+同义词扩展，检索侧）
    query_rewrite_enabled: bool = True
    # query_faq 工具超时（改写+检索+重排的总预算，比通用工具宽）
    faq_tool_timeout_seconds: float = 12.0

    # ch05 Agent 工作流
    evidence_conf_threshold: float = 0.2  # ch09 100 题双侧 F1 校准结果
    agent_max_steps: int = 5            # ReAct 步数上限（停止条件）

    # ---------- ch09 Langfuse 观测 ----------
    langfuse_enabled: bool = True        # 关掉=零开销零追踪（handler 不创建）
    langfuse_host: str = "http://127.0.0.1:3000"
    langfuse_public_key: str = ""        # 空值=视为未配置，不追踪
    langfuse_secret_key: str = ""
    eval_gen_limit: int = 30             # 每日评估生成段限样；0=全量
    eval_daily_hour: int = 3             # Asia/Shanghai 小时；<=0 关闭定时

    # ch06 分流器（意图识别/指代消解/降级链路）
    intent_model: str = ""                   # 意图/消解专用模型；空=沿用 llm_model
    intent_small_model: str = "deepseek-flash"  # 降级链路小模型（先行判断；官方仅两档模型）
    intent_escalation_enabled: bool = False  # 两级降级开关：开=小模型先行，低置信升级大模型重判
    intent_confidence_threshold: float = 0.6 # 小模型置信度低于此值触发升级

    # ch07 上下文管理（预算从模型窗口倒推，全可调；演示配置见 dev-notes/ch07-spec.md §2）
    model_context_window: int = 32768   # 模型上下文窗口（换模型只改这一个，预算表整体重算）
    max_output_tokens: int = 2000       # 输出预留
    max_user_input_tokens: int = 2000   # 单条用户输入预留
    tool_result_max_tokens: int = 1200  # 单条工具结果上限（层2 截短/峰值估算共用口径）
    ctx_system_prompt_tokens: int = 700 # 系统提示固定开销估算（预算侧常量，保证校准数稳定）
    ctx_want_rounds: int = 10           # 想留住的轮数（轮数侧 = 想留轮数×每轮稳态）
    ctx_steady_per_round: int = 565     # 每轮稳态占用（保守供给均值）
    ctx_safety_ratio: float = 0.10      # 安全余量比例（×窗口）
    ctx_summary_inject_segments: int = 3      # 注入最近几段梗概（其余留库不进上下文）
    ctx_summary_segment_max_tokens: int = 200 # 单段梗概上限（也是梗概预留的计费单位）
    ctx_evidence_per_chunk_tokens: int = 300  # 每条检索证据的预算估算
    ctx_assistant_head_chars: int = 80  # 层2 客服答复保留的前 N 字
    log_dir: str = "log"              # 文件日志目录（演示服务可单独指走,互不污染 grep 验收）

    # ch08 工具系统（MCP 双 Server,Streamable HTTP）
    mcp_logistics_url: str = "http://127.0.0.1:8101/mcp"
    mcp_aftersale_url: str = "http://127.0.0.1:8102/mcp"

    # ch10 主题分类（离线旁路，不进入实时客服主链路）
    topic_model_path: str = "training/model_out/best"
    topic_device: str = "auto"             # auto | cuda | cpu
    topic_batch_size: int = 64
    topic_threshold_fallback: float = 0.5    # 模型目录无阈值文件时才使用
    topic_auto_batch_enabled: bool = False   # 用户裁定：自动攒批默认关闭
    topic_auto_batch_min_pending: int = 100
    topic_auto_batch_limit: int = 500
    topic_auto_batch_interval_minutes: float = 10.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
