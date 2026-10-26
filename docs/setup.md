# 运行指南

以仓库根目录为工作目录。已完成 Windows 新虚拟环境、独立新数据卷的启动验收；模型文件复用了下载缓存，联网首次下载未单独验证，见[验收记录](validation.md)。

## 环境与本地配置

记录的开发环境为 Windows / Python 3.13.3。按[环境说明](environment.md)安装依赖。Linux/macOS 可将 `.\.venv\Scripts\python.exe` 换成 `.venv/bin/python`，其他平台尚未验收，不应使用 Windows 专用的约束文件。

复制 `.env.example` 为 `.env`，填写本地值，真实文件不进 Git。

| 配置 | 用途 | 说明 |
| --- | --- | --- |
| `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` | 模型接口 | 模板为 DeepSeek 兼容接口，密钥需填写 |
| `DB_HOST` / `DB_PORT` / `DB_USER` / `DB_PASSWORD` / `DB_NAME` | MySQL | 默认本机 3307，口令由本地配置提供 |
| `MILVUS_URI` | 向量库 | 默认 http://127.0.0.1:19530 |
| `RERANK_CANDIDATES` | 重排候选数 | 冻结评估为 6 |
| `EVIDENCE_CONF_THRESHOLD` | 知识分支证据闸 | 当前为 0.20，未采用诊断候选 0.05 |
| `MCP_LOGISTICS_URL` / `MCP_AFTERSALE_URL` | 工具服务 | 默认 8101、8102 的 `/mcp` |
| `HF_HOME` | 模型缓存 | `.hf_cache`，首次使用会尝试下载 |
| `LOCAL_MODEL_DTYPE` | 检索模型精度 | 代码默认为 `bfloat16`，平台不支持时可改 `float32` 后另行验证 |
| `LANGFUSE_ENABLED` | 链路观测 | 公开模板默认关闭 |
| `MINE_INTERVAL_HOURS` / `EVAL_DAILY_HOUR` | 后台挖掘与评估 | 公开模板默认关闭，设置正值可启用 |
| `TOPIC_MODEL_PATH` / `TOPIC_AUTO_BATCH_ENABLED` | 主题分类 | 需另行训练权重，自动攒批默认关闭 |

默认模型经配置的镜像下载；可在启动前设置 `HF_ENDPOINT=https://huggingface.co` 使用官方托管地址。首次下载和预热需要时间，服务启动不等于模型已经就绪。

## 数据服务与初始化

```powershell
docker compose up -d
docker compose -f docker-compose.milvus.yml up -d
docker compose ps
docker compose -f docker-compose.milvus.yml ps
# 等待容器健康后
.\.venv\Scripts\python.exe scripts/init_db.py --plan --seed-demo-faq
.\.venv\Scripts\python.exe scripts/init_db.py --seed-demo-faq
```

编排使用独立的 `ecommerce-assistant` 项目名称和数据卷。若与其他本地项目同时运行，可在 `.env` 中设置 `DB_PORT`、`MILVUS_PORT`、`MILVUS_HEALTH_PORT`；修改 `MILVUS_PORT` 时还需同步设置 `MILVUS_URI`。本次隔离验收使用 3317、19531、9092，避免复用原有服务。

初始化按基础、ch03、ch04、ch06、ch07、ch08、ch09、ch10 的顺序创建表，补齐会话摘要锚点列；FAQ 表为空时才导入演示数据。它不删除表、不重置数据，重复运行保留已有表。

该入口面向新演示库和已知结构的重复初始化，不是通用数据库迁移系统；遇到旧表缺列或约束差异需先核对结构。

## 准备演示知识

推荐加载当次评估的 39 条公开知识，不带入历史隔离材料：

```powershell
.\.venv\Scripts\python.exe scripts/load_eval_knowledge.py --plan
.\.venv\Scripts\python.exe scripts/load_eval_knowledge.py
```

它要求知识表和目标向量集合均为空，避免与已有知识混用。导入后设为 pending（待向量化），成功写入 Milvus 后更新状态。若向量化中断：

```powershell
.\.venv\Scripts\python.exe -m app.knowledge.build --resume
```

也可从 Markdown 文档建库，再导入 FAQ。以下是另一种准备路径，不必与快照导入连续执行：

```powershell
.\.venv\Scripts\python.exe -m app.knowledge.build
.\.venv\Scripts\python.exe -c "from app.knowledge.faq_import import import_faq; import_faq()"
```

重新切分、修改业务规则或更换模型后，不能直接沿用原评估数字。

## 启动工具与应用

三个终端均使用同一个虚拟环境，保持两个 MCP 服务运行：

```powershell
# 终端 1
.\.venv\Scripts\python.exe mcp_servers/logistics_server.py
# 终端 2
.\.venv\Scripts\python.exe mcp_servers/aftersale_server.py
# 终端 3
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

先打开 `/docs` 查看接口，再到 `/kb` 检查知识和检索，最后在 `/` 测试对话。主题页需要另行准备分类器权重，详见[训练说明](training.md)。

## 可选 Langfuse

复制 `langfuse.env.example` 为 `langfuse.env`，分别生成、填写密码与加密字段：

```powershell
python -c "import secrets; print(secrets.token_hex(32))"
docker compose --env-file langfuse.env -f docker-compose.langfuse.yml -p ecommerce-assistant-langfuse up -d
.\.venv\Scripts\python.exe scripts/langfuse_minio_init.py
```

在 http://127.0.0.1:3000 创建本地项目，将公钥、私钥填入 `.env`，设置 `LANGFUSE_ENABLED=true` 后重启应用。真实配置和追踪导出保持在本地。

## 测试与停止

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -m "not integration" -q
# 数据库已经初始化时，单独运行标记的数据库测试
.\.venv\Scripts\python.exe -m pytest tests/ -m integration -q
docker compose stop
docker compose -f docker-compose.milvus.yml stop
```

本次测试在完成演示知识导入后运行，包含依赖 MySQL 与检索模型的接口测试；没有旧的挖掘历史也可以运行。测试不是生产负载或显卡训练验收。评估脚本可能需要模型接口和真实服务，运行方式见[评估说明](evaluation.md)。
