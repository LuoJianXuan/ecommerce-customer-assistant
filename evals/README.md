# 评估导航

公开结果的单一索引是 [current.json](current.json)。

| 类型 | 固定目录 | 口径 |
| --- | --- | --- |
| 当前 RAG 审计 | [2026-10-26](releases/2026-10-26/report.md) | 100 题；73 条有参考证据，27 条库外 |
| 当前主题分类 | [2026-10-23-topic](releases/2026-10-23-topic/report.md) | 固定 510 条测试集，阈值 0.40 |

```powershell
python evals/verify_release.py
```

以上只核验已有文件与指标，不访问模型接口。完整重跑使用 `run_audit.py`，输出到新的 `tmp/` 目录；无法解析的模型裁判可用 `retry_judges.py` 仅重试裁判。详细命令、标签修订和限制见[评估说明](../docs/evaluation.md)。

章节目录保留历史实验和工作台入口。它们可能使用旧标签、旧配置和限样设置；`ch04/report.*`、`ch09/*` 以及 `/eval` 不能冒充固定日期的最新审计。`ch10/` 是训练评估脚本的可变输出目录，公开分类指标另有冻结副本。
