# 环境与依赖

## 本次通过验收的环境

2026-10-26，在 Windows x64 / Python 3.13.3 的新虚拟环境中安装依赖，使用 CPU（处理器）版 PyTorch 2.14.1 完成测试与真实接口验收。完整固定版本见 [windows-py313-validated.txt](../constraints/windows-py313-validated.txt)。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -c constraints/windows-py313-validated.txt
```

本次实际通过 uv（Python 包管理工具）按相同依赖与版本安装；pip 检查没有依赖冲突。已安装 uv 时也可使用：

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements.txt -c constraints/windows-py313-validated.txt
```

验收时使用 HTTPS 包镜像，遇到本机代理不可用可为单次命令配置可用网络或可信索引，不需要改项目业务代码。不要把认证信息写进公开依赖文件。

MCP（模型上下文协议）SDK 2.x 与当前适配器不兼容，本次实际验证的组合是 `mcp==1.30.0` 与 `langchain-mcp-adapters==0.3.1`；依赖清单已限制兼容范围，完整约束进一步固定本次安装版本。

该约束文件包含 Windows 专用依赖。其他平台使用基础依赖清单，尚未完成跨平台验收。

## 模型与资源

嵌入、重排和主题分类实际执行过推理；预训练模型与分类权重复用已有文件，没有在本次验收中重新训练，也没有单独验证所有模型的首次联网下载。

已有缓存可在启动进程前设置 `HF_HOME` 环境变量，确保它在模型库导入之前生效。新环境默认使用项目内 `.hf_cache`。CPU 运行只证明本次功能通过，响应时间受硬件与精度影响，本次没有负载或延迟指标承诺。

## CUDA 历史环境

原开发环境记录为 PyTorch 2.14.0+cu130；`constraints/windows-cuda130-observed.txt` 记录直接/关键依赖，属于历史观察快照，不是本次新装验收的版本。不要与 CPU 验证约束叠加使用。

如需 CUDA（显卡计算平台），按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)安装对应构建，再单独验证硬件、推理与训练。当前没有对 CUDA 训练作新的验收承诺。

## 配置

公开模板保留重排候选数 6、证据阈值 0.20；首次运行关闭后台挖掘、定时评估和 Langfuse（链路观测平台）。模型密钥和数据库口令需要自行填写。本次验收使用独立端口及新数据卷，没有复用旧业务库。

固定评估的软件版本保存在 `evals/releases/2026-10-26/run_config.json`。新安装验收与固定性能评估是两套记录，见[评估说明](evaluation.md)及[验收记录](validation.md)。
