# 代码、数据与第三方来源

## 本项目

教程仅提供需求和设计文档，项目代码按模块自行开发生成，没有以教程源码作为代码基础。MewHelp 是学习过程中的项目称呼；当前仓库使用 Intelligent E-commerce Customer Assistant 名称。

业务知识、示例商品与政策用于本地演示；训练数据包含合成问句、人工标签与定向补丁。原始运行配置、采集记录、日志和隔离的私人材料不进入公开版本。公开回归样例不代表生产用户规模。

本项目尚未指定开源许可证。这里没有擅自选择 MIT 或 Apache-2.0，也没有将第三方组件及模型权利归为本项目所有。

## 预训练模型

| 来源 | 用途 | 模型卡标注的许可 |
| --- | --- | --- |
| [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | 文本嵌入 | MIT |
| [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | 检索重排 | Apache-2.0 |
| [hfl/chinese-roberta-wwm-ext](https://huggingface.co/hfl/chinese-roberta-wwm-ext) | 主题分类基础模型 | Apache-2.0 |

模型卡信息于本次整理时核对。仓库只包含接入与训练代码，不附这些预训练权重；获取、使用或分发模型应遵循对应模型的条款。

## 组件与文档参考

- [FastAPI](https://fastapi.tiangolo.com/)：接口与应用服务。
- [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) 与 [LangChain](https://docs.langchain.com/oss/python/langchain/overview)：工作流、模型与工具接入。
- [Milvus](https://milvus.io/docs)：向量与关键词混合检索。
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)：工具服务。
- [Langfuse](https://langfuse.com/docs)：可选链路观测。
- [PyTorch](https://pytorch.org/get-started/locally/) 与 [Transformers](https://huggingface.co/docs/transformers/)：本地模型推理和训练。

各组件继续适用各自许可证；依赖范围见 `requirements.txt`，本次通过验收的完整安装版本见 `constraints/windows-py313-validated.txt`，历史 CUDA 环境记录另见 `constraints/windows-cuda130-observed.txt`。

README 借鉴 [AI-oral-exam](https://github.com/xyp-carry/AI-oral-exam) 的“模块、目标、具体能力”说明方式，没有将其预期功能或运行配置写成本项目的实现。
