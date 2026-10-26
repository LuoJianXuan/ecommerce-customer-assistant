"""ch09 一次性：给 Langfuse 栈的 MinIO 建 langfuse 桶（compose 已去掉 mc init，本脚本替代）。

用法：.venv/Scripts/python scripts/langfuse_minio_init.py
（凭据读 langfuse.env 的 MINIO_PASSWORD；桶已存在则幂等跳过。
  何时重跑：docker volume ls 里 langfuse_langfuse_minio_data 被删后重建栈时。）
"""
from pathlib import Path

from minio import Minio

ROOT = Path(__file__).resolve().parent.parent
env = {}
for line in (ROOT / "langfuse.env").read_text(encoding="utf-8").splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()

client = Minio("127.0.0.1:9100", access_key="minio", secret_key=env["MINIO_PASSWORD"], secure=False)
if client.bucket_exists("langfuse"):
    print("bucket langfuse 已存在，跳过")
else:
    client.make_bucket("langfuse")
    print("bucket langfuse 已创建")
