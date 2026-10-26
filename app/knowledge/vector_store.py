"""Milvus 向量库封装（ch04 升级：正文进 Milvus + 原生 BM25 全文检索 + hybrid RRF）。

集合 schema（文档核实：milvus.io v2.5.x full-text-search）：
- vector_id  VARCHAR(64) 主键（MySQL 侧确定性生成 "kb-{chunk_id}"，按主键幂等）
- chunk_id   INT64 标量
- text       VARCHAR(8192)：三格拼接正文，chinese analyzer（内置 jieba，建表定死不可改）
- category   VARCHAR(255)：检索侧元数据过滤字段
- dense      FLOAT_VECTOR 1024（BGE-M3，COSINE）
- sparse     SPARSE_FLOAT_VECTOR：由 BM25 Function 从 text 自动生成（插入只传 text）
索引：dense AUTOINDEX/COSINE；sparse SPARSE_INVERTED_INDEX/BM25。

检索三路：
- search_dense / search_bm25：单路（评估消融用）
- hybrid_search：AnnSearchRequest×2 + RRFRanker（默认 k=60）
检索直出 text+category；结构化三格/溯源字段由调用方拿 chunk_id 回 MySQL 补全。
"""

from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
)

from app.config import get_settings

VECTOR_DIM = 1024
TEXT_MAX_LEN = 8192


class MilvusKnowledgeStore:
    def __init__(self, uri: str | None = None, collection: str | None = None) -> None:
        settings = get_settings()
        self.uri = uri or settings.milvus_uri
        self.collection = collection or settings.milvus_collection
        self.client = MilvusClient(uri=self.uri)

    def ensure_collection(self) -> None:
        if self.client.has_collection(self.collection):
            return
        self.create_collection()

    def create_collection(self) -> None:
        """建集合（含 BM25 函数与 chinese analyzer——建表定死，改 analyzer 需重建）。"""
        schema = self.client.create_schema(auto_id=False)
        schema.add_field("vector_id", DataType.VARCHAR, max_length=64, is_primary=True)
        schema.add_field("chunk_id", DataType.INT64)
        schema.add_field(
            "text",
            DataType.VARCHAR,
            max_length=TEXT_MAX_LEN,
            enable_analyzer=True,
            enable_match=True,
            analyzer_params={"type": "chinese"},
        )
        schema.add_field("category", DataType.VARCHAR, max_length=255)
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(
            Function(
                name="text_bm25",
                input_field_names=["text"],
                output_field_names=["sparse"],
                function_type=FunctionType.BM25,
            )
        )
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense", index_type="AUTOINDEX", metric_type="COSINE"
        )
        index_params.add_index(
            field_name="sparse",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )
        self.client.create_collection(
            self.collection, schema=schema, index_params=index_params
        )

    def recreate(self) -> None:
        """删除重建（schema 升级用；MySQL 权威源在，重灌即恢复）。"""
        if self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)
        self.create_collection()

    def upsert(self, vector_id: str, chunk_id: int, text: str, category: str, dense: list[float]) -> None:
        self.client.upsert(
            self.collection,
            [
                {
                    "vector_id": vector_id,
                    "chunk_id": chunk_id,
                    "text": text,
                    "category": category,
                    "dense": dense,
                }
            ],
        )

    def delete_by_chunk_ids(self, chunk_ids: list[int]) -> None:
        if not chunk_ids:
            return
        self.client.delete(self.collection, filter=f"chunk_id in {chunk_ids}")

    # ---------- 检索 ----------

    def _out_fields(self) -> list[str]:
        return ["chunk_id", "text", "category"]

    @staticmethod
    def _hits(results: list) -> list[dict]:
        out = []
        for hit in results[0]:
            entity = hit.get("entity", {}) if isinstance(hit, dict) else {}
            out.append(
                {
                    "chunk_id": entity.get("chunk_id"),
                    "text": entity.get("text", ""),
                    "category": entity.get("category", ""),
                    "score": hit.get("distance"),
                }
            )
        return out

    def search_dense(self, query_vector: list[float], top_k: int, expr: str = "") -> list[dict]:
        return self._hits(
            self.client.search(
                self.collection,
                data=[query_vector],
                anns_field="dense",
                search_params={"metric_type": "COSINE"},
                limit=top_k,
                filter=expr or None,
                output_fields=self._out_fields(),
            )
        )

    def search_bm25(self, query_text: str, top_k: int, expr: str = "") -> list[dict]:
        """BM25 查询直传文本（sparse 由服务端分词生成）。"""
        return self._hits(
            self.client.search(
                self.collection,
                data=[query_text],
                anns_field="sparse",
                search_params={"metric_type": "BM25"},
                limit=top_k,
                filter=expr or None,
                output_fields=self._out_fields(),
            )
        )

    def hybrid_search(
        self,
        query_vector: list[float],
        query_text: str,
        recall: int = 50,
        limit: int = 10,
        expr: str = "",
    ) -> list[dict]:
        """dense + BM25 两路各 recall 条，RRF 融合取 limit。"""
        dense_req = AnnSearchRequest(
            data=[query_vector],
            anns_field="dense",
            param={"metric_type": "COSINE"},
            limit=recall,
            expr=expr or None,
        )
        sparse_req = AnnSearchRequest(
            data=[query_text],
            anns_field="sparse",
            param={"metric_type": "BM25"},
            limit=recall,
            expr=expr or None,
        )
        results = self.client.hybrid_search(
            self.collection,
            [dense_req, sparse_req],
            ranker=RRFRanker(),
            limit=limit,
            output_fields=self._out_fields(),
        )
        return self._hits(results)
