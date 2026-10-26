-- ch07 上下文管理：会话摘要表 + conversations 双锚点列
-- 约定：应用不建表，结构以本文件为准（对齐 schema*.sql 风格）。
-- 注意：锚点=messages 表 user 行 id；只挪数字不搬数据。

ALTER TABLE conversations
    ADD COLUMN summary_upto_msg_id BIGINT UNSIGNED NULL,
    ADD COLUMN layer1_from_msg_id BIGINT UNSIGNED NULL;

CREATE TABLE IF NOT EXISTS conversation_summaries (
    id              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    conversation_id BIGINT UNSIGNED NOT NULL,
    seq             INT NOT NULL,
    upto_msg_id     BIGINT UNSIGNED NOT NULL,
    content         TEXT NOT NULL,
    created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_conv_seq (conversation_id, seq),
    INDEX idx_sum_conv (conversation_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
