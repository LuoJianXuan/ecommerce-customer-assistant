-- ch08 工具系统：工具调用审计表（不挂外键；写审计失败不许反拦工具执行）
CREATE TABLE IF NOT EXISTS tool_audit_logs (
    id              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    conversation_id BIGINT UNSIGNED NULL,
    tool_call_id    VARCHAR(64) NULL,
    tool_name       VARCHAR(128) NOT NULL,
    source          VARCHAR(64) NOT NULL,
    params          JSON NULL,
    result_summary  VARCHAR(500) NULL,
    status          ENUM('成功','失败','超时','校验拦下','权限拒绝') NOT NULL,
    error           VARCHAR(500) NULL,
    retry_count     INT NOT NULL DEFAULT 0,
    duration_ms     INT NOT NULL DEFAULT 0,
    created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_audit_conv (conversation_id, created_at),
    INDEX idx_audit_tool (tool_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
