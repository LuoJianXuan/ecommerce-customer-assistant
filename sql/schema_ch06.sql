-- ch06 订单域：orders（订单选择器数据源，懒播种）+ refund_requests（退款表单落库）
-- 约定：应用不建表，表结构以本文件为准（对齐 schema.sql 风格）。

CREATE TABLE IF NOT EXISTS orders (
    id           BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    order_no     VARCHAR(32)  NOT NULL,
    user_id      VARCHAR(64)  NOT NULL,
    product_name VARCHAR(128) NOT NULL,
    amount       DECIMAL(10,2) NOT NULL,
    status       ENUM('待付款','已付款','已发货','已签收','已取消') NOT NULL,
    created_at   DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at   DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_orders_order_no (order_no),
    INDEX idx_orders_user (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS refund_requests (
    id          BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    refund_no   VARCHAR(32) NOT NULL,
    order_id    BIGINT UNSIGNED NOT NULL,
    user_id     VARCHAR(64) NOT NULL,
    reason      ENUM('七天无理由','质量问题','与描述不符','少件漏发','不想要了','其他') NOT NULL,
    status      ENUM('待审核','已通过','已驳回') NOT NULL DEFAULT '待审核',
    created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_refund_no (refund_no),
    CONSTRAINT fk_refund_order FOREIGN KEY (order_id) REFERENCES orders (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
