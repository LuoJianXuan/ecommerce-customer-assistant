-- ch10 · 低置信问题离线主题分类结果（多标签）
SET NAMES utf8mb4;

CREATE TABLE topic_classifications (
  id            BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  question_id   BIGINT UNSIGNED NOT NULL,
  labels        JSON            NOT NULL,
  scores        JSON            NOT NULL,
  model_version VARCHAR(128)    NOT NULL,
  batch_id      VARCHAR(64)     NOT NULL,
  classified_at DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_topic_question (question_id),
  KEY idx_topic_batch (batch_id),
  KEY idx_topic_model (model_version),
  KEY idx_topic_time (classified_at),
  CONSTRAINT fk_topic_question FOREIGN KEY (question_id)
    REFERENCES low_confidence_questions (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='低置信问题离线多标签主题分类结果';
