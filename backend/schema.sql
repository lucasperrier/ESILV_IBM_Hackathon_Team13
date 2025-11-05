CREATE DATABASE IF NOT EXISTS plv_helpcenter
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE plv_helpcenter;

-- Table des FAQ (issues du CSV)
CREATE TABLE IF NOT EXISTS faqs (
  id INT PRIMARY KEY,
  title TEXT NOT NULL,
  content MEDIUMTEXT NOT NULL,
  date DATE NULL,
  post_type VARCHAR(50),
  languages VARCHAR(100),
  topics TEXT,
  users VARCHAR(255),
  schools VARCHAR(255),
  status VARCHAR(50),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Logs d'interactions pour le self-learning
CREATE TABLE IF NOT EXISTS interactions (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  user_id VARCHAR(128) NULL,
  query_text TEXT NOT NULL,
  detected_lang VARCHAR(10) NULL,
  retrieved_faq_ids TEXT NULL,  -- ex: "2133,1012,..." (top-k)
  chosen_faq_id INT NULL,
  decision ENUM('answer','redirect','no_match') NOT NULL,
  answer_html MEDIUMTEXT NULL,
  FOREIGN KEY (chosen_faq_id) REFERENCES faqs(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

