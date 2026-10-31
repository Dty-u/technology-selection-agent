CREATE TABLE IF NOT EXISTS agent_projects (
    user_id VARCHAR(128) NOT NULL,
    project_id VARCHAR(128) NOT NULL,
    name VARCHAR(255) NOT NULL,
    created_at DATETIME(6) NOT NULL,
    updated_at DATETIME(6) NOT NULL,
    PRIMARY KEY (user_id, project_id),
    INDEX idx_agent_projects_user (user_id, updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS agent_sessions (
    thread_id VARCHAR(128) PRIMARY KEY,
    user_id VARCHAR(128) NOT NULL,
    project_id VARCHAR(128) NOT NULL,
    title VARCHAR(255) NOT NULL,
    status VARCHAR(32) NOT NULL,
    created_at DATETIME(6) NOT NULL,
    updated_at DATETIME(6) NOT NULL,
    INDEX idx_agent_sessions_scope (user_id, project_id, updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS agent_messages (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    thread_id VARCHAR(128) NOT NULL,
    request_id CHAR(36) NOT NULL,
    role ENUM('user', 'assistant', 'system') NOT NULL,
    content LONGTEXT NOT NULL,
    delivery VARCHAR(16) NOT NULL,
    kind VARCHAR(16) NOT NULL DEFAULT 'message',
    metadata JSON NULL,
    created_at DATETIME(6) NOT NULL,
    UNIQUE KEY uq_agent_messages_request (thread_id, request_id, role),
    INDEX idx_agent_messages_thread (thread_id, id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
