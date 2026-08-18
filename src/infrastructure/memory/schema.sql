CREATE TABLE IF NOT EXISTS memory_threads (
    thread_id VARCHAR(128) PRIMARY KEY,
    user_id VARCHAR(128) NOT NULL,
    project_id VARCHAR(128) NULL,
    status ENUM('active', 'closed') NOT NULL DEFAULT 'active',
    created_at DATETIME(6) NOT NULL,
    updated_at DATETIME(6) NOT NULL,
    INDEX idx_memory_threads_user (user_id, updated_at),
    INDEX idx_memory_threads_project (project_id, updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS memory_items (
    memory_id CHAR(36) PRIMARY KEY,
    idempotency_key VARCHAR(191) NOT NULL UNIQUE,
    user_id VARCHAR(128) NOT NULL,
    thread_id VARCHAR(128) NOT NULL,
    project_id VARCHAR(128) NULL,
    memory_type VARCHAR(64) NOT NULL,
    content LONGTEXT NOT NULL,
    status ENUM('pending', 'confirmed', 'approved', 'superseded', 'deleted') NOT NULL,
    source_kind VARCHAR(64) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    metadata_json JSON NOT NULL,
    version INT UNSIGNED NOT NULL DEFAULT 1,
    created_at DATETIME(6) NOT NULL,
    updated_at DATETIME(6) NOT NULL,
    INDEX idx_memory_items_user_scope (user_id, project_id, status, updated_at),
    INDEX idx_memory_items_thread (thread_id, updated_at),
    INDEX idx_memory_items_type (memory_type, status, updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS memory_versions (
    memory_id CHAR(36) NOT NULL,
    version INT UNSIGNED NOT NULL,
    content LONGTEXT NOT NULL,
    status VARCHAR(32) NOT NULL,
    metadata_json JSON NOT NULL,
    changed_by VARCHAR(128) NOT NULL,
    created_at DATETIME(6) NOT NULL,
    PRIMARY KEY (memory_id, version)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS selection_runs (
    run_id CHAR(36) PRIMARY KEY,
    user_id VARCHAR(128) NOT NULL,
    thread_id VARCHAR(128) NOT NULL,
    project_id VARCHAR(128) NOT NULL,
    recommendation_json JSON NOT NULL,
    approval_status ENUM('pending', 'approved', 'rejected') NOT NULL DEFAULT 'pending',
    created_at DATETIME(6) NOT NULL,
    updated_at DATETIME(6) NOT NULL,
    INDEX idx_selection_runs_project (project_id, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS memory_outbox (
    event_id CHAR(36) PRIMARY KEY,
    memory_id CHAR(36) NOT NULL,
    event_type ENUM('memory.upsert', 'memory.delete') NOT NULL,
    status ENUM('pending', 'processing', 'projected', 'failed') NOT NULL DEFAULT 'pending',
    attempts INT UNSIGNED NOT NULL DEFAULT 0,
    payload_json JSON NOT NULL,
    last_error TEXT NULL,
    available_at DATETIME(6) NOT NULL,
    created_at DATETIME(6) NOT NULL,
    updated_at DATETIME(6) NOT NULL,
    INDEX idx_memory_outbox_ready (status, available_at),
    INDEX idx_memory_outbox_memory (memory_id, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS memory_projections (
    memory_id CHAR(36) PRIMARY KEY,
    provider VARCHAR(32) NOT NULL,
    external_id VARCHAR(191) NOT NULL,
    collection_name VARCHAR(191) NOT NULL,
    projected_version INT UNSIGNED NOT NULL,
    projected_at DATETIME(6) NOT NULL,
    UNIQUE KEY uq_memory_projection_external (provider, external_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS memory_audit_events (
    audit_id CHAR(36) PRIMARY KEY,
    memory_id CHAR(36) NULL,
    user_id VARCHAR(128) NOT NULL,
    thread_id VARCHAR(128) NOT NULL,
    action VARCHAR(64) NOT NULL,
    details_json JSON NOT NULL,
    created_at DATETIME(6) NOT NULL,
    INDEX idx_memory_audit_memory (memory_id, created_at),
    INDEX idx_memory_audit_user (user_id, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
