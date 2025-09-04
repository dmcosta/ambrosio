-- Ambrosio Database Initialization Script
-- Creates necessary tables and indexes for the AI agent system

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Tasks table - stores all tasks and workflows
CREATE TABLE IF NOT EXISTS tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    title VARCHAR(255) NOT NULL,
    description TEXT,
    repository VARCHAR(255),
    branch VARCHAR(255),
    assignee_agent VARCHAR(50),
    priority VARCHAR(10) DEFAULT 'medium',
    complexity VARCHAR(10) DEFAULT 'medium',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}',
    result JSONB,
    error_message TEXT
);

-- Agents table - tracks agent status and configuration
CREATE TABLE IF NOT EXISTS agents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(50) UNIQUE NOT NULL,
    type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'inactive',
    current_model VARCHAR(100),
    current_provider VARCHAR(50),
    last_seen TIMESTAMP WITH TIME ZONE,
    total_tasks_completed INTEGER DEFAULT 0,
    total_tasks_failed INTEGER DEFAULT 0,
    average_completion_time_seconds DECIMAL(10,2),
    configuration JSONB DEFAULT '{}',
    metrics JSONB DEFAULT '{}'
);

-- Task executions table - detailed execution logs
CREATE TABLE IF NOT EXISTS task_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id UUID NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    agent_name VARCHAR(50) NOT NULL,
    execution_step VARCHAR(100),
    status VARCHAR(20) NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    duration_seconds DECIMAL(10,3),
    input_data JSONB,
    output_data JSONB,
    error_details TEXT,
    model_used VARCHAR(100),
    tokens_consumed INTEGER,
    cost_usd DECIMAL(10,6)
);

-- Agent conversations table - stores conversation history
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id UUID NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    agent_name VARCHAR(50) NOT NULL,
    user_id VARCHAR(100),
    channel_id VARCHAR(100),
    message_type VARCHAR(20) NOT NULL, -- 'user', 'agent', 'system'
    content TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- GitHub integrations table - tracks GitHub operations
CREATE TABLE IF NOT EXISTS github_operations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id UUID NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    operation_type VARCHAR(50) NOT NULL, -- 'create_branch', 'create_pr', 'create_issue', etc.
    repository VARCHAR(255) NOT NULL,
    github_id VARCHAR(50), -- GitHub issue/PR number
    github_url VARCHAR(500),
    status VARCHAR(20) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    details JSONB DEFAULT '{}'
);

-- System metrics table - performance and usage metrics
CREATE TABLE IF NOT EXISTS system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_type VARCHAR(20) NOT NULL, -- 'counter', 'gauge', 'histogram'
    value DECIMAL(15,6) NOT NULL,
    labels JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Configuration history table - tracks configuration changes
CREATE TABLE IF NOT EXISTS configuration_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_name VARCHAR(50) NOT NULL,
    configuration JSONB NOT NULL,
    changed_by VARCHAR(100),
    change_reason TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_type ON tasks(task_type);
CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks(created_at);
CREATE INDEX IF NOT EXISTS idx_tasks_repository ON tasks(repository);
CREATE INDEX IF NOT EXISTS idx_tasks_assignee ON tasks(assignee_agent);

CREATE INDEX IF NOT EXISTS idx_agents_name ON agents(name);
CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status);
CREATE INDEX IF NOT EXISTS idx_agents_type ON agents(type);

CREATE INDEX IF NOT EXISTS idx_task_executions_task_id ON task_executions(task_id);
CREATE INDEX IF NOT EXISTS idx_task_executions_agent ON task_executions(agent_name);
CREATE INDEX IF NOT EXISTS idx_task_executions_status ON task_executions(status);

CREATE INDEX IF NOT EXISTS idx_conversations_task_id ON conversations(task_id);
CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp);

CREATE INDEX IF NOT EXISTS idx_github_ops_task_id ON github_operations(task_id);
CREATE INDEX IF NOT EXISTS idx_github_ops_repo ON github_operations(repository);
CREATE INDEX IF NOT EXISTS idx_github_ops_type ON github_operations(operation_type);

CREATE INDEX IF NOT EXISTS idx_metrics_name ON system_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON system_metrics(timestamp);

CREATE INDEX IF NOT EXISTS idx_config_history_agent ON configuration_history(agent_name);
CREATE INDEX IF NOT EXISTS idx_config_history_created ON configuration_history(created_at);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add trigger to tasks table
DROP TRIGGER IF EXISTS update_tasks_updated_at ON tasks;
CREATE TRIGGER update_tasks_updated_at
    BEFORE UPDATE ON tasks
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Insert default agent configurations
INSERT INTO agents (name, type, status, current_model, current_provider) VALUES
    ('planner', 'planning', 'inactive', 'gemma2:9b', 'gemini'),
    ('coder', 'development', 'inactive', 'claude-sonnet-4-20250514', 'claude'),
    ('reviewer', 'quality_assurance', 'inactive', 'llama3.1:8b', 'ollama')
ON CONFLICT (name) DO NOTHING;

-- Create views for common queries
CREATE OR REPLACE VIEW task_summary AS
SELECT 
    t.id,
    t.task_type,
    t.status,
    t.title,
    t.repository,
    t.assignee_agent,
    t.priority,
    t.created_at,
    t.completed_at,
    EXTRACT(EPOCH FROM (COALESCE(t.completed_at, NOW()) - t.created_at)) as duration_seconds,
    COUNT(te.id) as execution_count,
    AVG(te.duration_seconds) as avg_step_duration
FROM tasks t
LEFT JOIN task_executions te ON t.id = te.task_id
GROUP BY t.id, t.task_type, t.status, t.title, t.repository, t.assignee_agent, 
         t.priority, t.created_at, t.completed_at;

CREATE OR REPLACE VIEW agent_performance AS
SELECT 
    a.name,
    a.type,
    a.status,
    a.current_model,
    a.total_tasks_completed,
    a.total_tasks_failed,
    a.average_completion_time_seconds,
    COUNT(te.id) as recent_executions,
    AVG(te.duration_seconds) as avg_recent_duration,
    MAX(a.last_seen) as last_activity
FROM agents a
LEFT JOIN task_executions te ON a.name = te.agent_name 
    AND te.started_at > NOW() - INTERVAL '24 hours'
GROUP BY a.name, a.type, a.status, a.current_model, a.total_tasks_completed, 
         a.total_tasks_failed, a.average_completion_time_seconds;

-- Grant permissions to the application user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ambrosio_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO ambrosio_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO ambrosio_user;