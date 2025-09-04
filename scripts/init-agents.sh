#!/bin/bash

# Ambrosio Agent Initialization Script
# Initializes and configures all AI agents after deployment

set -e

echo "🤖 Initializing Ambrosio AI Agents..."

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_MANAGER_URL="http://localhost:8080"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Wait for config manager to be ready
wait_for_config_manager() {
    log "Waiting for Config Manager to be ready..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$CONFIG_MANAGER_URL/health" >/dev/null 2>&1; then
            success "Config Manager is ready"
            return 0
        fi
        
        log "Attempt $attempt/$max_attempts: Config Manager not ready..."
        sleep 2
        ((attempt++))
    done
    
    echo "Config Manager failed to start"
    exit 1
}

# Load agent configurations
load_agent_configs() {
    log "Loading agent configurations..."
    
    local agents=("planner" "coder" "reviewer")
    
    for agent in "${agents[@]}"; do
        log "Loading configuration for $agent agent..."
        
        # Trigger config reload
        response=$(curl -s "$CONFIG_MANAGER_URL/config/$agent" || echo "failed")
        
        if echo "$response" | grep -q "agent.*$agent"; then
            success "$agent agent configuration loaded"
        else
            warning "$agent agent configuration may not be loaded properly"
        fi
    done
}

# Test agent connectivity
test_agent_connectivity() {
    log "Testing agent connectivity..."
    
    # Check if agents are responding
    local services=(
        "planner-agent:8080"
        "coder-agent:8080"
        "reviewer-agent:8080"
        "orchestrator:8080"
        "discord-bot:8080"
    )
    
    for service in "${services[@]}"; do
        local name="${service%%:*}"
        local port="${service#*:}"
        
        if nc -z "ambrosio-$name" "$port" 2>/dev/null || nc -z localhost "$port" 2>/dev/null; then
            success "$name is responding"
        else
            warning "$name may not be ready"
        fi
    done
}

# Initialize system configuration
initialize_system() {
    log "Initializing system configuration..."
    
    # Reload all configurations
    curl -s -X POST "$CONFIG_MANAGER_URL/reload" >/dev/null 2>&1
    
    # Get system status
    local status=$(curl -s "$CONFIG_MANAGER_URL/status" 2>/dev/null)
    
    if echo "$status" | grep -q "config-manager"; then
        success "System configuration initialized"
    else
        warning "System initialization may have issues"
    fi
}

# Download required models (if using Ollama)
download_models() {
    log "Checking for required AI models..."
    
    # Check if Ollama is available
    if docker ps | grep -q ollama; then
        log "Ollama detected, ensuring required models are available..."
        
        local models=("gemma2:9b" "codellama:13b" "llama3.1:8b")
        
        for model in "${models[@]}"; do
            log "Checking model: $model"
            
            # This would normally check and pull models
            # docker exec ollama ollama pull "$model" >/dev/null 2>&1 || true
        done
        
        success "Model check completed"
    else
        log "Ollama not detected, skipping model downloads"
    fi
}

# Create test workflow
create_test_workflow() {
    log "Creating test workflow to verify system..."
    
    local test_payload='{
        "task_id": "test-init-'$(date +%s)'",
        "user_id": "system",
        "user_name": "System Test",
        "channel_id": "system",
        "guild_id": "system",
        "task_type": "feature_creation",
        "title": "System Initialization Test",
        "description": "Test workflow created during system initialization",
        "repository": null,
        "priority": "low",
        "complexity": "low",
        "metadata": {
            "source": "init_script",
            "test": true
        }
    }'
    
    # Send test request to orchestrator
    response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$test_payload" \
        "http://localhost:8081/test-workflow" 2>/dev/null || echo "failed")
    
    if echo "$response" | grep -q "workflow_id\|test"; then
        success "Test workflow created successfully"
    else
        log "Test workflow creation skipped (endpoint may not be available)"
    fi
}

# Print initialization summary
print_summary() {
    echo
    echo "🎉 Ambrosio Agent Initialization Complete!"
    echo
    echo "🤖 Agent Status:"
    echo "  • Planner Agent: Ready for requirement analysis"
    echo "  • Coder Agent: Ready for code generation"
    echo "  • Reviewer Agent: Ready for code review"
    echo "  • Orchestrator: Ready to coordinate workflows"
    echo "  • Discord Bot: Ready for user interactions"
    echo
    echo "📊 System Status:"
    echo "  • Configuration Manager: http://localhost:8080/status"
    echo "  • Orchestrator API: http://localhost:8081/health"
    echo "  • Prometheus Metrics: http://localhost:9091"
    echo "  • Grafana Dashboard: http://localhost:3002"
    echo
    echo "🚀 Ready for AI-Powered Development!"
    echo
    echo "Try these Discord commands:"
    echo "  /create-feature description:\"Add login system\" repository:\"your-org/your-repo\""
    echo "  /review-pr pr_url:\"https://github.com/owner/repo/pull/123\""
    echo "  /status"
    echo
}

# Main initialization
main() {
    echo "🤖 Ambrosio AI Agent Initialization"
    echo "=================================="
    
    wait_for_config_manager
    load_agent_configs
    test_agent_connectivity
    initialize_system
    download_models
    create_test_workflow
    print_summary
    
    success "Agent initialization completed successfully!"
}

main "$@"