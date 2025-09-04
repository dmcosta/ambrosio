#!/bin/bash

# Ambrosio UNRAID Deployment Script
# Deploys the complete AI agent system to UNRAID

set -e

echo "🚀 Starting Ambrosio deployment on UNRAID..."

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
APPDATA_DIR="/mnt/user/appdata/ambrosio"
LOG_FILE="$APPDATA_DIR/deployment.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE" 2>/dev/null || true
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    echo "[ERROR] $1" >> "$LOG_FILE" 2>/dev/null || true
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
    echo "[SUCCESS] $1" >> "$LOG_FILE" 2>/dev/null || true
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
    echo "[WARNING] $1" >> "$LOG_FILE" 2>/dev/null || true
}

# Check if running on UNRAID
check_unraid() {
    if [ ! -f "/etc/unraid-version" ]; then
        warning "This script is designed for UNRAID. Proceeding anyway..."
    else
        success "Running on UNRAID $(cat /etc/unraid-version)"
    fi
}

# Create required directories
create_directories() {
    log "Creating UNRAID directory structure..."
    
    mkdir -p "$APPDATA_DIR"/{logs,config,data}
    mkdir -p "$APPDATA_DIR"/data/{redis-data,postgres-data,prometheus-data,grafana-data,loki-data}
    
    # Set permissions
    chmod -R 755 "$APPDATA_DIR"
    
    success "Directory structure created at $APPDATA_DIR"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        error "Docker is not running or not accessible"
        exit 1
    fi
    
    # Check if Docker Compose is available
    if ! command -v docker-compose >/dev/null 2>&1; then
        error "Docker Compose is not installed"
        echo "Please install Docker Compose Manager plugin from Community Applications"
        exit 1
    fi
    
    # Check for .env file
    if [ ! -f "$PROJECT_DIR/.env" ]; then
        error ".env file not found"
        echo "Please copy .env.example to .env and configure your API keys"
        exit 1
    fi
    
    # Check required environment variables
    source "$PROJECT_DIR/.env"
    
    if [ -z "$DISCORD_TOKEN" ]; then
        error "DISCORD_TOKEN not set in .env file"
        exit 1
    fi
    
    if [ -z "$GITHUB_TOKEN" ]; then
        warning "GITHUB_TOKEN not set - GitHub integration will be limited"
    fi
    
    if [ -z "$GEMINI_API_KEY" ] && [ -z "$CLAUDE_API_KEY" ]; then
        error "At least one AI provider API key (GEMINI_API_KEY or CLAUDE_API_KEY) must be set"
        exit 1
    fi
    
    success "Prerequisites check passed"
}

# Stop existing services
stop_existing_services() {
    log "Stopping any existing Ambrosio services..."
    
    cd "$PROJECT_DIR"
    
    # Stop services gracefully
    docker-compose down --remove-orphans 2>/dev/null || true
    
    # Remove any dangling containers
    docker container prune -f >/dev/null 2>&1 || true
    
    success "Existing services stopped"
}

# Copy configuration files
copy_configurations() {
    log "Copying configuration files..."
    
    # Copy config files to UNRAID appdata
    cp -r "$PROJECT_DIR/config"/* "$APPDATA_DIR/config/"
    
    # Copy environment file
    cp "$PROJECT_DIR/.env" "$APPDATA_DIR/"
    
    success "Configuration files copied"
}

# Build and start services
deploy_services() {
    log "Building and starting Ambrosio services..."
    
    cd "$PROJECT_DIR"
    
    # Build services
    log "Building Docker images..."
    docker-compose build --no-cache
    
    # Start infrastructure services first
    log "Starting infrastructure services..."
    docker-compose up -d zookeeper kafka redis postgres
    
    # Wait for infrastructure to be ready
    log "Waiting for infrastructure services to be ready..."
    sleep 30
    
    # Check service health
    check_service_health "kafka" "9092"
    check_service_health "redis" "6380" 
    check_service_health "postgres" "5433"
    
    # Start configuration manager
    log "Starting configuration manager..."
    docker-compose up -d config-manager
    sleep 10
    
    # Start agent services
    log "Starting AI agent services..."
    docker-compose up -d planner-agent coder-agent reviewer-agent
    sleep 15
    
    # Start orchestrator
    log "Starting orchestrator..."
    docker-compose up -d orchestrator
    sleep 10
    
    # Start chat interface
    log "Starting Discord bot..."
    docker-compose up -d discord-bot
    sleep 5
    
    # Start monitoring services
    log "Starting monitoring services..."
    docker-compose up -d prometheus grafana loki
    
    success "All services started"
}

check_service_health() {
    local service=$1
    local port=$2
    local max_attempts=30
    local attempt=1
    
    log "Checking health of $service on port $port..."
    
    while [ $attempt -le $max_attempts ]; do
        if nc -z localhost "$port" 2>/dev/null; then
            success "$service is healthy"
            return 0
        fi
        
        log "Attempt $attempt/$max_attempts: $service not ready yet..."
        sleep 2
        ((attempt++))
    done
    
    error "$service failed to start properly"
    return 1
}

# Verify deployment
verify_deployment() {
    log "Verifying deployment..."
    
    # Check if all containers are running
    local expected_containers=(
        "ambrosio-zookeeper"
        "ambrosio-kafka" 
        "ambrosio-redis"
        "ambrosio-postgres"
        "ambrosio-config-manager"
        "ambrosio-planner"
        "ambrosio-coder"
        "ambrosio-reviewer"
        "ambrosio-orchestrator"
        "ambrosio-discord-bot"
        "ambrosio-prometheus"
        "ambrosio-grafana"
        "ambrosio-loki"
    )
    
    local running_containers=$(docker ps --format "{{.Names}}" | grep "^ambrosio-" || true)
    
    for container in "${expected_containers[@]}"; do
        if echo "$running_containers" | grep -q "^$container$"; then
            success "✅ $container is running"
        else
            error "❌ $container is not running"
        fi
    done
    
    # Check service endpoints
    log "Checking service endpoints..."
    
    local endpoints=(
        "Config Manager:http://localhost:8080/health"
        "Orchestrator:http://localhost:8081/health"
        "Prometheus:http://localhost:9091"
        "Grafana:http://localhost:3002"
        "Loki:http://localhost:3101/ready"
    )
    
    for endpoint in "${endpoints[@]}"; do
        local name="${endpoint%%:*}"
        local url="${endpoint#*:}"
        
        if curl -s "$url" >/dev/null 2>&1; then
            success "✅ $name is accessible"
        else
            warning "⚠️  $name may not be ready yet"
        fi
    done
}

# Initialize the system
initialize_system() {
    log "Initializing Ambrosio system..."
    
    # Wait for config manager to be ready
    sleep 10
    
    # Reload agent configurations
    curl -s -X POST http://localhost:8080/reload >/dev/null 2>&1 || true
    
    # Check system status
    curl -s http://localhost:8080/status >/dev/null 2>&1 || true
    
    success "System initialization completed"
}

# Print deployment summary
print_summary() {
    echo
    echo "🎉 Ambrosio deployment completed successfully!"
    echo
    echo "📊 Access Points:"
    echo "  • Grafana Dashboard: http://$(hostname -I | awk '{print $1}'):3002 (admin/ambrosio)"
    echo "  • Prometheus: http://$(hostname -I | awk '{print $1}'):9091"
    echo "  • Config Manager: http://$(hostname -I | awk '{print $1}'):8080"
    echo "  • Orchestrator API: http://$(hostname -I | awk '{print $1}'):8081"
    echo
    echo "🤖 Discord Bot:"
    echo "  • Bot should now be online in your Discord server"
    echo "  • Try: /create-feature description:\"Test feature\" repository:\"your-org/your-repo\""
    echo
    echo "📁 Data Location:"
    echo "  • Logs: $APPDATA_DIR/logs"
    echo "  • Data: $APPDATA_DIR/data"
    echo "  • Config: $APPDATA_DIR/config"
    echo
    echo "🔧 Management Commands:"
    echo "  • View logs: docker-compose logs -f [service-name]"
    echo "  • Restart: cd $PROJECT_DIR && docker-compose restart"
    echo "  • Stop: cd $PROJECT_DIR && docker-compose down"
    echo "  • Status: cd $PROJECT_DIR && docker-compose ps"
    echo
    echo "📖 Next Steps:"
    echo "  1. Configure your Discord bot permissions and add it to your server"
    echo "  2. Set up GitHub webhooks for automated triggers (optional)"
    echo "  3. Customize agent configurations in $APPDATA_DIR/config"
    echo "  4. Check Grafana dashboard for system metrics"
    echo
}

# Main deployment function
main() {
    echo "🤖 Ambrosio AI Development Team - UNRAID Deployment"
    echo "=================================================="
    
    check_unraid
    create_directories
    check_prerequisites
    stop_existing_services
    copy_configurations
    deploy_services
    verify_deployment
    initialize_system
    print_summary
    
    success "Deployment completed! Check the summary above for access information."
}

# Handle script interruption
trap 'error "Deployment interrupted"; exit 1' INT TERM

# Run main function
main "$@"