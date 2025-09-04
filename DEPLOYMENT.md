# Ambrosio - Deployment Guide

## Quick Start

1. **Configure Environment**:
```bash
cd ~/Workspaces/Personal/Ambrosio
cp .env.example .env
# Edit .env with your API keys
```

2. **Deploy to UNRAID**:
```bash
./scripts/deploy-unraid.sh
```

3. **Initialize Agents**:
```bash
./scripts/init-agents.sh
```

## Required API Keys

### Discord Bot
1. Go to https://discord.com/developers/applications
2. Create new application → Bot → Copy token
3. Set `DISCORD_TOKEN` in `.env`

### GitHub Integration
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate token with repo permissions
3. Set `GITHUB_TOKEN` in `.env`

### AI Models
**Option A - Gemini (Recommended for planning):**
1. Go to https://aistudio.google.com/app/apikey
2. Create API key
3. Set `GEMINI_API_KEY` in `.env`

**Option B - Claude (Recommended for coding):**
1. Go to https://console.anthropic.com/
2. Create API key
3. Set `CLAUDE_API_KEY` in `.env`

**Option C - Local Ollama (Free but requires more resources):**
- Uses your existing Ollama installation
- Models: gemma2:9b, codellama:13b, llama3.1:8b

## System Requirements

### UNRAID Server
- **CPU**: 8+ cores recommended
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 500GB+ available space
- **GPU**: Optional (RTX 4070+ for local models)

### Network Ports
- `8080`: Configuration Manager
- `8081`: Orchestrator API
- `9091`: Prometheus Metrics
- `3002`: Grafana Dashboard
- `3101`: Loki Logs

## Architecture Overview

```
Discord/Mattermost ──► Orchestrator ──► AI Agents
                            │              │
                            ▼              ▼
                       GitHub API     Local/Cloud
                                         LLMs
```

### Services
- **Config Manager**: Dynamic agent configuration
- **Discord Bot**: Chat interface with slash commands
- **Planner Agent**: Requirements analysis & task breakdown
- **Coder Agent**: Code generation & implementation
- **Reviewer Agent**: Code review & quality assurance
- **Orchestrator**: Workflow coordination & GitHub integration

### Data Flow
1. User sends Discord command: `/create-feature`
2. Discord bot validates and routes to Orchestrator
3. Orchestrator creates workflow and assigns to agents
4. Agents process tasks using configured AI models
5. Results create GitHub branches, issues, and PRs
6. User receives progress updates in Discord

## Configuration

### Agent Behavior
Edit files in `config/`:
- `planner-agent-config.yaml`
- `coder-agent-config.yaml` 
- `reviewer-agent-config.yaml`

### Model Selection
Set in `.env`:
```bash
# Per-agent model configuration
PLANNER_MODEL_PROVIDER=gemini
PLANNER_MODEL=gemma2:9b

CODER_MODEL_PROVIDER=claude
CODER_MODEL=claude-sonnet-4-20250514

REVIEWER_MODEL_PROVIDER=ollama
REVIEWER_MODEL=llama3.1:8b
```

## Monitoring & Troubleshooting

### Access Points
- **Grafana**: http://your-unraid-ip:3002 (admin/ambrosio)
- **Prometheus**: http://your-unraid-ip:9091
- **Logs**: `/mnt/user/appdata/ambrosio/logs/`

### Common Issues
1. **Discord bot offline**: Check `DISCORD_TOKEN` and bot permissions
2. **No GitHub integration**: Verify `GITHUB_TOKEN` permissions
3. **Agents not responding**: Check model API keys and quotas
4. **Memory issues**: Reduce concurrent tasks or use smaller models

### Log Analysis
```bash
# View service logs
docker-compose logs -f discord-bot
docker-compose logs -f planner-agent
docker-compose logs -f orchestrator

# View aggregated logs
tail -f /mnt/user/appdata/ambrosio/logs/*.log
```

## Scaling & Performance

### Resource Allocation
Adjust in `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      memory: 8G  # Increase for complex models
      cpus: '4'   # Increase for faster processing
```

### Model Optimization
- **Low complexity tasks**: Use Ollama local models
- **High complexity tasks**: Use Claude/Gemini APIs
- **Batch processing**: Configure in agent templates

## Security

### Network Security
- Services communicate via internal Docker network
- Only necessary ports exposed to host
- No direct internet access for agents

### Data Privacy
- Conversation data stored in Redis (TTL configured)
- GitHub tokens encrypted at rest
- API keys managed via environment variables

## Backup & Recovery

### Data Backup
```bash
# Backup configuration and data
tar -czf ambrosio-backup-$(date +%Y%m%d).tar.gz \
  /mnt/user/appdata/ambrosio/config \
  /mnt/user/appdata/ambrosio/data
```

### Disaster Recovery
```bash
# Restore from backup
tar -xzf ambrosio-backup-*.tar.gz -C /

# Restart services
docker-compose down && docker-compose up -d
```

## Updates

### Update Ambrosio
```bash
cd ~/Workspaces/Personal/Ambrosio
git pull
./scripts/deploy-unraid.sh
```

### Update Individual Service
```bash
docker-compose build [service-name]
docker-compose up -d [service-name]
```