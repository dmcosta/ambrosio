# Ambrosio - AI Development Team Microservices

**Ambrosio** is an intelligent, microservices-based AI agent system that provides GitHub-integrated development assistance through Discord/Mattermost chat interfaces. Named after the wise advisor, Ambrosio coordinates specialized AI agents to handle complex development workflows.

## 🏗️ Architecture Overview

Ambrosio follows a microservices architecture where each AI agent runs in its own container with single responsibility:

- **🤖 Planner Agent**: Requirements analysis and task breakdown
- **👨‍💻 Coder Agent**: Code generation and implementation  
- **🔍 Reviewer Agent**: Code review and quality assurance
- **🎭 Orchestrator**: Workflow coordination and GitHub integration
- **💬 Chat Bots**: Discord/Mattermost interface layer

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- UNRAID server (or Docker-capable Linux system)
- Discord Bot Token
- GitHub Personal Access Token
- API keys for LLM providers (Gemini, Claude, or local Ollama)

### Installation

1. **Clone and Setup**:
```bash
cd ~/Workspaces/Personal/Ambrosio
cp .env.example .env
# Edit .env with your API keys and tokens
```

2. **Deploy to UNRAID**:
```bash
./scripts/deploy-unraid.sh
```

3. **Initialize Agents**:
```bash
./scripts/init-agents.sh
```

## 🔧 Configuration

Agents are configured through YAML templates in `config/agent-templates/`. Each agent supports:

- **Model Selection**: Choose between Gemini, Claude, or local Ollama models
- **Behavior Customization**: Adjust persona, communication style, and decision-making
- **Tool Integration**: Enable/disable specific capabilities per agent
- **Resource Limits**: Configure memory and CPU constraints

## 📊 Monitoring

- **Grafana Dashboard**: http://your-unraid:3001
- **Prometheus Metrics**: http://your-unraid:9090  
- **Log Aggregation**: Centralized logging via Loki
- **Real-time Chat**: Discord bot responds with status updates

## 🗂️ Project Structure

```
ambrosio/
├── services/               # Microservice implementations
│   ├── discord-bot/       # Chat interface
│   ├── planner-agent/     # Requirements & planning
│   ├── coder-agent/       # Code generation
│   ├── reviewer-agent/    # Code review
│   ├── orchestrator/      # Workflow coordination
│   └── config-manager/    # Dynamic configuration
├── config/                # Agent templates & system config
├── scripts/              # Deployment and utility scripts
├── docker-compose.yml    # Main orchestration file
└── docs/                # Documentation
```

## 🌟 Features

- **Multi-Platform LLM Support**: Seamlessly switch between cloud APIs and local models
- **Dynamic Agent Configuration**: Runtime behavior modification without restarts
- **GitHub Integration**: Automatic branch creation, PR management, issue tracking
- **Scalable Architecture**: Independent scaling of agent services
- **UNRAID Optimized**: Native integration with UNRAID Community Apps
- **Comprehensive Monitoring**: Full observability stack included

## 📝 Usage Examples

### Discord Commands
```
/ambrosio create-feature "Add user authentication system" --repo myproject --priority high
/ambrosio review-pr "https://github.com/user/repo/pull/123"
/ambrosio analyze-codebase --repo myproject --focus security
```

### REST API (for advanced users)
```bash
curl -X POST http://ambrosio:8080/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{"type": "feature", "description": "...", "repo": "..."}'
```

---

**🧠 Ambrosio**: Your AI-powered development companion, inspired by the wisdom of artificial intelligence and the efficiency of microservices.# ambrosio
