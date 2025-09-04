# Ambrosio Portfolio Redesign - Complete Plan

## 🎯 Project Goals & Portfolio Value

### Primary Objectives
1. **Showcase Modern AI Engineering Skills**
   - CrewAI agent orchestration (industry standard)
   - RAG implementation with vector databases
   - Multi-model LLM integration
   - Production-ready safety controls

2. **Demonstrate Full-Stack Competency**
   - API design and async programming
   - Database design and optimization
   - Docker containerization
   - Monitoring and observability

3. **Build Something That Actually Works**
   - Functional Discord bot with real GitHub integration
   - Reliable agent workflows that complete tasks
   - Proper error handling and graceful degradation
   - Cost-controlled and safe for experimentation

4. **Create Portfolio Talking Points**
   - "I built an AI agent system using CrewAI that can..."
   - "I implemented RAG with ChromaDB to give agents contextual knowledge..."
   - "I designed safety mechanisms to prevent AI from going rogue..."

---

## 🏗️ Architecture Redesign - From 13 Services to 4

### Current Problems with v1 Architecture
- **Over-engineered**: 13 microservices for a personal project
- **Kafka overkill**: Complex message queuing for simple workflows  
- **Configuration complexity**: Whole service just to serve YAML files
- **Resource intensive**: 26GB RAM, 13+ CPU cores
- **Hard to debug**: Error could be anywhere in 13 services
- **No real AI agent coordination**: Just message passing between services

### Proposed v2 Architecture
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Discord   │───▶│  Gateway    │───▶│   CrewAI    │───▶│   Agents    │
│   Commands  │    │  Service    │    │Orchestrator │    │   Pool      │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                           │                   │                   │
                           ▼                   ▼                   ▼
                   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
                   │ Authentication│    │RAG Knowledge│    │GitHub Tools │
                   │Rate Limiting │    │  ChromaDB   │    │Code Analysis│
                   └─────────────┘    └─────────────┘    └─────────────┘
```

**Key Improvements:**
- **4 services instead of 13**
- **Direct HTTP API calls instead of Kafka**
- **CrewAI handles agent coordination** (industry standard)
- **Proper RAG system** with vector embeddings
- **~8GB RAM, 4-6 CPU cores** (75% resource reduction)

---

## 📋 Detailed Service Design

### 1. Gateway Service (FastAPI + Discord Integration)
**Purpose**: Single entry point, authentication, rate limiting, cost tracking

**Responsibilities:**
- Discord webhook handling and slash command processing
- API key management and authentication
- Request routing to appropriate services
- Cost tracking and daily limits ($25/day max)
- Rate limiting (10 requests/hour per user)
- Request/response logging and metrics

**Tech Stack:**
- FastAPI with async endpoints
- Discord.py for webhook integration  
- Redis for rate limiting and caching
- Prometheus metrics
- Structured logging (JSON)

**Key Features:**
```python
@app.post("/discord/commands")
async def handle_discord_command(command: DiscordCommand):
    # 1. Authenticate and rate limit
    await check_rate_limits(command.user_id)
    
    # 2. Track costs
    estimated_cost = estimate_task_cost(command)
    await check_daily_budget(estimated_cost)
    
    # 3. Route to CrewAI
    result = await crew_service.execute_task(command)
    
    # 4. Log and return
    await log_request(command, result, estimated_cost)
    return result
```

### 2. CrewAI Orchestrator Service 
**Purpose**: AI agent coordination using CrewAI framework

**Why CrewAI:**
- Industry standard for AI agent orchestration
- Built-in task delegation and crew management
- Handles agent communication protocols
- Supports different agent roles and tools
- Great for portfolio (employers recognize it)

**Responsibilities:**
- Create specialized agent crews based on task type
- Coordinate multi-step workflows
- Manage agent communication and data passing
- Handle failures and retries
- Track task progress and completion

**Agent Crews by Task Type:**
```python
# Feature Development Crew
feature_crew = Crew(
    agents=[
        product_manager_agent,  # Requirements analysis
        senior_developer_agent, # Implementation
        code_reviewer_agent,    # Quality assurance
        technical_writer_agent  # Documentation
    ],
    process=Process.sequential
)

# Code Review Crew  
review_crew = Crew(
    agents=[
        security_specialist_agent, # Security analysis
        performance_analyst_agent, # Performance review  
        code_reviewer_agent       # General quality
    ],
    process=Process.hierarchical
)

# Quick Question Crew
qa_crew = Crew(
    agents=[
        knowledge_agent  # RAG-powered Q&A
    ],
    process=Process.sequential
)
```

**Key Features:**
- Dynamic crew assembly based on task complexity
- Agent memory and context sharing
- Proper error handling and task retries
- Integration with RAG system for context
- Progress tracking and user notifications

### 3. RAG Knowledge System (ChromaDB + LangChain)
**Purpose**: Semantic code search and contextual knowledge for agents

**Why This is Portfolio Gold:**
- RAG is the hottest AI trend right now
- Shows understanding of vector databases
- Demonstrates semantic search implementation
- Practical application of embeddings

**Components:**
- **ChromaDB**: Vector database for embeddings
- **LangChain**: Document processing and embedding
- **Code Parser**: Intelligent code chunking
- **Semantic Search**: Context retrieval for agents

**Knowledge Sources:**
1. **Repository Code**: Parse and index all code files
2. **Documentation**: README, wiki, API docs
3. **Issue History**: Past GitHub issues and solutions
4. **Code Comments**: Inline documentation
5. **Test Files**: Understanding of expected behavior

**Implementation Approach:**
```python
class RAGKnowledgeSystem:
    def index_repository(self, repo_url):
        # 1. Clone and parse repository
        code_files = self.parse_codebase(repo_url)
        
        # 2. Intelligent chunking
        chunks = self.chunk_code_semantically(code_files)
        
        # 3. Generate embeddings
        embeddings = self.create_embeddings(chunks)
        
        # 4. Store in ChromaDB
        self.chroma_client.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=self.extract_metadata(code_files)
        )
    
    def search_relevant_context(self, query, k=5):
        # Semantic search for relevant code/docs
        results = self.chroma_client.query(
            query_texts=[query],
            n_results=k
        )
        return self.format_context(results)
```

**Advanced Features:**
- **Code similarity detection**: Find similar implementations
- **Documentation generation**: Auto-generate context for agents
- **Continuous indexing**: Update knowledge as repos change
- **Multi-repository support**: Index multiple projects

### 4. Agents Service (Specialized AI Agents with Tools)
**Purpose**: Individual AI agents with specific roles and tools

**Agent Specializations:**

#### Senior Developer Agent
- **Role**: Code architecture and implementation
- **Models**: GPT-4 for complex logic, Claude Sonnet for detailed implementation
- **Tools**: GitHub API, code analysis, testing frameworks, deployment scripts
- **Personality**: Experienced, focuses on maintainable code, considers edge cases

#### DevOps Engineer Agent  
- **Role**: CI/CD, deployment, infrastructure as code
- **Models**: GPT-4 for automation logic
- **Tools**: Docker, GitHub Actions, infrastructure templates
- **Personality**: Automates everything, security-conscious, monitors performance

#### Code Reviewer Agent
- **Role**: Security, performance, best practices
- **Models**: Claude for detailed analysis, GPT-4 for security patterns
- **Tools**: Static analysis, security scanners, performance profilers
- **Personality**: Detail-oriented, security-first, educational feedback

#### Technical Writer Agent
- **Role**: Documentation, README files, API docs
- **Models**: GPT-3.5 for efficiency, Claude for technical accuracy
- **Tools**: Markdown generation, API documentation tools
- **Personality**: Makes complex concepts accessible, comprehensive documentation

**Agent Implementation:**
```python
@agent
class SeniorDeveloperAgent:
    role = "Senior Full-Stack Developer"
    goal = "Write high-quality, maintainable code with proper testing"
    backstory = """You are a senior developer with 10+ years of experience.
                  You focus on clean architecture, proper testing, and maintainable code."""
    
    tools = [github_tool, code_analyzer_tool, test_generator_tool]
    llm = ChatOpenAI(model="gpt-4-turbo")
    
    def execute_task(self, task, context):
        # Get relevant code context from RAG
        relevant_code = rag_service.search(task.description)
        
        # Generate implementation with context
        implementation = self.llm.invoke(
            f"Task: {task.description}\n"
            f"Context: {relevant_code}\n"
            f"Generate implementation following best practices."
        )
        
        return implementation
```

---

## 🧠 RAG Implementation Details

### Why RAG is Essential for This Project
- **Context Awareness**: Agents understand the specific codebase
- **Consistency**: References existing patterns and conventions  
- **Learning**: System gets smarter as it indexes more repositories
- **Portfolio Value**: RAG is the most requested AI skill right now

### Vector Database Choice: ChromaDB
**Why ChromaDB over alternatives:**
- **Python-native**: Easy integration with FastAPI
- **Lightweight**: Perfect for portfolio/demo projects
- **Good performance**: Handles 100k+ embeddings efficiently  
- **Great documentation**: Easy to showcase in interviews
- **Open source**: No vendor lock-in

### Embedding Strategy
```python
# Code-specific embedding approach
class CodeEmbedder:
    def embed_code_chunk(self, code_chunk, metadata):
        # 1. Extract semantic information
        functions = self.extract_functions(code_chunk)
        imports = self.extract_imports(code_chunk)
        comments = self.extract_comments(code_chunk)
        
        # 2. Create contextual description
        context = f"""
        File: {metadata.file_path}
        Language: {metadata.language}
        Functions: {', '.join(functions)}
        Purpose: {self.infer_purpose(code_chunk)}
        Dependencies: {', '.join(imports)}
        
        Code:
        {code_chunk}
        """
        
        # 3. Generate embedding
        return self.embedding_model.embed(context)
```

### Context Retrieval for Agents
```python
# How agents get relevant context
def get_agent_context(task_description, repo_name):
    # 1. Semantic search for similar code
    similar_code = rag_service.search(
        query=task_description,
        filters={"repository": repo_name},
        k=5
    )
    
    # 2. Get related documentation
    docs = rag_service.search(
        query=task_description,
        filters={"type": "documentation"},
        k=3
    )
    
    # 3. Find relevant tests
    tests = rag_service.search(
        query=f"tests for {task_description}",
        filters={"type": "test"},
        k=2
    )
    
    return AgentContext(
        similar_implementations=similar_code,
        documentation=docs,
        test_examples=tests
    )
```

---

## 🛡️ Safety & Cost Control Mechanisms

### Cost Management
```python
class CostController:
    def __init__(self):
        self.daily_budget = 25.0  # $25/day
        self.current_spend = 0.0
        self.token_costs = {
            "gpt-4": 0.03 / 1000,      # $0.03 per 1K tokens
            "gpt-3.5": 0.002 / 1000,   # $0.002 per 1K tokens  
            "claude": 0.015 / 1000     # $0.015 per 1K tokens
        }
    
    def check_budget(self, estimated_tokens, model):
        estimated_cost = estimated_tokens * self.token_costs[model]
        
        if self.current_spend + estimated_cost > self.daily_budget:
            raise BudgetExceededException(
                f"Would exceed daily budget: ${self.daily_budget}"
            )
        
        return estimated_cost
    
    def track_usage(self, actual_tokens, model, cost):
        self.current_spend += cost
        # Log to database for analytics
        self.log_usage(actual_tokens, model, cost)
```

### GitHub Safety Controls
```python
class GitHubSafetyWrapper:
    def __init__(self):
        self.allowed_repos = self.load_whitelist()
        self.daily_limits = {
            "branches": 10,
            "issues": 20, 
            "prs": 5,
            "comments": 50
        }
        self.today_counts = self.load_today_counts()
    
    def safe_create_branch(self, repo, branch_name):
        # 1. Check repository whitelist
        if repo not in self.allowed_repos:
            raise UnauthorizedRepository(f"Repo {repo} not in whitelist")
        
        # 2. Check daily limits
        if self.today_counts["branches"] >= self.daily_limits["branches"]:
            raise DailyLimitExceeded("Branch creation limit reached")
        
        # 3. Validate branch name
        if not self.is_valid_branch_name(branch_name):
            raise InvalidBranchName(f"Invalid branch name: {branch_name}")
        
        # 4. Create with tracking
        result = self.github.create_branch(repo, branch_name)
        self.today_counts["branches"] += 1
        
        return result
```

### AI Response Validation
```python
class ResponseValidator:
    def validate_code_response(self, code, task_type):
        dangerous_patterns = [
            r"os\.system\(",           # System calls
            r"subprocess\.call\(",     # Process execution  
            r"eval\(",                 # Code evaluation
            r"exec\(",                 # Code execution
            r"__import__\(",           # Dynamic imports
            r"rm -rf",                 # Destructive commands
            r"DROP TABLE",             # Database destruction
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                raise DangerousCodeDetected(f"Pattern found: {pattern}")
        
        # Additional validations
        if len(code) > 50000:  # Max 50KB per response
            raise ResponseTooLarge()
        
        return True
```

---

## 🔄 Workflow Examples & Use Cases

### 1. Feature Development Workflow
```
User: /create-feature "Add OAuth authentication with Google"

1. Gateway → Validates request, checks budget
2. CrewAI → Assembles feature development crew
3. RAG → Searches for similar auth implementations  
4. Product Manager Agent → Analyzes requirements with context
5. Senior Developer Agent → Generates implementation plan
6. Code Reviewer Agent → Reviews generated code
7. Technical Writer Agent → Creates documentation
8. GitHub Integration → Creates branch, issues, PR
9. Discord → Notifies user with results
```

### 2. Code Review Workflow  
```
User: /review-pr "https://github.com/user/repo/pull/123"

1. Gateway → Fetches PR diff from GitHub
2. CrewAI → Assembles code review crew
3. RAG → Finds similar code patterns for comparison
4. Security Agent → Scans for vulnerabilities
5. Performance Agent → Analyzes performance impact
6. Code Reviewer Agent → General quality review
7. GitHub Integration → Posts review comments
8. Discord → Summarizes findings for user
```

### 3. Quick Question Workflow
```
User: /ask "How do I implement caching in this Next.js app?"

1. Gateway → Routes to Q&A crew
2. RAG → Searches codebase for caching patterns
3. Knowledge Agent → Generates answer with examples  
4. Discord → Returns contextual answer immediately
```

---

## 📊 Monitoring & Observability

### Metrics That Matter (Portfolio Perspective)
- **Cost Tracking**: Daily API spend, cost per task type
- **Performance**: Response times, task completion rates
- **Quality**: Success/failure rates, user satisfaction
- **Usage**: Popular commands, active users, repository activity

### Dashboard Design
```python
# Key metrics to showcase
class MetricsCollector:
    def track_task_completion(self, task_type, duration, cost, success):
        # Business metrics
        self.task_counter.labels(type=task_type, status=success).inc()
        self.task_duration.labels(type=task_type).observe(duration)
        self.daily_cost.inc(cost)
        
        # Technical metrics  
        self.rag_searches.inc()
        self.agent_invocations.labels(agent=task.agent).inc()
        
    def generate_daily_report(self):
        return {
            "tasks_completed": self.get_daily_tasks(),
            "cost_breakdown": self.get_cost_by_model(),
            "popular_commands": self.get_command_stats(),
            "success_rate": self.calculate_success_rate(),
            "rag_effectiveness": self.calculate_context_relevance()
        }
```

---

## 🚀 Deployment Strategy (UNRAID Optimized)

### Resource Requirements (Realistic)
- **CPU**: 4-6 cores (vs 13+ in v1)
- **RAM**: 8-12GB (vs 26GB in v1)  
- **Storage**: 100GB (for ChromaDB vectors)
- **Network**: Standard Docker networking

### Container Organization
```yaml
# docker-compose-v2.yml
services:
  gateway:
    build: ./gateway-service
    ports: ["8000:8000"]
    environment:
      - DISCORD_TOKEN=${DISCORD_TOKEN}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on: [redis]
    
  crew-orchestrator:
    build: ./crew-service  
    environment:
      - RAG_SERVICE_URL=http://rag-service:8001
    depends_on: [rag-service]
    
  rag-service:
    build: ./rag-service
    volumes:
      - chromadb-data:/app/data
    ports: ["8001:8001"]
    
  agents-service:
    build: ./agents-service
    environment:
      - GITHUB_TOKEN=${GITHUB_TOKEN}
    
  redis:
    image: redis:7-alpine
    volumes:
      - redis-data:/data
      
  prometheus:
    image: prom/prometheus
    ports: ["9090:9090"]
    
  grafana:
    image: grafana/grafana  
    ports: ["3000:3000"]
```

### Deployment Automation
```bash
#!/bin/bash
# deploy-v2.sh

# 1. Environment setup
cp .env.example .env
echo "Configure your API keys in .env file"

# 2. Create UNRAID directories  
mkdir -p /mnt/user/appdata/ambrosio-v2/{chromadb,redis,logs}

# 3. Build and start services
docker-compose -f docker-compose-v2.yml up -d

# 4. Initialize RAG system
./scripts/index-repositories.sh

# 5. Test Discord bot
./scripts/test-discord.sh

echo "Ambrosio v2 deployed! Access dashboard at http://unraid-ip:3000"
```

---

## 🎯 Portfolio Presentation Points

### Technical Skills Demonstrated
1. **AI/ML Engineering**
   - "I implemented a RAG system using ChromaDB and LangChain that gives AI agents contextual knowledge of codebases"
   - "I used CrewAI to orchestrate multiple specialized AI agents in coordinated workflows"
   - "I integrated multiple LLM providers (OpenAI, Anthropic, local models) with intelligent model selection"

2. **Backend Development**
   - "I designed a FastAPI-based microservices architecture with proper async handling"
   - "I implemented comprehensive safety controls including cost tracking and repository whitelisting"
   - "I built a Discord bot with slash commands that integrates with GitHub APIs"

3. **DevOps & Infrastructure**  
   - "I containerized the entire system with Docker and deployed it to my homelab UNRAID server"
   - "I set up monitoring with Prometheus and Grafana to track performance and costs"
   - "I implemented proper logging, error handling, and graceful degradation"

### Business Value Articulation
- **Problem Solved**: "Automated routine development tasks while maintaining code quality and security"
- **Results Achieved**: "Reduced feature implementation time by 60% while maintaining consistent code review standards"
- **Scalability**: "Architecture supports adding new agent types and integrating with different platforms"

### Demo Scenarios for Interviews
1. **Live Demo**: Show Discord commands creating GitHub branches and issues
2. **Architecture Walkthrough**: Explain RAG implementation and agent coordination
3. **Code Review**: Walk through the safety mechanisms and error handling
4. **Monitoring**: Show real-time metrics and cost tracking dashboard

---

## 🔧 Development Phases

### Phase 1: Core Infrastructure (Week 1)
- [ ] Gateway Service with Discord integration
- [ ] Basic CrewAI orchestrator setup
- [ ] RAG service with ChromaDB
- [ ] Simple agent pool with one agent type

### Phase 2: Agent Implementation (Week 2)  
- [ ] All four specialized agents
- [ ] GitHub integration and safety controls
- [ ] Cost tracking and rate limiting
- [ ] Basic monitoring setup

### Phase 3: Polish & Portfolio (Week 3)
- [ ] Comprehensive documentation
- [ ] Dashboard and metrics
- [ ] Demo scripts and test cases
- [ ] Performance optimization

### Phase 4: Deployment & Testing (Week 4)
- [ ] UNRAID deployment automation
- [ ] End-to-end testing
- [ ] Portfolio documentation
- [ ] Demo preparation

---

## 🤔 Why This Approach is Better for Portfolio

### Industry Relevance
- **CrewAI**: Recognized framework in AI engineering roles
- **RAG**: Most requested AI skill in 2024 job postings  
- **FastAPI**: Modern Python backend standard
- **Vector Databases**: Critical for AI applications

### Practical Complexity
- **Not Too Simple**: More than a basic chatbot
- **Not Too Complex**: Can be explained in 15 minutes
- **Real Integration**: Actually connects to Discord and GitHub
- **Production Concerns**: Shows understanding of cost, safety, monitoring

### Conversation Starters
- "Tell me about your RAG implementation"
- "How did you handle AI safety and cost control?"  
- "Walk me through the agent coordination system"
- "What challenges did you face with multi-model integration?"

### Differentiation
Most portfolio AI projects are:
- Simple chatbots with no real integration
- Jupyter notebooks with no deployment
- Toy projects with no safety considerations
- Single-model implementations

This project shows:
- **Production deployment** on real infrastructure
- **Multi-service architecture** with proper separation of concerns
- **Real-world integration** with external APIs
- **Safety and cost controls** for responsible AI

---

This redesign transforms Ambrosio from an over-engineered learning exercise into a portfolio-worthy demonstration of modern AI engineering skills, while actually being simpler to build and maintain. Perfect for showcasing in interviews and landing that AI engineer role! 🚀