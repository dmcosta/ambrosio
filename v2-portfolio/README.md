# Ambrosio v2 - Portfolio Edition 
## AI Agent System with RAG & CrewAI

**🎯 Purpose**: Showcase modern AI engineering skills with practical agent orchestration, RAG implementation, and real-world Discord/GitHub integration.

**🔥 Tech Stack**: CrewAI • ChromaDB • FastAPI • LangChain • Vector Embeddings • OpenAI/Claude APIs

---

## 🏗️ Simplified Architecture (Portfolio-Ready)

```
Discord Commands → API Gateway → CrewAI Crew → Specialized Agents → GitHub Actions
                      ↓              ↓              ↓              ↓
                 Route & Auth → Orchestrate → Execute Tasks → Create Artifacts
                                      ↓
                              ChromaDB RAG System ← Repository Knowledge
```

### Core Services (4 instead of 13!)

1. **API Gateway** (`gateway-service/`)
   - FastAPI with async endpoints
   - Discord webhook integration  
   - Request routing and authentication
   - Cost tracking and rate limiting

2. **CrewAI Orchestrator** (`crew-service/`)
   - CrewAI-powered agent coordination
   - Dynamic crew assembly based on task complexity
   - Workflow state management
   - Agent communication protocols

3. **RAG Knowledge System** (`rag-service/`)  
   - ChromaDB vector database
   - Repository indexing and embedding
   - Semantic code search
   - Context retrieval for agents

4. **Agent Pool** (`agents-service/`)
   - Specialized AI agents with tools
   - GitHub API integration
   - Code generation and review
   - Multi-model support (GPT-4, Claude, local models)

---

## 🤖 Agent Specializations (CrewAI Powered)

### **Senior Developer Agent**
- **Role**: Code architecture and implementation
- **Tools**: GitHub API, code analysis, testing frameworks
- **Models**: GPT-4 for complex logic, Claude for code review
- **Backstory**: 10+ years experience, focuses on maintainable code

### **DevOps Engineer Agent**  
- **Role**: CI/CD, deployment, infrastructure
- **Tools**: Docker, GitHub Actions, deployment scripts
- **Models**: GPT-4 for complex automation
- **Backstory**: Infrastructure expert, automates everything

### **Technical Writer Agent**
- **Role**: Documentation, README files, code comments
- **Tools**: Markdown generation, API documentation
- **Models**: GPT-3.5 for efficiency, Claude for technical accuracy
- **Backstory**: Makes complex tech accessible

### **Code Reviewer Agent**
- **Role**: Security, performance, best practices
- **Tools**: Static analysis, security scanners, performance profilers  
- **Models**: Claude for detailed analysis
- **Backstory**: Security-focused, catches edge cases

---

## 🧠 RAG Implementation (ChromaDB)

### Knowledge Sources
```python
class KnowledgeIndexer:
    def index_repository(self, repo_url):
        # 1. Clone and parse codebase
        files = self.extract_code_files(repo_url)
        
        # 2. Create semantic chunks
        chunks = self.chunk_code_intelligently(files)
        
        # 3. Generate embeddings
        embeddings = self.embed_with_context(chunks)
        
        # 4. Store in ChromaDB
        self.chroma_client.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=[{"file": f.path, "type": f.language}]
        )
```

### Context Retrieval
```python
class RAGRetriever:
    def get_relevant_context(self, query, k=5):
        # Semantic search for relevant code
        results = self.chroma_client.query(
            query_texts=[query],
            n_results=k,
            include=['documents', 'metadatas']
        )
        return self.format_context(results)
```

---

## 🔄 Workflow Examples

### Feature Creation Flow
```python
@crew.task
def analyze_feature_request(agent, task):
    # 1. RAG: Get similar implementations
    context = rag_service.search("user authentication patterns")
    
    # 2. Agent: Analyze requirements  
    analysis = agent.analyze_with_context(task.description, context)
    
    return FeatureAnalysis(
        requirements=analysis.requirements,
        architecture=analysis.suggested_architecture,
        similar_implementations=context.examples
    )

@crew.task  
def implement_feature(agent, analysis):
    # 1. RAG: Get relevant code patterns
    patterns = rag_service.search(f"implement {analysis.requirements}")
    
    # 2. Agent: Generate code
    code = agent.generate_code(analysis, patterns)
    
    # 3. Create GitHub branch and files
    return agent.create_github_artifacts(code)
```

### Code Review Flow
```python
@crew.task
def review_pull_request(agent, pr_url):
    # 1. Get PR diff from GitHub
    diff = github_client.get_pr_diff(pr_url)
    
    # 2. RAG: Find similar code patterns
    similar_code = rag_service.search_similar_code(diff)
    
    # 3. Agent: Comprehensive review
    review = agent.analyze_code_with_context(diff, similar_code)
    
    # 4. Post review as GitHub comments
    return agent.post_github_review(review)
```

---

## 🛡️ Safety & Cost Controls

### Built-in Safeguards
```python
class SafetyManager:
    def __init__(self):
        self.daily_api_cost = 0
        self.github_actions_today = 0
        self.max_daily_cost = 25  # $25/day limit
        self.max_github_actions = 50
    
    def check_limits(self, operation):
        if self.daily_api_cost > self.max_daily_cost:
            raise CostLimitExceeded()
        
        if operation.type == "github" and self.github_actions_today > self.max_github_actions:
            raise GitHubLimitExceeded()
    
    def track_cost(self, model_call):
        estimated_cost = self.calculate_token_cost(model_call)
        self.daily_api_cost += estimated_cost
```

### Repository Safety
```python
class GitHubSafetyWrapper:
    def __init__(self, token, allowed_repos):
        self.client = Github(token)
        self.allowed_repos = allowed_repos  # Whitelist only
    
    def create_branch(self, repo_name, branch_name):
        if repo_name not in self.allowed_repos:
            raise UnauthorizedRepository()
        
        if self.count_branches_today(repo_name) > 10:
            raise TooManyBranches()
        
        return self.client.create_branch(repo_name, branch_name)
```

---

## 📊 Portfolio Highlights

### **Modern AI Engineering**
- ✅ **CrewAI Integration**: Industry-standard agent orchestration
- ✅ **RAG Implementation**: Semantic search with ChromaDB  
- ✅ **Multi-Model Support**: GPT-4, Claude, local models
- ✅ **Vector Embeddings**: Proper context retrieval

### **Production Practices**  
- ✅ **FastAPI**: Async, typed, documented APIs
- ✅ **Monitoring**: Prometheus metrics, structured logging
- ✅ **Safety Controls**: Cost limits, repository whitelisting
- ✅ **Error Handling**: Graceful degradation, retry logic

### **Real-World Integration**
- ✅ **Discord Bots**: Slash commands, interactive responses
- ✅ **GitHub API**: Branch creation, PR reviews, issue management
- ✅ **Docker Deployment**: Container orchestration on UNRAID
- ✅ **CI/CD Pipeline**: Automated testing and deployment

---

## 🚀 Getting Started

### 1. Setup (5 minutes)
```bash
cd ~/Workspaces/Personal/Ambrosio/v2-portfolio
cp .env.example .env
# Add your API keys
./scripts/deploy-simple.sh
```

### 2. Try It Out
```
# Discord Commands
/analyze-repo repository:"your-org/your-repo" 
/create-feature description:"Add user authentication"
/review-pr url:"https://github.com/owner/repo/pull/123"
/ask question:"How do I implement OAuth in this codebase?"
```

### 3. Monitor & Learn
- **Dashboard**: http://your-server:3000
- **API Docs**: http://your-server:8000/docs  
- **RAG Search**: http://your-server:8001/search

---

## 📈 Learning Outcomes

Building this project demonstrates mastery of:

1. **AI Agent Orchestration** (CrewAI, multi-agent systems)
2. **RAG Systems** (vector databases, semantic search, context retrieval)
3. **API Design** (FastAPI, async programming, webhooks)  
4. **LLM Integration** (multiple providers, prompt engineering, cost management)
5. **DevOps Practices** (Docker, monitoring, deployment automation)
6. **Real-world Integration** (Discord, GitHub, external APIs)

Perfect for showcasing in interviews for:
- **AI Engineer** roles
- **Backend Developer** positions  
- **DevOps Engineer** roles
- **Full-stack Developer** with AI focus

---

## 🎯 Portfolio Story

*"I built Ambrosio to showcase modern AI engineering practices. It's a practical system that real development teams could use, featuring proper RAG implementation with ChromaDB for codebase knowledge, CrewAI for agent orchestration, and real GitHub integration. The system can analyze repositories, generate code, review pull requests, and maintain context across conversations - all while staying within cost and safety limits I designed."*

**Key differentiator**: Not just a toy project, but a functional system with production concerns (monitoring, safety, cost control) that demonstrates both AI/ML engineering and traditional software engineering skills.

---

Ready to build this simplified, portfolio-worthy version? 🚀