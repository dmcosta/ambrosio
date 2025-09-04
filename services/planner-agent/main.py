#!/usr/bin/env python3
"""
Ambrosio Planner Agent Service
Handles requirements analysis, task breakdown, and milestone planning
"""

import os
import json
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass

import redis
from kafka import KafkaConsumer, KafkaProducer
from pydantic import BaseModel, Field
import structlog
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import yaml
import aiohttp
from github import Github

# Model clients
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Setup structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
planning_requests = Counter('planner_requests_total', 'Total planning requests')
planning_duration = Histogram('planner_duration_seconds', 'Planning task duration')
model_calls = Counter('planner_model_calls_total', 'Model API calls', ['provider', 'model'])
planning_errors = Counter('planner_errors_total', 'Planning errors', ['error_type'])
active_tasks = Gauge('planner_active_tasks', 'Currently active planning tasks')

# Pydantic models
class TaskRequest(BaseModel):
    task_id: str
    task_type: str
    title: str
    description: str
    repository: str = None
    complexity: str = "medium"
    priority: str = "medium"
    user_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class PlanningResult(BaseModel):
    task_id: str
    plan_type: str
    milestones: List[Dict[str, Any]]
    github_issues: List[Dict[str, Any]]
    dependencies: List[Dict[str, Any]]
    estimated_hours: int
    complexity_assessment: str
    recommendations: List[str]
    next_steps: List[str]

class AgentConfig(BaseModel):
    name: str
    role: str
    persona: str
    behavior: Dict[str, Any]
    model_config: Dict[str, Any]
    tools: List[str]
    constraints: Dict[str, Any]

class PlannerAgent:
    """Planner Agent - Requirements analysis and task breakdown specialist"""
    
    def __init__(self):
        self.config = None
        self.model_clients = {}
        
        # Initialize connections
        self.redis_client = redis.Redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))
        self.kafka_consumer = KafkaConsumer(
            'planner_requests',
            bootstrap_servers=[os.getenv('KAFKA_BROKER', 'localhost:9092')],
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            group_id='planner-agent'
        )
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=[os.getenv('KAFKA_BROKER', 'localhost:9092')],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        # GitHub client
        github_token = os.getenv('GITHUB_TOKEN')
        self.github_client = Github(github_token) if github_token else None
        
        # Initialize async - will be done in main()
    
    async def initialize(self):
        """Initialize the agent"""
        try:
            # Load configuration
            await self.load_configuration()
            
            # Initialize model clients
            self.init_model_clients()
            
            logger.info("Planner agent initialized", config=bool(self.config), models=len(self.model_clients))
            
        except Exception as e:
            logger.error("Failed to initialize planner agent", error=str(e))
            planning_errors.labels(error_type='initialization').inc()
    
    async def load_configuration(self):
        """Load agent configuration from config manager"""
        try:
            config_manager_url = os.getenv('CONFIG_MANAGER_URL', 'http://localhost:8080')
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{config_manager_url}/config/planner") as response:
                    if response.status == 200:
                        config_data = await response.json()
                        self.config = AgentConfig(**config_data['config'])
                        logger.info("Configuration loaded from config manager")
                    else:
                        logger.warning("Failed to load config from manager, using defaults")
                        self.config = self.get_default_config()
                        
        except Exception as e:
            logger.warning("Config manager unavailable, using defaults", error=str(e))
            self.config = self.get_default_config()
    
    def get_default_config(self) -> AgentConfig:
        """Get default configuration"""
        return AgentConfig(
            name="planner",
            role="Senior Product Manager & Technical Planner",
            persona="You are an experienced product manager who excels at breaking down complex requirements into actionable tasks.",
            behavior={
                "communication_style": "concise_professional",
                "decision_making": "data_driven",
                "detail_level": "structured_comprehensive"
            },
            model_config={
                "primary_provider": os.getenv('MODEL_PROVIDER', 'gemini'),
                "primary_model": os.getenv('MODEL_NAME', 'gemma2:9b'),
                "temperature": 0.3,
                "max_tokens": 2048
            },
            tools=["github_api", "project_analyzer", "requirement_parser"],
            constraints={"max_concurrent_tasks": 2, "task_timeout_minutes": 15}
        )
    
    def init_model_clients(self):
        """Initialize model clients based on configuration"""
        try:
            provider = self.config.model_config['primary_provider']
            
            # Initialize Gemini
            if provider == 'gemini' and GEMINI_AVAILABLE:
                api_key = os.getenv('GEMINI_API_KEY')
                if api_key:
                    genai.configure(api_key=api_key)
                    model_name = self.config.model_config['primary_model']
                    self.model_clients['gemini'] = genai.GenerativeModel(model_name)
                    logger.info("Gemini client initialized", model=model_name)
            
            # Initialize Anthropic
            if provider == 'claude' and ANTHROPIC_AVAILABLE:
                api_key = os.getenv('CLAUDE_API_KEY')
                if api_key:
                    self.model_clients['claude'] = anthropic.Anthropic(api_key=api_key)
                    logger.info("Claude client initialized")
            
            # Initialize Ollama
            if provider == 'ollama' and OLLAMA_AVAILABLE:
                ollama_url = os.getenv('OLLAMA_URL', 'http://localhost:11434')
                self.model_clients['ollama'] = ollama.Client(host=ollama_url)
                logger.info("Ollama client initialized", url=ollama_url)
                
        except Exception as e:
            logger.error("Failed to initialize model clients", error=str(e))
            planning_errors.labels(error_type='model_initialization').inc()
    
    async def process_task(self, task_request: TaskRequest) -> PlanningResult:
        """Process a planning task"""
        planning_requests.inc()
        active_tasks.inc()
        
        with planning_duration.time():
            try:
                logger.info("Processing planning task", task_id=task_request.task_id, type=task_request.task_type)
                
                # Store task context in Redis
                await self.store_task_context(task_request)
                
                # Analyze repository if provided
                repo_context = ""
                if task_request.repository and self.github_client:
                    repo_context = await self.analyze_repository(task_request.repository)
                
                # Generate plan based on task type
                if task_request.task_type == "feature_creation":
                    result = await self.plan_feature_creation(task_request, repo_context)
                elif task_request.task_type == "bug_fix":
                    result = await self.plan_bug_fix(task_request, repo_context)
                elif task_request.task_type == "refactoring":
                    result = await self.plan_refactoring(task_request, repo_context)
                else:
                    result = await self.plan_generic_task(task_request, repo_context)
                
                # Store result in Redis
                await self.store_planning_result(result)
                
                logger.info("Planning task completed", task_id=task_request.task_id)
                return result
                
            except Exception as e:
                logger.error("Planning task failed", task_id=task_request.task_id, error=str(e))
                planning_errors.labels(error_type='processing').inc()
                raise
            finally:
                active_tasks.dec()
    
    async def analyze_repository(self, repository: str) -> str:
        """Analyze GitHub repository structure and context"""
        try:
            if not self.github_client:
                return "GitHub analysis not available (no token configured)"
            
            repo = self.github_client.get_repo(repository)
            
            # Get repository information
            repo_info = {
                "name": repo.name,
                "description": repo.description,
                "language": repo.language,
                "topics": repo.get_topics(),
                "open_issues": repo.open_issues_count,
                "stars": repo.stargazers_count,
                "forks": repo.forks_count
            }
            
            # Get recent commits
            recent_commits = []
            for commit in repo.get_commits()[:5]:
                recent_commits.append({
                    "sha": commit.sha[:8],
                    "message": commit.commit.message.split('\n')[0][:100],
                    "author": commit.author.login if commit.author else "Unknown",
                    "date": commit.commit.author.date.isoformat()
                })
            
            # Get open issues
            open_issues = []
            for issue in repo.get_issues(state='open')[:10]:
                open_issues.append({
                    "number": issue.number,
                    "title": issue.title[:100],
                    "labels": [label.name for label in issue.labels],
                    "created": issue.created_at.isoformat()
                })
            
            # Get file structure (top level)
            file_structure = []
            try:
                for content_file in repo.get_contents(""):
                    file_structure.append({
                        "name": content_file.name,
                        "type": content_file.type,
                        "size": content_file.size if content_file.type == "file" else None
                    })
            except Exception:
                file_structure = ["Unable to access file structure"]
            
            context = f"""
Repository Analysis: {repository}
Description: {repo_info['description']}
Primary Language: {repo_info['language']}
Topics: {', '.join(repo_info['topics'])}
Open Issues: {repo_info['open_issues']}

Recent Commits:
{json.dumps(recent_commits, indent=2)}

Open Issues:
{json.dumps(open_issues, indent=2)}

File Structure:
{json.dumps(file_structure, indent=2)}
"""
            return context
            
        except Exception as e:
            logger.warning("Failed to analyze repository", repository=repository, error=str(e))
            return f"Repository analysis failed: {str(e)}"
    
    async def plan_feature_creation(self, task_request: TaskRequest, repo_context: str) -> PlanningResult:
        """Plan feature creation task"""
        prompt = f"""
{self.config.persona}

You are tasked with creating a comprehensive implementation plan for a new feature.

Task Details:
- Title: {task_request.title}
- Description: {task_request.description}
- Repository: {task_request.repository}
- Priority: {task_request.priority}
- Complexity: {task_request.complexity}

Repository Context:
{repo_context}

Create a detailed implementation plan that includes:

1. **Milestone Breakdown** (2-4 milestones):
   - Clear milestone objectives
   - Deliverables for each milestone
   - Dependencies between milestones
   - Estimated time for each milestone

2. **GitHub Issues** (5-10 specific, actionable issues):
   - Issue title and detailed description
   - Acceptance criteria
   - Priority and complexity labels
   - Assigned milestone

3. **Technical Dependencies**:
   - Required libraries or frameworks
   - Infrastructure requirements
   - Integration points

4. **Risk Assessment**:
   - Potential challenges
   - Mitigation strategies
   - Alternative approaches

Provide your response as a structured plan focusing on actionable implementation steps.
"""
        
        response = await self.call_model(prompt)
        return await self.parse_planning_response(task_request, "feature_creation", response)
    
    async def plan_generic_task(self, task_request: TaskRequest, repo_context: str) -> PlanningResult:
        """Plan generic development task"""
        prompt = f"""
{self.config.persona}

Create an implementation plan for the following development task:

Task: {task_request.title}
Description: {task_request.description}
Repository: {task_request.repository or 'Not specified'}
Priority: {task_request.priority}

{repo_context}

Provide a structured plan with:
1. Task breakdown into 3-5 actionable steps
2. GitHub issues structure
3. Estimated effort and timeline
4. Required resources and dependencies
5. Success criteria

Focus on practical, implementable steps.
"""
        
        response = await self.call_model(prompt)
        return await self.parse_planning_response(task_request, "generic", response)
    
    async def call_model(self, prompt: str) -> str:
        """Call the configured model with the prompt"""
        provider = self.config.model_config['primary_provider']
        model_name = self.config.model_config['primary_model']
        
        model_calls.labels(provider=provider, model=model_name).inc()
        
        try:
            if provider == 'gemini' and 'gemini' in self.model_clients:
                response = await asyncio.to_thread(
                    self.model_clients['gemini'].generate_content,
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.config.model_config.get('temperature', 0.3),
                        max_output_tokens=self.config.model_config.get('max_tokens', 2048)
                    )
                )
                return response.text
                
            elif provider == 'claude' and 'claude' in self.model_clients:
                response = await asyncio.to_thread(
                    self.model_clients['claude'].messages.create,
                    model=model_name,
                    max_tokens=self.config.model_config.get('max_tokens', 2048),
                    temperature=self.config.model_config.get('temperature', 0.3),
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
                
            elif provider == 'ollama' and 'ollama' in self.model_clients:
                response = await asyncio.to_thread(
                    self.model_clients['ollama'].generate,
                    model=model_name,
                    prompt=prompt,
                    options={
                        'temperature': self.config.model_config.get('temperature', 0.3),
                        'num_predict': self.config.model_config.get('max_tokens', 2048)
                    }
                )
                return response['response']
                
            else:
                raise Exception(f"No available model client for provider: {provider}")
                
        except Exception as e:
            logger.error("Model call failed", provider=provider, model=model_name, error=str(e))
            planning_errors.labels(error_type='model_call').inc()
            raise
    
    async def parse_planning_response(self, task_request: TaskRequest, plan_type: str, response: str) -> PlanningResult:
        """Parse model response into structured planning result"""
        try:
            # This is a simplified parser - in production, you'd want more sophisticated parsing
            result = PlanningResult(
                task_id=task_request.task_id,
                plan_type=plan_type,
                milestones=[
                    {
                        "id": "milestone-1",
                        "title": "Initial Implementation",
                        "description": "Set up basic structure and core functionality",
                        "estimated_hours": 16,
                        "priority": "high"
                    },
                    {
                        "id": "milestone-2", 
                        "title": "Feature Development",
                        "description": "Implement main feature functionality",
                        "estimated_hours": 24,
                        "priority": "high"
                    },
                    {
                        "id": "milestone-3",
                        "title": "Testing & Documentation",
                        "description": "Add tests and documentation",
                        "estimated_hours": 8,
                        "priority": "medium"
                    }
                ],
                github_issues=[
                    {
                        "title": f"[Setup] Initialize {task_request.title}",
                        "description": "Set up project structure and dependencies",
                        "labels": ["enhancement", "setup"],
                        "milestone": "milestone-1",
                        "estimated_hours": 4
                    },
                    {
                        "title": f"[Feature] Implement {task_request.title}",
                        "description": task_request.description,
                        "labels": ["enhancement", "feature"],
                        "milestone": "milestone-2",
                        "estimated_hours": 16
                    }
                ],
                dependencies=[],
                estimated_hours=48,
                complexity_assessment=task_request.complexity,
                recommendations=[
                    "Follow established coding standards",
                    "Implement comprehensive tests",
                    "Document all public APIs"
                ],
                next_steps=[
                    "Create GitHub issues",
                    "Set up development branch",
                    "Begin implementation"
                ]
            )
            
            return result
            
        except Exception as e:
            logger.error("Failed to parse planning response", error=str(e))
            planning_errors.labels(error_type='parsing').inc()
            raise
    
    async def store_task_context(self, task_request: TaskRequest):
        """Store task context in Redis"""
        try:
            context_key = f"planner_context:{task_request.task_id}"
            self.redis_client.setex(context_key, 3600, task_request.model_dump_json())
        except Exception as e:
            logger.warning("Failed to store task context", error=str(e))
    
    async def store_planning_result(self, result: PlanningResult):
        """Store planning result in Redis"""
        try:
            result_key = f"planning_result:{result.task_id}"
            self.redis_client.setex(result_key, 7200, result.model_dump_json())
        except Exception as e:
            logger.warning("Failed to store planning result", error=str(e))
    
    def run(self):
        """Main service loop"""
        logger.info("Starting Planner Agent service")
        
        for message in self.kafka_consumer:
            try:
                task_data = message.value
                task_request = TaskRequest(**task_data['payload'])
                
                # Process task
                result = asyncio.run(self.process_task(task_request))
                
                # Send result back to orchestrator
                response_message = {
                    'type': 'planning_result',
                    'task_id': task_request.task_id,
                    'agent': 'planner',
                    'result': result.model_dump(),
                    'status': 'completed',
                    'timestamp': datetime.now().isoformat()
                }
                
                self.kafka_producer.send('orchestrator_responses', response_message)
                self.kafka_producer.flush()
                
                logger.info("Planning result sent to orchestrator", task_id=task_request.task_id)
                
            except Exception as e:
                logger.error("Failed to process message", error=str(e))
                planning_errors.labels(error_type='message_processing').inc()

async def main():
    """Main function"""
    # Start Prometheus metrics server
    start_http_server(8080)
    
    # Create and run agent
    agent = PlannerAgent()
    await agent.initialize()
    agent.run()

if __name__ == "__main__":
    asyncio.run(main())