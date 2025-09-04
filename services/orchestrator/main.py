#!/usr/bin/env python3
"""
Ambrosio Orchestrator Service
Coordinates workflows between agents and manages GitHub integration
"""

import os
import json
import asyncio
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

import redis
import psycopg2
from github import Github
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import structlog
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import uvicorn

# Setup logging
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

# Prometheus metrics - clear registry first to avoid duplicates
from prometheus_client import REGISTRY
REGISTRY._collector_to_names.clear()
REGISTRY._names_to_collectors.clear()

workflow_requests = Counter('orchestrator_workflows_total', 'Total workflow requests', ['workflow_type'])
workflow_duration = Histogram('orchestrator_workflow_duration_seconds', 'Workflow execution duration')
active_workflows = Gauge('orchestrator_active_workflows', 'Currently active workflows')
github_operations = Counter('orchestrator_github_ops_total', 'GitHub operations', ['operation_type', 'status'])

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class AgentQueue:
    """Queue for managing agent tasks with concurrency control"""
    
    def __init__(self, agent_name: str, max_concurrent: int = 1):
        self.agent_name = agent_name
        self.max_concurrent = max_concurrent
        self.queue = asyncio.Queue()
        self.active_tasks = set()
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self._worker_task = None
        self.redis_client = None
        
    async def initialize(self, redis_client):
        """Initialize the queue with Redis client and start worker"""
        self.redis_client = redis_client
        self._worker_task = asyncio.create_task(self._worker())
        logger.info("Agent queue initialized", agent=self.agent_name, max_concurrent=self.max_concurrent)
    
    async def add_task(self, task_data: Dict[str, Any]):
        """Add task to queue"""
        queue_position = self.queue.qsize() + 1
        task_id = task_data.get('task_id')
        
        logger.info("Task queued for agent", 
                   agent=self.agent_name, 
                   task_id=task_id,
                   queue_position=queue_position)
        
        # Notify Discord about queue position
        await self._notify_discord_progress(task_id, f"Queued for {self.agent_name} (position {queue_position})")
        
        await self.queue.put(task_data)
    
    async def _worker(self):
        """Background worker that processes queued tasks"""
        while True:
            try:
                # Wait for a task
                task_data = await self.queue.get()
                
                # Wait for semaphore (concurrency control)
                await self.semaphore.acquire()
                
                # Process task in background
                asyncio.create_task(self._process_task(task_data))
                
            except Exception as e:
                logger.error("Agent queue worker error", agent=self.agent_name, error=str(e))
                await asyncio.sleep(1)
    
    async def _process_task(self, task_data: Dict[str, Any]):
        """Process individual task"""
        task_id = task_data.get('task_id')
        
        try:
            self.active_tasks.add(task_id)
            
            await self._notify_discord_progress(task_id, f"{self.agent_name} started processing")
            
            # Simulate agent processing (replace with actual agent call)
            logger.info("Agent processing task", agent=self.agent_name, task_id=task_id)
            await asyncio.sleep(10)  # Simulate work
            
            result = {
                'task_id': task_id,
                'status': 'completed',
                'result': f'Simulated result from {self.agent_name}',
                'agent': self.agent_name,
                'timestamp': datetime.now().isoformat()
            }
            
            # Publish result
            if self.redis_client:
                self.redis_client.publish('agent_results', json.dumps(result))
            
            await self._notify_discord_progress(task_id, f"{self.agent_name} completed successfully")
            
            logger.info("Agent task completed", agent=self.agent_name, task_id=task_id)
            
        except Exception as e:
            logger.error("Agent task failed", agent=self.agent_name, task_id=task_id, error=str(e))
            
            # Publish failure result
            result = {
                'task_id': task_id,
                'status': 'failed',
                'error': str(e),
                'agent': self.agent_name,
                'timestamp': datetime.now().isoformat()
            }
            
            if self.redis_client:
                self.redis_client.publish('agent_results', json.dumps(result))
                
        finally:
            self.active_tasks.discard(task_id)
            self.semaphore.release()
    
    async def _notify_discord_progress(self, task_id: str, message: str):
        """Send progress update to Discord"""
        if self.redis_client:
            notification = {
                'type': 'progress_update',
                'task_id': task_id,
                'message': message,
                'timestamp': datetime.now().isoformat()
            }
            self.redis_client.publish('discord_notifications', json.dumps(notification))
    
    def get_status(self) -> Dict[str, Any]:
        """Get queue status"""
        return {
            'agent': self.agent_name,
            'queue_size': self.queue.qsize(),
            'active_tasks': len(self.active_tasks),
            'max_concurrent': self.max_concurrent
        }

class WorkflowStep(BaseModel):
    step_id: str
    agent_name: str
    task_type: str
    status: TaskStatus = TaskStatus.PENDING
    input_data: Dict[str, Any] = {}
    output_data: Dict[str, Any] = {}
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

class Workflow(BaseModel):
    workflow_id: str
    workflow_type: str
    status: TaskStatus = TaskStatus.PENDING
    user_id: str
    repository: Optional[str] = None
    title: str
    description: str
    steps: List[WorkflowStep] = []
    metadata: Dict[str, Any] = {}
    created_at: datetime
    updated_at: datetime

class OrchestoratorService:
    """Main orchestrator service that coordinates AI agents"""
    
    def __init__(self):
        self.redis_client = redis.Redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))
        
        # GitHub integration
        github_token = os.getenv('GITHUB_TOKEN')
        self.github_client = Github(github_token) if github_token else None
        
        # Agent queues with concurrency limits
        self.agent_queues = {
            'planner': AgentQueue('planner', max_concurrent=1),
            'coder': AgentQueue('coder', max_concurrent=1), 
            'reviewer': AgentQueue('reviewer', max_concurrent=2)  # Reviews can be more concurrent
        }
        
        # Active workflows tracking
        self.active_workflows_dict = {}
        
        # Redis pub/sub
        self.pubsub = None
    
    async def initialize(self):
        """Initialize agent queues and Redis pub/sub"""
        # Initialize agent queues
        for queue in self.agent_queues.values():
            await queue.initialize(self.redis_client)
        
        # Initialize Redis pub/sub
        self.pubsub = self.redis_client.pubsub()
        self.pubsub.subscribe('orchestrator_requests', 'agent_results')
        
        logger.info("Orchestrator initialized with agent queues and Redis pub/sub")
        
    async def process_task_request(self, request_data: Dict[str, Any]) -> str:
        """Process incoming task request and create workflow"""
        workflow_requests.labels(workflow_type=request_data.get('task_type', 'unknown')).inc()
        active_workflows.inc()
        
        workflow_id = str(uuid.uuid4())
        
        try:
            # Create workflow based on task type
            workflow = await self.create_workflow(workflow_id, request_data)
            
            # Store workflow
            self.active_workflows_dict[workflow_id] = workflow
            await self.store_workflow(workflow)
            
            # Start workflow execution
            await self.execute_workflow(workflow)
            
            logger.info("Workflow created and started", workflow_id=workflow_id, type=workflow.workflow_type)
            return workflow_id
            
        except Exception as e:
            logger.error("Failed to process task request", error=str(e))
            active_workflows.dec()
            raise
    
    async def create_workflow(self, workflow_id: str, request_data: Dict[str, Any]) -> Workflow:
        """Create workflow with appropriate steps based on task type"""
        task_type = request_data.get('task_type', 'generic')
        
        workflow = Workflow(
            workflow_id=workflow_id,
            workflow_type=task_type,
            user_id=request_data.get('user_id', 'unknown'),
            repository=request_data.get('repository'),
            title=request_data.get('title', 'Untitled Task'),
            description=request_data.get('description', ''),
            metadata=request_data.get('metadata', {}),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Define workflow steps based on task type
        if task_type == "feature_creation":
            workflow.steps = [
                WorkflowStep(
                    step_id=f"{workflow_id}-planning",
                    agent_name="planner",
                    task_type="milestone_planning",
                    input_data=request_data
                ),
                WorkflowStep(
                    step_id=f"{workflow_id}-implementation", 
                    agent_name="coder",
                    task_type="code_generation",
                    input_data={}  # Will be populated with planner output
                ),
                WorkflowStep(
                    step_id=f"{workflow_id}-review",
                    agent_name="reviewer", 
                    task_type="code_review",
                    input_data={}  # Will be populated with coder output
                )
            ]
        elif task_type == "code_review":
            workflow.steps = [
                WorkflowStep(
                    step_id=f"{workflow_id}-review",
                    agent_name="reviewer",
                    task_type="pr_review",
                    input_data=request_data
                )
            ]
        else:
            # Generic workflow
            workflow.steps = [
                WorkflowStep(
                    step_id=f"{workflow_id}-planning",
                    agent_name="planner",
                    task_type="generic_planning", 
                    input_data=request_data
                )
            ]
        
        return workflow
    
    async def execute_workflow(self, workflow: Workflow):
        """Execute workflow steps sequentially"""
        logger.info("Starting workflow execution", workflow_id=workflow.workflow_id)
        
        workflow.status = TaskStatus.IN_PROGRESS
        await self.update_workflow(workflow)
        
        try:
            for step in workflow.steps:
                await self.execute_step(workflow, step)
                
                if step.status == TaskStatus.FAILED:
                    workflow.status = TaskStatus.FAILED
                    await self.update_workflow(workflow)
                    return
            
            # All steps completed successfully
            workflow.status = TaskStatus.COMPLETED
            await self.finalize_workflow(workflow)
            
        except Exception as e:
            logger.error("Workflow execution failed", workflow_id=workflow.workflow_id, error=str(e))
            workflow.status = TaskStatus.FAILED
            await self.update_workflow(workflow)
        finally:
            active_workflows.dec()
    
    async def execute_step(self, workflow: Workflow, step: WorkflowStep):
        """Execute individual workflow step"""
        logger.info("Executing step", workflow_id=workflow.workflow_id, step=step.step_id, agent=step.agent_name)
        
        step.status = TaskStatus.IN_PROGRESS
        step.started_at = datetime.now()
        await self.update_workflow(workflow)
        
        # Send task to agent queue
        task_data = {
            'task_id': step.step_id,
            'workflow_id': workflow.workflow_id,
            'task_type': step.task_type,
            'payload': step.input_data,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add to appropriate agent queue
        agent_queue = self.agent_queues.get(step.agent_name)
        if agent_queue:
            await agent_queue.add_task(task_data)
        else:
            raise ValueError(f"Unknown agent: {step.agent_name}")
        
        # Wait for agent response
        response = await self.wait_for_agent_response(step.step_id)
        
        if response and response.get('status') == 'completed':
            step.status = TaskStatus.COMPLETED
            step.output_data = response.get('result', {})
            step.completed_at = datetime.now()
            
            # Pass output to next step if available
            await self.propagate_step_output(workflow, step)
            
        else:
            step.status = TaskStatus.FAILED
            step.error_message = response.get('error', 'Unknown error') if response else 'No response from agent'
            step.completed_at = datetime.now()
        
        await self.update_workflow(workflow)
    
    async def wait_for_agent_response(self, step_id: str, timeout_seconds: int = 300) -> Optional[Dict[str, Any]]:
        """Wait for agent response via Redis pub/sub"""
        start_time = asyncio.get_event_loop().time()
        
        while (asyncio.get_event_loop().time() - start_time) < timeout_seconds:
            try:
                # Check for messages in pub/sub
                message = self.pubsub.get_message(timeout=1.0)
                if message and message['type'] == 'message':
                    data = json.loads(message['data'].decode('utf-8'))
                    
                    # Check if this is the response we're waiting for
                    if data.get('task_id') == step_id:
                        return data
                
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error("Error waiting for agent response", step_id=step_id, error=str(e))
                await asyncio.sleep(1)
        
        # Timeout
        logger.warning("Timeout waiting for agent response", step_id=step_id)
        return None
    
    async def propagate_step_output(self, workflow: Workflow, completed_step: WorkflowStep):
        """Pass output from completed step to next steps"""
        current_index = workflow.steps.index(completed_step)
        
        if current_index < len(workflow.steps) - 1:
            next_step = workflow.steps[current_index + 1]
            
            # Combine original input with previous step output
            next_step.input_data.update({
                'previous_step_output': completed_step.output_data,
                'workflow_context': {
                    'workflow_id': workflow.workflow_id,
                    'repository': workflow.repository,
                    'title': workflow.title,
                    'description': workflow.description
                }
            })
    
    async def finalize_workflow(self, workflow: Workflow):
        """Finalize completed workflow and create GitHub artifacts"""
        logger.info("Finalizing workflow", workflow_id=workflow.workflow_id)
        
        if workflow.workflow_type == "feature_creation" and self.github_client and workflow.repository:
            await self.create_github_artifacts(workflow)
        
        # Send completion notification
        await self.send_completion_notification(workflow)
        
        workflow.updated_at = datetime.now()
        await self.update_workflow(workflow)
        
        logger.info("Workflow finalized", workflow_id=workflow.workflow_id)
    
    async def create_github_artifacts(self, workflow: Workflow):
        """Create GitHub issues, branches, etc."""
        try:
            repo = self.github_client.get_repo(workflow.repository)
            
            # Create feature branch
            main_branch = repo.get_branch("main")
            branch_name = f"feature/{workflow.workflow_id[:8]}-{workflow.title.lower().replace(' ', '-')}"
            
            try:
                repo.create_git_ref(
                    ref=f"refs/heads/{branch_name}",
                    sha=main_branch.commit.sha
                )
                github_operations.labels(operation_type='create_branch', status='success').inc()
                logger.info("Branch created", repository=workflow.repository, branch=branch_name)
            except Exception as e:
                github_operations.labels(operation_type='create_branch', status='failed').inc()
                logger.warning("Failed to create branch", error=str(e))
            
            # Create GitHub issue
            issue_body = f"""
## Feature Request

{workflow.description}

### Workflow Details
- **Workflow ID**: `{workflow.workflow_id}`
- **Created**: {workflow.created_at.isoformat()}
- **Requested by**: {workflow.user_id}

### Implementation Plan
This issue was created automatically by Ambrosio AI Development Team.

Branch: `{branch_name}`
"""
            
            try:
                issue = repo.create_issue(
                    title=workflow.title,
                    body=issue_body,
                    labels=['enhancement', 'ai-generated']
                )
                github_operations.labels(operation_type='create_issue', status='success').inc()
                logger.info("Issue created", repository=workflow.repository, issue_number=issue.number)
            except Exception as e:
                github_operations.labels(operation_type='create_issue', status='failed').inc()
                logger.warning("Failed to create issue", error=str(e))
                
        except Exception as e:
            logger.error("Failed to create GitHub artifacts", workflow_id=workflow.workflow_id, error=str(e))
    
    async def send_completion_notification(self, workflow: Workflow):
        """Send workflow completion notification back to Discord"""
        notification = {
            'type': 'workflow_completed',
            'workflow_id': workflow.workflow_id,
            'status': workflow.status.value,
            'title': workflow.title,
            'repository': workflow.repository,
            'user_id': workflow.user_id,
            'steps_completed': len([s for s in workflow.steps if s.status == TaskStatus.COMPLETED]),
            'total_steps': len(workflow.steps),
            'timestamp': datetime.now().isoformat()
        }
        
        self.redis_client.publish('discord_notifications', json.dumps(notification))
    
    async def store_workflow(self, workflow: Workflow):
        """Store workflow in Redis"""
        workflow_key = f"workflow:{workflow.workflow_id}"
        self.redis_client.setex(workflow_key, 7200, workflow.model_dump_json())
    
    async def update_workflow(self, workflow: Workflow):
        """Update workflow in storage"""
        workflow.updated_at = datetime.now()
        await self.store_workflow(workflow)
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow status"""
        workflow_data = self.redis_client.get(f"workflow:{workflow_id}")
        if workflow_data:
            workflow = Workflow.model_validate_json(workflow_data)
            return {
                'workflow_id': workflow.workflow_id,
                'status': workflow.status.value,
                'title': workflow.title,
                'progress': {
                    'completed_steps': len([s for s in workflow.steps if s.status == TaskStatus.COMPLETED]),
                    'total_steps': len(workflow.steps),
                    'current_step': next((s.agent_name for s in workflow.steps if s.status == TaskStatus.IN_PROGRESS), None)
                },
                'created_at': workflow.created_at.isoformat(),
                'updated_at': workflow.updated_at.isoformat()
            }
        return None

# FastAPI application
app = FastAPI(title="Ambrosio Orchestrator", version="1.0.0")
orchestrator = OrchestoratorService()

@app.on_event("startup")
async def startup_event():
    """Start background tasks on app startup"""
    import asyncio
    asyncio.create_task(process_redis_messages())

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_workflows": len(orchestrator.active_workflows_dict)
    }

@app.get("/workflow/{workflow_id}/status")
async def get_workflow_status(workflow_id: str):
    """Get workflow status"""
    status = await orchestrator.get_workflow_status(workflow_id)
    if not status:
        return {"error": "Workflow not found"}
    return status

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Background task processing
async def process_redis_messages():
    """Process incoming Redis pub/sub messages"""
    logger.info("Starting Redis message processing")
    
    # Initialize orchestrator
    await orchestrator.initialize()
    logger.info("Orchestrator initialized")
    
    while True:
        try:
            # Get messages from Redis pub/sub
            message = orchestrator.pubsub.get_message(timeout=1.0)
            
            if message and message['type'] == 'message':
                channel = message['channel'].decode('utf-8')
                data = json.loads(message['data'].decode('utf-8'))
                
                logger.info("Received Redis message", type=data.get('type'), channel=channel)
                
                if channel == 'orchestrator_requests' and data.get('type') == 'task_request':
                    # Process task request asynchronously
                    asyncio.create_task(orchestrator.process_task_request(data['payload']))
                
            # Small delay to prevent busy waiting
            await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.error("Failed to process Redis messages", error=str(e))
            await asyncio.sleep(5)  # Wait before retrying

def main():
    """Main function"""
    logger.info("Starting Ambrosio Orchestrator Service")
    
    # Start FastAPI server (background task will be started via app startup event)
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )

if __name__ == "__main__":
    main()