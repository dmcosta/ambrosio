#!/usr/bin/env python3
"""
Ambrosio Configuration Manager Service
Handles dynamic agent configuration, templates, and runtime behavior management
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path

import redis
import yaml
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import structlog
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import uvicorn

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

# Prometheus metrics - clear registry first to avoid duplicates
from prometheus_client import REGISTRY, CollectorRegistry
REGISTRY._collector_to_names.clear()
REGISTRY._names_to_collectors.clear()

config_requests = Counter('config_requests_total', 'Total configuration requests', ['agent_type', 'operation'])
config_cache_hits = Counter('config_cache_hits_total', 'Configuration cache hits', ['agent_type'])
config_cache_misses = Counter('config_cache_misses_total', 'Configuration cache misses', ['agent_type'])
config_update_duration = Histogram('config_update_duration_seconds', 'Configuration update duration')
active_agents = Gauge('active_agents_total', 'Number of active agents', ['agent_type'])

# Pydantic models
class AgentConfig(BaseModel):
    name: str
    role: str
    persona: str
    behavior: Dict[str, Any]
    model_config: Dict[str, Any]
    tools: List[str]
    output_format: Dict[str, Any]
    memory: Dict[str, Any]
    constraints: Dict[str, Any] = Field(default_factory=dict)

class TaskComplexityLevel(BaseModel):
    level: str = Field(..., pattern="^(low|medium|high|critical)$")
    description: str
    model_overrides: Dict[str, Any] = Field(default_factory=dict)

class ConfigurationManager:
    """Manages dynamic agent configurations and templates"""
    
    def __init__(self):
        self.redis_client = redis.Redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))
        self.config_cache = {}
        self.templates_path = Path('/app/config')
        self.last_reload = datetime.now()
        
        # Load initial configurations - will be done async after startup
        
    async def load_all_configs(self):
        """Load all agent configurations from templates"""
        try:
            templates_dir = Path('/app/config')
            if not templates_dir.exists():
                logger.warning("Config templates directory not found", path=str(templates_dir))
                return
                
            for config_file in templates_dir.glob('*.yaml'):
                if config_file.stem.endswith('-config'):
                    agent_name = config_file.stem.replace('-config', '')
                    await self.load_agent_config(agent_name, config_file)
                    
            logger.info("All agent configurations loaded", count=len(self.config_cache))
            
        except Exception as e:
            logger.error("Failed to load configurations", error=str(e))
    
    async def load_agent_config(self, agent_name: str, config_file: Path):
        """Load individual agent configuration"""
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            agent_config = AgentConfig(**config_data['agent_config'])
            
            # Store in cache and Redis
            self.config_cache[agent_name] = agent_config
            await self.store_config_in_redis(agent_name, agent_config)
            
            logger.info("Agent configuration loaded", agent=agent_name, file=str(config_file))
            
        except Exception as e:
            logger.error("Failed to load agent config", agent=agent_name, error=str(e))
    
    async def store_config_in_redis(self, agent_name: str, config: AgentConfig):
        """Store configuration in Redis with TTL"""
        try:
            config_json = config.model_dump_json()
            self.redis_client.setex(f"agent_config:{agent_name}", 3600, config_json)
            
            # Store metadata
            metadata = {
                'last_updated': datetime.now().isoformat(),
                'version': 1,
                'active': True
            }
            self.redis_client.setex(f"agent_metadata:{agent_name}", 3600, json.dumps(metadata))
            
        except Exception as e:
            logger.error("Failed to store config in Redis", agent=agent_name, error=str(e))
    
    async def get_agent_config(self, agent_name: str, task_complexity: str = "medium") -> Optional[AgentConfig]:
        """Get agent configuration with complexity-based model selection"""
        config_requests.labels(agent_type=agent_name, operation='get').inc()
        
        try:
            # Try cache first
            if agent_name in self.config_cache:
                config_cache_hits.labels(agent_type=agent_name).inc()
                base_config = self.config_cache[agent_name]
            else:
                # Try Redis
                config_data = self.redis_client.get(f"agent_config:{agent_name}")
                if config_data:
                    config_cache_hits.labels(agent_type=agent_name).inc()
                    base_config = AgentConfig.model_validate_json(config_data)
                    self.config_cache[agent_name] = base_config
                else:
                    config_cache_misses.labels(agent_type=agent_name).inc()
                    logger.warning("Agent configuration not found", agent=agent_name)
                    return None
            
            # Apply complexity-based model selection
            enhanced_config = await self.apply_complexity_overrides(base_config, task_complexity)
            
            return enhanced_config
            
        except Exception as e:
            logger.error("Failed to get agent config", agent=agent_name, error=str(e))
            return None
    
    async def apply_complexity_overrides(self, config: AgentConfig, complexity: str) -> AgentConfig:
        """Apply model overrides based on task complexity"""
        if complexity == "high" or complexity == "critical":
            # Use more powerful models for complex tasks
            if config.name == "coder":
                config.model_config["primary_model"] = "claude-sonnet-4-20250514"
                config.model_config["primary_provider"] = "claude"
                config.model_config["temperature"] = 0.1  # More focused
                config.model_config["max_tokens"] = 4096
                
            elif config.name == "planner":
                config.model_config["primary_model"] = "gemini-2.0-flash"
                config.model_config["temperature"] = 0.3
                
        elif complexity == "low":
            # Use efficient local models for simple tasks
            if config.name in ["coder", "reviewer", "planner"]:
                config.model_config["primary_provider"] = "ollama"
                config.model_config["temperature"] = 0.7  # More creative for simple tasks
        
        return config
    
    async def update_agent_behavior(self, agent_name: str, behavior_updates: Dict[str, Any]) -> bool:
        """Update agent behavior at runtime"""
        with config_update_duration.time():
            try:
                config = await self.get_agent_config(agent_name)
                if not config:
                    return False
                
                # Update behavior
                config.behavior.update(behavior_updates)
                
                # Store updated configuration
                await self.store_config_in_redis(agent_name, config)
                self.config_cache[agent_name] = config
                
                config_requests.labels(agent_type=agent_name, operation='update').inc()
                
                logger.info("Agent behavior updated", agent=agent_name, updates=behavior_updates)
                return True
                
            except Exception as e:
                logger.error("Failed to update agent behavior", agent=agent_name, error=str(e))
                return False
    
    async def get_all_active_agents(self) -> List[str]:
        """Get list of all active agents"""
        try:
            agent_keys = self.redis_client.keys("agent_metadata:*")
            active_agents_list = []
            
            for key in agent_keys:
                metadata = json.loads(self.redis_client.get(key))
                if metadata.get('active', False):
                    agent_name = key.decode('utf-8').replace('agent_metadata:', '')
                    active_agents_list.append(agent_name)
                    active_agents.labels(agent_type=agent_name).set(1)
            
            return active_agents_list
            
        except Exception as e:
            logger.error("Failed to get active agents", error=str(e))
            return []
    
    async def reload_configurations(self):
        """Reload all configurations from disk"""
        try:
            self.config_cache.clear()
            await self.load_all_configs()
            self.last_reload = datetime.now()
            logger.info("Configurations reloaded successfully")
            return True
        except Exception as e:
            logger.error("Failed to reload configurations", error=str(e))
            return False

# FastAPI application
app = FastAPI(title="Ambrosio Configuration Manager", version="1.0.0")
config_manager = ConfigurationManager()

@app.on_event("startup")
async def startup_event():
    """Load configurations on startup"""
    await config_manager.load_all_configs()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Redis connection
        config_manager.redis_client.ping()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "redis": "connected",
                "config_cache": len(config_manager.config_cache)
            }
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.get("/config/{agent_name}")
async def get_agent_configuration(agent_name: str, complexity: str = "medium"):
    """Get agent configuration with complexity-based optimization"""
    config = await config_manager.get_agent_config(agent_name, complexity)
    if not config:
        raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")
    
    return {
        "agent": agent_name,
        "complexity": complexity,
        "config": config.model_dump(),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/config/{agent_name}/behavior")
async def update_agent_behavior(agent_name: str, behavior_updates: Dict[str, Any]):
    """Update agent behavior at runtime"""
    success = await config_manager.update_agent_behavior(agent_name, behavior_updates)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to update agent behavior")
    
    return {
        "message": f"Agent {agent_name} behavior updated successfully",
        "updates": behavior_updates,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/agents")
async def list_active_agents():
    """List all active agents"""
    agents = await config_manager.get_all_active_agents()
    return {
        "active_agents": agents,
        "count": len(agents),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/reload")
async def reload_configurations(background_tasks: BackgroundTasks):
    """Reload all configurations from disk"""
    background_tasks.add_task(config_manager.reload_configurations)
    return {
        "message": "Configuration reload initiated",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/status")
async def get_system_status():
    """Get comprehensive system status"""
    return {
        "service": "config-manager",
        "version": "1.0.0",
        "uptime": datetime.now() - config_manager.last_reload,
        "cache_size": len(config_manager.config_cache),
        "last_reload": config_manager.last_reload.isoformat(),
        "redis_connected": config_manager.redis_client.ping(),
        "active_agents": await config_manager.get_all_active_agents()
    }

if __name__ == "__main__":
    logger.info("Starting Ambrosio Configuration Manager")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        log_level="info",
        access_log=True,
        reload=False
    )