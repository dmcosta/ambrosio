#!/usr/bin/env python3
"""
Ambrosio Discord Bot Service
Provides Discord interface for the AI agent system
"""

import os
import json
import asyncio
import uuid
from typing import Dict, Any, Optional
from datetime import datetime

import discord
from discord.ext import commands, tasks
import redis
from pydantic import BaseModel
import structlog
from prometheus_client import Counter, Histogram, start_http_server

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
command_requests = Counter('discord_commands_total', 'Total Discord commands', ['command', 'user'])
message_responses = Counter('discord_responses_total', 'Total Discord responses', ['type'])
task_creations = Counter('discord_tasks_created_total', 'Tasks created via Discord', ['task_type'])
response_time = Histogram('discord_response_time_seconds', 'Discord command response time')

class TaskRequest(BaseModel):
    task_id: str
    user_id: str
    user_name: str
    channel_id: str
    guild_id: str
    task_type: str
    title: str
    description: str
    repository: str = None
    branch: str = None
    priority: str = "medium"
    complexity: str = "medium"
    metadata: Dict[str, Any] = {}

class AmbrosioBot(commands.Bot):
    """Ambrosio Discord Bot - AI Development Assistant"""
    
    def __init__(self):
        intents = discord.Intents.default()
        # Only enable non-privileged intents for slash commands
        intents.guilds = True  # Non-privileged - needed for server events
        
        super().__init__(
            command_prefix='/',
            intents=intents,
            description='Ambrosio - Your AI Development Team Assistant'
        )
        
        # Initialize connections
        self.redis_client = redis.Redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))
        
        # Task tracking
        self.active_tasks = {}
        self.user_sessions = {}
        
    async def on_ready(self):
        """Called when the bot is ready"""
        logger.info("Ambrosio bot ready", guilds=len(self.guilds), users=len(set(self.get_all_members())))
        
        # Start background tasks
        if not self.status_updater.is_running():
            self.status_updater.start()
        
        # Set bot status
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="for /ambrosio commands"
            )
        )
        
        # Sync slash commands
        try:
            synced = await self.tree.sync()
            logger.info("Slash commands synced", count=len(synced))
        except Exception as e:
            logger.error("Failed to sync commands", error=str(e))
    
    async def on_command_error(self, ctx, error):
        """Handle command errors"""
        logger.error("Command error", command=ctx.command, error=str(error), user=str(ctx.author))
        
        if isinstance(error, commands.CommandNotFound):
            await ctx.send("❌ Command not found. Use `/ambrosio help` to see available commands.")
        elif isinstance(error, commands.MissingRequiredArgument):
            await ctx.send(f"❌ Missing required argument: `{error.param.name}`")
        else:
            await ctx.send("❌ An error occurred while processing your command. Please try again.")
    
    async def send_task_to_orchestrator(self, task_request: TaskRequest):
        """Send task request to orchestrator via Redis pub/sub"""
        try:
            message = {
                'type': 'task_request',
                'task_id': task_request.task_id,
                'payload': task_request.model_dump(),
                'timestamp': datetime.now().isoformat()
            }
            
            # Send to Redis pub/sub
            self.redis_client.publish('orchestrator_requests', json.dumps(message))
            
            # Store task in Redis for tracking
            self.redis_client.setex(
                f"discord_task:{task_request.task_id}",
                3600,  # 1 hour TTL
                task_request.model_dump_json()
            )
            
            task_creations.labels(task_type=task_request.task_type).inc()
            logger.info("Task sent to orchestrator via Redis", task_id=task_request.task_id, type=task_request.task_type)
            
        except Exception as e:
            logger.error("Failed to send task to orchestrator", task_id=task_request.task_id, error=str(e))
            raise
    
    @tasks.loop(minutes=5)
    async def status_updater(self):
        """Update bot status periodically"""
        try:
            active_count = len(self.active_tasks)
            activity_text = f"{active_count} active tasks" if active_count > 0 else "for /ambrosio commands"
            
            await self.change_presence(
                activity=discord.Activity(
                    type=discord.ActivityType.watching,
                    name=activity_text
                )
            )
        except Exception as e:
            logger.error("Failed to update status", error=str(e))

# Create bot instance
bot = AmbrosioBot()

@bot.tree.command(name="create-feature", description="Create a new feature implementation task")
async def create_feature(
    interaction: discord.Interaction,
    description: str,
    repository: str,
    priority: str = "medium"
):
    """Create a new feature implementation task"""
    command_requests.labels(command='create-feature', user=str(interaction.user)).inc()
    
    with response_time.time():
        try:
            await interaction.response.defer()
            
            task_id = str(uuid.uuid4())
            
            task_request = TaskRequest(
                task_id=task_id,
                user_id=str(interaction.user.id),
                user_name=str(interaction.user),
                channel_id=str(interaction.channel.id),
                guild_id=str(interaction.guild.id) if interaction.guild else "DM",
                task_type="feature_creation",
                title=f"Feature: {description[:100]}...",
                description=description,
                repository=repository,
                priority=priority,
                complexity="medium",
                metadata={
                    'source': 'discord',
                    'command': 'create-feature'
                }
            )
            
            await bot.send_task_to_orchestrator(task_request)
            bot.active_tasks[task_id] = task_request
            
            embed = discord.Embed(
                title="🚀 Feature Creation Started",
                description=f"Task ID: `{task_id}`",
                color=discord.Color.green(),
                timestamp=datetime.now()
            )
            embed.add_field(name="Description", value=description[:1000], inline=False)
            embed.add_field(name="Repository", value=repository, inline=True)
            embed.add_field(name="Priority", value=priority.title(), inline=True)
            embed.add_field(name="Status", value="🔄 Processing", inline=True)
            embed.set_footer(text="Ambrosio will update you on progress")
            
            await interaction.followup.send(embed=embed)
            
            logger.info("Feature creation task created", task_id=task_id, user=str(interaction.user))
            
        except Exception as e:
            logger.error("Failed to create feature task", error=str(e))
            await interaction.followup.send("❌ Failed to create feature task. Please try again.")

@bot.tree.command(name="review-pr", description="Review a GitHub pull request")
async def review_pr(
    interaction: discord.Interaction,
    pr_url: str,
    focus: str = "general"
):
    """Review a GitHub pull request"""
    command_requests.labels(command='review-pr', user=str(interaction.user)).inc()
    
    with response_time.time():
        try:
            await interaction.response.defer()
            
            # Extract repository from PR URL
            parts = pr_url.replace('https://github.com/', '').split('/')
            if len(parts) < 4 or parts[2] != 'pull':
                await interaction.followup.send("❌ Invalid PR URL format. Use: https://github.com/owner/repo/pull/123")
                return
                
            repository = f"{parts[0]}/{parts[1]}"
            pr_number = parts[3]
            
            task_id = str(uuid.uuid4())
            
            task_request = TaskRequest(
                task_id=task_id,
                user_id=str(interaction.user.id),
                user_name=str(interaction.user),
                channel_id=str(interaction.channel.id),
                guild_id=str(interaction.guild.id) if interaction.guild else "DM",
                task_type="code_review",
                title=f"PR Review: {repository}#{pr_number}",
                description=f"Review pull request with focus on: {focus}",
                repository=repository,
                complexity="medium",
                metadata={
                    'source': 'discord',
                    'command': 'review-pr',
                    'pr_url': pr_url,
                    'pr_number': pr_number,
                    'focus': focus
                }
            )
            
            await bot.send_task_to_orchestrator(task_request)
            bot.active_tasks[task_id] = task_request
            
            embed = discord.Embed(
                title="🔍 PR Review Started",
                description=f"Task ID: `{task_id}`",
                color=discord.Color.blue(),
                timestamp=datetime.now()
            )
            embed.add_field(name="Pull Request", value=f"[{repository}#{pr_number}]({pr_url})", inline=False)
            embed.add_field(name="Focus Area", value=focus.title(), inline=True)
            embed.add_field(name="Status", value="🔄 Analyzing", inline=True)
            embed.set_footer(text="Review will be posted as GitHub comments")
            
            await interaction.followup.send(embed=embed)
            
        except Exception as e:
            logger.error("Failed to create PR review task", error=str(e))
            await interaction.followup.send("❌ Failed to start PR review. Please check the URL and try again.")

@bot.tree.command(name="status", description="Check status of a task or system")
async def status(
    interaction: discord.Interaction,
    task_id: str = None
):
    """Check task or system status"""
    command_requests.labels(command='status', user=str(interaction.user)).inc()
    
    try:
        await interaction.response.defer()
        
        if task_id:
            # Check specific task status
            task_data = bot.redis_client.get(f"discord_task:{task_id}")
            if not task_data:
                await interaction.followup.send("❌ Task not found or expired.")
                return
            
            task_info = json.loads(task_data)
            
            embed = discord.Embed(
                title=f"📋 Task Status: {task_id[:8]}...",
                color=discord.Color.blue(),
                timestamp=datetime.now()
            )
            embed.add_field(name="Type", value=task_info.get('task_type', 'Unknown'), inline=True)
            embed.add_field(name="Repository", value=task_info.get('repository', 'N/A'), inline=True)
            embed.add_field(name="Priority", value=task_info.get('priority', 'medium').title(), inline=True)
            embed.add_field(name="Description", value=task_info.get('description', 'No description')[:500], inline=False)
            
            await interaction.followup.send(embed=embed)
            
        else:
            # Show system status
            active_count = len(bot.active_tasks)
            
            embed = discord.Embed(
                title="🤖 Ambrosio System Status",
                color=discord.Color.green(),
                timestamp=datetime.now()
            )
            embed.add_field(name="Active Tasks", value=str(active_count), inline=True)
            embed.add_field(name="Connected Guilds", value=str(len(bot.guilds)), inline=True)
            embed.add_field(name="Uptime", value="✅ Online", inline=True)
            embed.add_field(name="Agent Services", value="🔄 Checking...", inline=False)
            embed.set_footer(text="Use /ambrosio help for available commands")
            
            await interaction.followup.send(embed=embed)
            
    except Exception as e:
        logger.error("Failed to get status", error=str(e))
        await interaction.followup.send("❌ Failed to retrieve status. Please try again.")

@bot.tree.command(name="help", description="Show available Ambrosio commands")
async def help_command(interaction: discord.Interaction):
    """Show help information"""
    command_requests.labels(command='help', user=str(interaction.user)).inc()
    
    embed = discord.Embed(
        title="🤖 Ambrosio - AI Development Team Assistant",
        description="Your intelligent development companion powered by specialized AI agents.",
        color=discord.Color.purple(),
        timestamp=datetime.now()
    )
    
    embed.add_field(
        name="🚀 `/create-feature`",
        value="Create a new feature implementation task\n`description` `repository` `[priority]`",
        inline=False
    )
    
    embed.add_field(
        name="🔍 `/review-pr`", 
        value="Review a GitHub pull request\n`pr_url` `[focus]`",
        inline=False
    )
    
    embed.add_field(
        name="📋 `/status`",
        value="Check task or system status\n`[task_id]`",
        inline=False
    )
    
    embed.add_field(
        name="💡 Examples",
        value=(
            "`/create-feature description:Add user auth repository:myorg/myapp priority:high`\n"
            "`/review-pr pr_url:https://github.com/owner/repo/pull/123 focus:security`\n"
            "`/status task_id:12345678`"
        ),
        inline=False
    )
    
    embed.set_footer(text="More commands coming soon!")
    
    await interaction.response.send_message(embed=embed)

async def main():
    """Main function to start the bot"""
    # Start Prometheus metrics server
    start_http_server(8080)
    
    logger.info("Starting Ambrosio Discord Bot")
    
    try:
        await bot.start(os.getenv('DISCORD_TOKEN'))
    except discord.LoginFailure:
        logger.error("Invalid Discord token")
    except Exception as e:
        logger.error("Bot startup failed", error=str(e))

if __name__ == "__main__":
    asyncio.run(main())