import os
import sys
import time
import re
import asyncio
import datetime
from typing import Dict, Any

import modal
from fastapi import FastAPI

# Modal app setup
app = modal.App("discord-chat-bot")

# Mount the src directory and data directory properly
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "discord.py==2.5.2",
    "openai>=1.0.0",
    "chromadb>=0.4.0", 
    "tiktoken>=0.5.0",
    "python-dotenv>=1.0.0",
    "fastapi>=0.100.0",
    "uvicorn>=0.20.0",
    "pandas==2.2.3",
    "numpy>=1.24.0",
    "pynacl>=1.5.0",
    "requests>=2.31.0"
).add_local_dir("src", "/root/src").add_local_dir("data", "/root/data")

# Define secrets
secrets = [
    modal.Secret.from_name("openai-secret"),
    modal.Secret.from_name("discord-secret-2"),
]

# Define Modal volumes for persistence
discord_bot_volume = modal.Volume.from_name("discord-bot-volume-3", create_if_missing=True)
chroma_volume = modal.Volume.from_name("chroma-db-volume-3", create_if_missing=True)

volume_mounts = {
    "/data/db": discord_bot_volume,
    "/root/chroma_db": chroma_volume
}

def setup_python_path():
    """Ensure the src directory is in Python path"""
    import sys
    if "/root/src" not in sys.path:
        sys.path.insert(0, "/root/src")

def bot_is_mentioned(content: str, client_user) -> bool:
    """Checks if the bot is mentioned or addressed in the message content."""
    return (
        client_user.mention in content
        or re.search(r"\bbot\b", content, re.IGNORECASE) is not None
    )

@app.function(
    image=image,
    secrets=secrets,
    volumes=volume_mounts,
    timeout=300
)
def fetch_api(question: str) -> Dict[str, Any]:
    """Get answer from OpenAI using context from vector database"""
    # Add /root to Python path so imports work
    import sys
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    
    # Import dependencies (should work now)
    from database import init_db, log_track_interaction
    from vector_emb import answer_question, llm_answer_question, get_openai_client
    
    start_time = time.time()
    
    try:
        # Initialize database
        init_db()
        
        # Get context and answer
        context, sources, chunks = answer_question(question)
        client = get_openai_client()
        response, context_info = llm_answer_question(client, context, sources, chunks, question)
        
        end_time = time.time()
        
        # Log the interaction
        log_id = log_track_interaction(
            question=question,
            response=response,
            context_info=context_info,
            model="gpt-4o-mini",
            start_time=start_time,
            end_time=end_time,
            success=True
        )
        
        return {
            "answer": response,
            "log_id": log_id,
            "context_info": context_info
        }
        
    except Exception as e:
        end_time = time.time()
        error_msg = f"Error generating response: {str(e)}"
        print(f"❌ Error in fetch_api: {e}")
        print(f"❌ Python path: {sys.path}")
        
        # Log failed interaction
        try:
            log_track_interaction(
                question=question,
                response=error_msg,
                context_info={"error": str(e)},
                model="gpt-4o-mini",
                start_time=start_time,
                end_time=end_time,
                success=False
            )
        except:
            pass
        
        return {"answer": error_msg, "log_id": None, "context_info": {}}

@app.function(image=image, timeout=10)
def health_check():
    """Simple health check function"""
    import sys
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    
    try:
        # Test imports
        from database import init_db
        from vector_emb import get_openai_client
        from utils import count_tokens
        from config import CHUNK_SIZE
        
        return {
            "status": "healthy", 
            "timestamp": datetime.datetime.now().isoformat(),
            "imports": "success",
            "python_path": sys.path[:3]  # First 3 paths
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "python_path": sys.path[:3]
        }

@app.function(
    image=image,
    secrets=secrets,
    volumes=volume_mounts,
    timeout=30
)
def store_user_feedback(log_id: str, feedback_type: str, user_id: str):
    """Store user feedback for a specific log entry"""
    import sys
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    
    from database import log_track_feedback
    
    try:
        if feedback_type == "👍":
            rating = "positive"
            reason = "helpful"
        elif feedback_type == "👎":
            rating = "negative"
            reason = "not_helpful"
        else:
            rating = "neutral"
            reason = "other"
            
        success = log_track_feedback(
            log_id=log_id,
            rating=rating,
            reason=reason,
            notes="",
            user=str(user_id)
        )
        
        return success
        
    except Exception as e:
        print(f"❌ Error storing feedback: {str(e)}")
        return False

# Global to track bot state
message_log_mapping = {}

@app.function(
    image=image,
    secrets=secrets,
    volumes=volume_mounts,
    timeout=3600,  # 1 hour
    schedule=modal.Period(minutes=50)  # Restart every 50 minutes
)
async def discord_bot_runner():
    """Main Discord bot function with auto-restart"""
    import sys
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    
    import discord
    from discord.ext import commands
    
    print(f"🚀 Starting Discord bot at {datetime.datetime.now()}")
    
    def setup_discord_bot():
        intents = discord.Intents.default()
        intents.message_content = True
        intents.reactions = True
        bot = commands.Bot(command_prefix="!", intents=intents)

        @bot.event
        async def on_ready():
            print(f"✅ Bot ready! Logged in as {bot.user.name}")
            print(f"Connected to {len(bot.guilds)} guilds")

        @bot.event
        async def on_message(message):
            if message.author == bot.user:
                return

            if bot_is_mentioned(message.content, bot.user):
                print(f"📝 Question received: {message.content}")
                thread = await message.create_thread(
                    name=f"Question from {message.author.display_name}",
                    auto_archive_duration=60,
                )
                await thread.send(f"Hey {message.author.mention}, let me think about that...")
                
                try:
                    result = fetch_api.remote(message.content)
                    answer = result["answer"]
                    log_id = result["log_id"]
                    
                    await thread.send(f"**Answer:** {answer}")
                    feedback_msg = await thread.send(
                        "Was this answer helpful? 👍 for yes, 👎 for no"
                    )
                    await feedback_msg.add_reaction("👍")
                    await feedback_msg.add_reaction("👎")
                    
                    message_log_mapping[feedback_msg.id] = log_id
                    
                except Exception as e:
                    await thread.send(f"Sorry, I encountered an error: {str(e)}")
                    print(f"❌ Error processing question: {e}")

            await bot.process_commands(message)

        @bot.event
        async def on_reaction_add(reaction, user):
            if user.bot:
                return
                
            if reaction.message.id in message_log_mapping:
                log_id = message_log_mapping[reaction.message.id]
                feedback_emoji = str(reaction.emoji)
                
                if feedback_emoji in ["👍", "👎"]:
                    try:
                        success = store_user_feedback.remote(log_id, feedback_emoji, user.id)
                        if success:
                            await reaction.message.reply(f"Thank you {user.mention}! Feedback recorded. 🙏")
                    except Exception as e:
                        print(f"❌ Error processing feedback: {str(e)}")

        return bot
    
    try:
        discord_token = os.environ.get("DISCORD_BOT_TOKEN")
        if not discord_token:
            raise ValueError("DISCORD_BOT_TOKEN not found in environment")
        
        client = setup_discord_bot()
        await client.start(discord_token)

    except Exception as e:
        print(f"❌ Bot error: {e}")
        if 'client' in locals() and not client.is_closed():
            await client.close()

@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    """FastAPI endpoints for monitoring"""
    web_app = FastAPI(title="Discord Chat Bot API")
    
    @web_app.get("/health")
    def health():
        return health_check.remote()
    
    @web_app.get("/test") 
    def test_api():
        result = fetch_api.remote("What is machine learning?")
        return {"test_result": result}
    
    return web_app

@app.local_entrypoint()
def main():
    """Entry point for testing"""
    print("🚀 Discord Chat Bot - Modal Deployment")
    print("Use 'modal deploy deploy_discord_bot.py' to deploy")
    
    # Test health check
    result = health_check.remote()
    print(f"Health check: {result}")

if __name__ == "__main__":
    main()
